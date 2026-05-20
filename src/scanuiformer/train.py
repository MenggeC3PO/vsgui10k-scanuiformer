import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

from dataset import (
    read_jsonl_gz,
    split_trials,
    build_trial_index,
    build_cue_vocab,
    build_seg_index,
    build_ui_type_vocab,
    build_hybrid_trial_samples,
    check_hybrid_trial_samples,
    HybridTrialDecoderTargetCropDataset,
    filter_target_present_trials,
    save_json,
)
from model import HybridTemporalTargetDurationDecoderModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def get_first(d: Dict[str, Any], keys: List[str], default=None):
    """Return the first non-empty value among possible key names."""
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return default


def image_stem_from_trial(t: Dict[str, Any]):
    v = get_first(t, [
        "image", "image_path", "img", "img_path", "screenshot", "screenshot_path",
        "filename", "image_name", "screenshot_name"
    ])
    if v is None:
        # Some VSGUI versions store image id directly.
        v = get_first(t, ["image_id", "img_id", "screenshot_id"])
    if v is None:
        return None
    return Path(str(v)).stem


def make_seekui_like_id(t: Dict[str, Any]):
    """Build the SeekUI released-test id from this project's trial format.

    Our preprocessed trials store the needed fields as:
      trial["key"]["img_name"] -> e.g. "00b3a0.png"
      trial["key"]["pid"]      -> e.g. "3b43f3"
      trial["key"]["tgt_id"]   -> e.g. "JX3WsRW113"

    SeekUI uses ids like:
      00b3a0_3b43f3_txt_JX3WsRW113

    The middle token is the participant id and "txt" is the released
    text-target prefix, not the raw cue field from our trial_id.
    """
    k = t.get("key", {}) if isinstance(t, dict) else {}
    img_name = k.get("img_name", None)
    pid = k.get("pid", None)
    tgt_id = k.get("tgt_id", None)

    if img_name is None or pid is None or tgt_id is None:
        return None

    img_stem = Path(str(img_name)).stem
    return f"{img_stem}_{pid}_txt_{tgt_id}"


def load_id_set(path: str):
    if path is None or str(path).strip() == "":
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"seekui_test_ids_path not found: {path}")
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def split_trials_with_seekui_holdout(trials, cfg):
    """Split trials while forcing released SeekUI test IDs out of train/val.

    Modes:
    - force_seekui_test_as_test=True and include_random_test_with_seekui_holdout=False:
      train/val are made from all non-SeekUI trials; test is exactly matched SeekUI IDs.
    - include_random_test_with_seekui_holdout=True:
      normal random test from remaining data is appended to SeekUI holdout.
    """
    seekui_ids = load_id_set(cfg.get("seekui_test_ids_path", ""))
    if not seekui_ids:
        train_trials, val_trials, test_trials = split_trials(
            trials, cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
        )
        return train_trials, val_trials, test_trials, {
            "seekui_holdout_enabled": False,
            "seekui_test_ids_loaded": 0,
        }

    holdout, remaining = [], []
    missing_sid = 0
    matched_ids = set()
    for t in trials:
        sid = make_seekui_like_id(t)
        if sid is None:
            missing_sid += 1
            remaining.append(t)
            continue
        if sid in seekui_ids:
            holdout.append(t)
            matched_ids.add(sid)
        else:
            remaining.append(t)

    force_as_test = bool(cfg.get("force_seekui_test_as_test", True))
    include_random_test = bool(cfg.get("include_random_test_with_seekui_holdout", False))

    if force_as_test and not include_random_test:
        rng = random.Random(int(cfg["seed"]))
        rem = list(remaining)
        rng.shuffle(rem)
        train_ratio = float(cfg.get("train_ratio", 0.8))
        val_ratio = float(cfg.get("val_ratio", 0.1))
        denom = max(1e-8, train_ratio + val_ratio)
        n_train = int(len(rem) * train_ratio / denom)
        train_trials = rem[:n_train]
        val_trials = rem[n_train:]
        test_trials = holdout
    else:
        train_trials, val_trials, random_test = split_trials(
            remaining, cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
        )
        test_trials = holdout + (random_test if include_random_test else [])

    train_ids = {make_seekui_like_id(t) for t in train_trials}
    val_ids = {make_seekui_like_id(t) for t in val_trials}
    test_ids = {make_seekui_like_id(t) for t in test_trials}

    overlap_train = sorted(seekui_ids & train_ids)
    overlap_val = sorted(seekui_ids & val_ids)
    overlap_test = sorted(seekui_ids & test_ids)

    meta = {
        "seekui_holdout_enabled": True,
        "seekui_test_ids_loaded": len(seekui_ids),
        "seekui_ids_matched_in_trials": len(matched_ids),
        "seekui_ids_missing_from_trials": len(seekui_ids - matched_ids),
        "trials_missing_seekui_like_id": missing_sid,
        "holdout_trials": len(holdout),
        "remaining_trials": len(remaining),
        "force_seekui_test_as_test": force_as_test,
        "include_random_test_with_seekui_holdout": include_random_test,
        "seekui_test_intersection_train": len(overlap_train),
        "seekui_test_intersection_val": len(overlap_val),
        "seekui_test_intersection_test": len(overlap_test),
        "seekui_test_train_examples": overlap_train[:10],
        "seekui_test_val_examples": overlap_val[:10],
        "seekui_test_test_examples": overlap_test[:10],
    }
    if overlap_train or overlap_val:
        raise RuntimeError(
            "SeekUI test leakage detected after split: "
            f"train_overlap={len(overlap_train)}, val_overlap={len(overlap_val)}"
        )
    return train_trials, val_trials, test_trials, meta


def filter_trials_by_cue(trials, cfg):
    allowed = cfg.get("target_cue_filter", None)
    if allowed is None or allowed == [] or allowed == "":
        return trials
    if isinstance(allowed, str):
        allowed_set = {allowed}
    else:
        allowed_set = {str(x) for x in allowed}
    out = []
    for t in trials:
        cue = str(t.get("key", {}).get("cue", ""))
        if cue in allowed_set:
            out.append(t)
    print(f"target_cue_filter = {sorted(allowed_set)} | kept {len(out)}/{len(trials)} trials")
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=None, help="Optional override for ui_memory_scale")
    parser.add_argument("--freeze_patch", action="store_true")
    parser.add_argument("--no_freeze_patch", action="store_true")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def euclidean_dist(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=-1) + 1e-12)


def require_finite(name: str, x: torch.Tensor, context: str = ""):
    if not torch.isfinite(x).all():
        msg = f"Non-finite tensor detected: {name}"
        if context:
            msg += f" | {context}"
        finite = torch.isfinite(x)
        try:
            msg += (
                f" | shape={tuple(x.shape)} "
                f"min={float(torch.nan_to_num(x).min().item())} "
                f"max={float(torch.nan_to_num(x).max().item())} "
                f"finite_ratio={float(finite.float().mean().item())}"
            )
        except Exception:
            pass
        raise RuntimeError(msg)




def transform_duration_feature(dur: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """Transform raw fixation duration into the feature scale used by history/model.

    Default is log1p(dur / duration_scale), which is robust for long-tailed
    fixation durations. If your durations are already normalized, set
    duration_target="raw" and duration_scale=1.0 in the config.
    """
    dur = torch.nan_to_num(dur.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    scale = float(cfg.get("duration_scale", 1.0))
    scale = max(scale, 1e-8)
    mode = str(cfg.get("duration_target", "log1p")).lower()
    if mode in ["log1p", "log"]:
        return torch.log1p(dur / scale)
    if mode in ["sqrt"]:
        return torch.sqrt(dur / scale)
    if mode in ["raw", "linear", "none"]:
        return dur / scale
    raise ValueError(f"Unknown duration_target: {mode}")


def prepare_model_scanpath_xydur(scanpath_xydur: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    out = scanpath_xydur.clone()
    out[..., 2] = transform_duration_feature(scanpath_xydur[..., 2], cfg)
    max_feat = cfg.get("duration_feature_clip", None)
    if max_feat is not None:
        out[..., 2] = out[..., 2].clamp(0.0, float(max_feat))
    return out

def build_coord_criterion(cfg: Dict[str, Any]):
    coord_loss = cfg.get("coord_loss", "smooth_l1").lower()
    if coord_loss in ["smooth_l1", "huber"]:
        return nn.SmoothL1Loss(beta=float(cfg.get("smooth_l1_beta", 0.05)), reduction="none")
    if coord_loss == "mse":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"Unknown coord_loss: {coord_loss}")




def duration_loss_all(pred_dur: torch.Tensor, target_dur: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    loss_name = str(cfg.get("duration_loss", "smooth_l1")).lower()
    if loss_name in ["smooth_l1", "huber"]:
        return nn.functional.smooth_l1_loss(
            pred_dur,
            target_dur,
            beta=float(cfg.get("duration_smooth_l1_beta", 0.10)),
            reduction="none",
        )
    if loss_name == "mse":
        return nn.functional.mse_loss(pred_dur, target_dur, reduction="none")
    if loss_name == "l1":
        return nn.functional.l1_loss(pred_dur, target_dur, reduction="none")
    raise ValueError(f"Unknown duration_loss: {loss_name}")

def reduce_coord_loss(loss_raw: torch.Tensor) -> torch.Tensor:
    if loss_raw.ndim == 2:
        return loss_raw.mean(dim=1)
    return loss_raw


def expand_bbox_torch(bbox: torch.Tensor, margin: float) -> torch.Tensor:
    out = bbox.clone()
    out[:, 0] = (out[:, 0] - margin).clamp(0.0, 1.0)
    out[:, 1] = (out[:, 1] - margin).clamp(0.0, 1.0)
    out[:, 2] = (out[:, 2] + margin).clamp(0.0, 1.0)
    out[:, 3] = (out[:, 3] + margin).clamp(0.0, 1.0)
    return out


def point_inside_bbox(points_xy: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    x0 = bbox[..., 0]
    y0 = bbox[..., 1]
    x1 = bbox[..., 2]
    y1 = bbox[..., 3]
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def point_to_bbox_distance(points_xy: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """Differentiable distance from point to bbox, 0 if inside."""
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x0, y0, x1, y1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    dx_left = (x0 - x).clamp_min(0.0)
    dx_right = (x - x1).clamp_min(0.0)
    dy_top = (y0 - y).clamp_min(0.0)
    dy_bottom = (y - y1).clamp_min(0.0)
    dx = dx_left + dx_right
    dy = dy_top + dy_bottom
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def soft_ui_candidate_targets(ui_geom: torch.Tensor, ui_mask: torch.Tensor, target_bbox_norm: torch.Tensor, tau: float = 0.08) -> torch.Tensor:
    """Build soft target distribution over UI tokens for candidate supervision.

    Uses target bbox during training only. At inference the model predicts the
    candidate distribution from UI tokens + cue/target crop.
    """
    if ui_geom.size(-1) >= 10:
        centers = ui_geom[..., 4:6]
        x1 = ui_geom[..., 0]
        y1 = ui_geom[..., 1]
        x2 = ui_geom[..., 2]
        y2 = ui_geom[..., 3]
    else:
        centers = ui_geom[..., 0:2]
        w = ui_geom[..., 2].clamp_min(1e-6)
        h = ui_geom[..., 3].clamp_min(1e-6)
        x1 = centers[..., 0] - 0.5 * w
        y1 = centers[..., 1] - 0.5 * h
        x2 = centers[..., 0] + 0.5 * w
        y2 = centers[..., 1] + 0.5 * h

    tb = target_bbox_norm.to(ui_geom.device, ui_geom.dtype)
    tx1, ty1, tx2, ty2 = tb[:, 0:1], tb[:, 1:2], tb[:, 2:3], tb[:, 3:4]
    cx, cy = centers[..., 0], centers[..., 1]
    dx_out = torch.maximum(torch.maximum(tx1 - cx, cx - tx2), torch.zeros_like(cx))
    dy_out = torch.maximum(torch.maximum(ty1 - cy, cy - ty2), torch.zeros_like(cy))
    dist = torch.sqrt(dx_out * dx_out + dy_out * dy_out + 1e-8)

    inter_x1 = torch.maximum(x1, tx1)
    inter_y1 = torch.maximum(y1, ty1)
    inter_x2 = torch.minimum(x2, tx2)
    inter_y2 = torch.minimum(y2, ty2)
    inter = (inter_x2 - inter_x1).clamp_min(0.0) * (inter_y2 - inter_y1).clamp_min(0.0)
    area_ui = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)
    area_t = (tx2 - tx1).clamp_min(0.0) * (ty2 - ty1).clamp_min(0.0)
    iou = inter / (area_ui + area_t - inter + 1e-8)

    score = torch.exp(-dist / max(float(tau), 1e-6)) + 2.0 * iou
    score = score * (ui_mask > 0).float()
    denom = score.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return score / denom


def soft_candidate_ce_loss(candidate_logits: torch.Tensor, soft_targets: torch.Tensor, ui_mask: torch.Tensor) -> torch.Tensor:
    logits = candidate_logits.masked_fill(ui_mask <= 0, -1e9)
    logp = torch.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1)


def compute_first_hit_indices(scanpath_xydur: torch.Tensor, scanpath_mask: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """Return first index t where GT scanpath hits target bbox, else -1."""
    B, L, _ = scanpath_xydur.shape
    device = scanpath_xydur.device
    first = torch.full((B,), -1, dtype=torch.long, device=device)
    for t in range(L):
        active = scanpath_mask[:, t] > 0
        inside = point_inside_bbox(scanpath_xydur[:, t, :2], bbox) & active
        update = (first < 0) & inside
        first = torch.where(update, torch.full_like(first, t), first)
    return first


def make_target_stop_for_step(
    t: int,
    active: torch.Tensor,
    scanpath_len: torch.Tensor,
    is_truncated: torch.Tensor,
    first_hit_idx: torch.Tensor,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build target-aware STOP labels and a STOP-loss validity mask.

    Positive STOP is assigned at the first human fixation entering the expanded
    target bbox. For steps after that first hit, STOP loss is ignored, because
    the model should already have stopped. If no hit is observed, fall back to
    the true final fixation for untruncated trials.
    """
    device = active.device
    train_min_stop_step = int(cfg.get("train_min_stop_step", cfg.get("min_stop_step", 2)))
    target_stop = torch.zeros_like(active, dtype=torch.float32)
    stop_valid = active.clone()

    has_hit = first_hit_idx >= 0
    # If f0 is already inside the target, the first trainable stop decision is t=1.
    effective_hit_idx = torch.where(has_hit, first_hit_idx.clamp_min(1), first_hit_idx)
    effective_hit_idx = torch.where(
        has_hit,
        torch.maximum(effective_hit_idx, torch.full_like(effective_hit_idx, train_min_stop_step)),
        effective_hit_idx,
    )

    t_tensor = torch.full_like(first_hit_idx, int(t))
    hit_positive = has_hit & (t_tensor == effective_hit_idx)
    after_hit = has_hit & (t_tensor > effective_hit_idx)

    no_hit = ~has_hit
    final_positive = no_hit & (t_tensor == (scanpath_len - 1)) & (is_truncated == 0)

    target_stop = (hit_positive | final_positive).float()
    # Ignore stop loss after the first target hit because training should stop there.
    stop_valid = active & (~after_hit)
    return target_stop, stop_valid




def make_found_target_for_step(
    mode: str,
    active: torch.Tensor,
    target_stop: torch.Tensor,
    stop_valid: torch.Tensor,
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    bbox_expanded: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a FOUND supervision signal.

    Safe default is stop_label, which preserves the original uionly3 behavior.
    Other modes are available for controlled experiments, but should be used
    carefully because stronger FOUND labels can cause early-stop collapse.
    """
    mode = str(mode or "stop_label").lower()
    if mode in ["stop", "stop_label", "target_stop", "original"]:
        return target_stop, stop_valid
    if mode in ["hard_gt_bbox", "gt_bbox", "gt_inside"]:
        found_target = point_inside_bbox(target_xy, bbox_expanded).float()
        return found_target, active
    if mode in ["hard_pred_bbox", "pred_bbox", "pred_inside"]:
        # Uses predicted point for label construction during training. This is
        # intentionally detached from target generation to avoid changing coord gradients.
        found_target = point_inside_bbox(pred_xy.detach(), bbox_expanded).float()
        return found_target, active
    raise ValueError(f"Unknown found_target_mode: {mode}")

def compute_stop_pos_weight_trial(train_samples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> float:
    # Approximate class balance with one positive per untruncated trial.
    max_len = int(cfg.get("max_scanpath_len", cfg.get("max_rollout_steps", 20)))
    num_pos = 0
    num_neg = 0
    for s in train_samples:
        raw_len = int(s["scanpath_len"])
        eff_len = min(raw_len, max_len)
        if eff_len < 2:
            continue
        if raw_len <= max_len:
            num_pos += 1
            num_neg += max(0, eff_len - 2)
        else:
            num_neg += max(0, eff_len - 1)
    raw = num_neg / max(1, num_pos)
    cap = float(cfg.get("stop_pos_weight_cap", cfg.get("max_stop_pos_weight", 10.0)))
    return float(min(raw, cap))


def make_history_window(generated_history: torch.Tensor, history_len: int):
    B, G, C = generated_history.shape
    device = generated_history.device
    dtype = generated_history.dtype
    if G >= history_len:
        hist = generated_history[:, G - history_len:G, :]
        mask = torch.ones(B, history_len, device=device, dtype=dtype)
        return hist, mask
    pad_len = history_len - G
    pad = torch.zeros(B, pad_len, C, device=device, dtype=dtype)
    hist = torch.cat([pad, generated_history], dim=1)
    mask = torch.cat([
        torch.zeros(B, pad_len, device=device, dtype=dtype),
        torch.ones(B, G, device=device, dtype=dtype),
    ], dim=1)
    return hist, mask


def append_next_history(generated_history, pred_xy, pred_dur, gt_next_xydur, train: bool, cfg: Dict[str, Any]):
    B = pred_xy.size(0)
    device = pred_xy.device
    dtype = pred_xy.dtype
    detach_hist = bool(cfg.get("detach_rollout_history", True))
    pred_for_history = pred_xy.detach() if detach_hist else pred_xy
    pred_dur_for_history = pred_dur.detach() if detach_hist else pred_dur
    pred_dur_for_history = pred_dur_for_history.clamp_min(0.0)
    max_feat = cfg.get("duration_feature_clip", None)
    if max_feat is not None:
        pred_dur_for_history = pred_dur_for_history.clamp(0.0, float(max_feat))

    next_xydur = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    next_xydur[:, 0, 0:2] = pred_for_history.clamp(0.0, 1.0)
    next_xydur[:, 0, 2] = pred_dur_for_history.to(dtype=dtype)

    if train:
        tf_ratio = float(cfg.get("rollout_teacher_forcing_ratio", 0.0))
        if tf_ratio > 0.0:
            use_gt = (torch.rand(B, device=device) < tf_ratio).view(B, 1, 1)
            gt = gt_next_xydur[:, None, :].to(device=device, dtype=dtype)
            next_xydur = torch.where(use_gt, gt, next_xydur)
    return torch.cat([generated_history, next_xydur], dim=1)


def cosine_direction_loss(pred_delta, gt_delta, eps=1e-6):
    pred_norm = torch.linalg.norm(pred_delta, dim=1)
    gt_norm = torch.linalg.norm(gt_delta, dim=1)
    valid = (pred_norm > eps) & (gt_norm > eps)
    cos = torch.zeros(pred_delta.size(0), device=pred_delta.device, dtype=pred_delta.dtype)
    if valid.any():
        cos_valid = nn.functional.cosine_similarity(pred_delta[valid], gt_delta[valid], dim=1, eps=eps)
        cos[valid] = cos_valid
    loss = 1.0 - cos
    return loss, valid


def revisit_loss_from_history(pred_xy: torch.Tensor, generated_history: torch.Tensor, active: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """IOR-style penalty: discourage predicting a fixation too close to recent generated fixations.

    Returns one scalar loss per sample. The loss is zero when the nearest recent
    fixation is farther than revisit_radius. This is meant to reduce local loops
    and repeated back-and-forth jitter during autoregressive rollout.
    """
    weight = float(cfg.get("revisit_loss_weight", 0.0))
    if weight <= 0.0 or generated_history.size(1) <= 0:
        return torch.zeros(pred_xy.size(0), device=pred_xy.device, dtype=pred_xy.dtype)

    recent_k = int(cfg.get("revisit_recent_k", 6))
    radius = float(cfg.get("revisit_radius", 0.06))
    eps = 1e-12

    recent_xy = generated_history[:, -recent_k:, 0:2]
    d = torch.sqrt(torch.sum((pred_xy.unsqueeze(1) - recent_xy) ** 2, dim=-1) + eps)
    min_d = d.min(dim=1).values
    loss = torch.relu(radius - min_d) / max(radius, 1e-6)
    return torch.where(active, loss, torch.zeros_like(loss))


def anti_reversal_loss_from_history(pred_xy: torch.Tensor, generated_history: torch.Tensor, active: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalize immediate direction reversal.

    If previous movement and current movement have negative cosine similarity,
    this returns a positive penalty. It does not ban turning, but discourages
    repeated clock/Z-shaped oscillations.
    """
    if generated_history.size(1) < 2:
        return torch.zeros(pred_xy.size(0), device=pred_xy.device, dtype=pred_xy.dtype)

    prev_xy = generated_history[:, -2, 0:2]
    cur_xy = generated_history[:, -1, 0:2]
    prev_delta = cur_xy - prev_xy
    cur_delta = pred_xy - cur_xy

    prev_norm = torch.linalg.norm(prev_delta, dim=1)
    cur_norm = torch.linalg.norm(cur_delta, dim=1)
    valid = active & (prev_norm > eps) & (cur_norm > eps)
    cos = torch.zeros(pred_xy.size(0), device=pred_xy.device, dtype=pred_xy.dtype)
    if valid.any():
        cos[valid] = nn.functional.cosine_similarity(cur_delta[valid], prev_delta[valid], dim=1, eps=eps)
    loss = torch.relu(-cos)
    return torch.where(valid, loss, torch.zeros_like(loss))


def safe_mean_sum(total_sum, total_count):
    return float(total_sum / max(1.0, total_count))


def run_epoch_rollout(
    model,
    loader,
    optimizer,
    coord_criterion,
    stop_criterion,
    device,
    cfg,
    train=True,
    debug_print=False,
):
    model.train() if train else model.eval()

    history_len = int(cfg["history_len"])
    max_scanpath_len = int(cfg["max_scanpath_len"])
    stop_loss_weight = float(cfg.get("stop_loss_weight", cfg.get("lambda_stop", 0.2)))
    delta_loss_weight = float(cfg.get("delta_loss_weight", 0.5))
    step_len_loss_weight = float(cfg.get("step_len_loss_weight", 0.2))
    direction_loss_weight = float(cfg.get("direction_loss_weight", 0.1))
    stop_far_loss_weight = float(cfg.get("stop_far_loss_weight", 0.1))
    duration_loss_weight = float(cfg.get("duration_loss_weight", 0.2))
    revisit_loss_weight = float(cfg.get("revisit_loss_weight", 0.0))
    anti_reversal_loss_weight = float(cfg.get("anti_reversal_loss_weight", 0.0))
    candidate_loss_weight = float(cfg.get("candidate_loss_weight", 0.0))
    found_loss_weight = float(cfg.get("found_loss_weight", 0.0))
    target_margin = float(cfg.get("target_stop_margin", cfg.get("target_margin", 0.05)))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    totals = {
        "loss": 0.0,
        "coord_loss": 0.0,
        "stop_loss": 0.0,
        "delta_loss": 0.0,
        "step_len_loss": 0.0,
        "direction_loss": 0.0,
        "stop_far_loss": 0.0,
        "duration_loss": 0.0,
        "revisit_loss": 0.0,
        "anti_reversal_loss": 0.0,
        "candidate_loss": 0.0,
        "found_loss": 0.0,
        "found_correct": 0.0,
        "found_prob_pos": 0.0,
        "found_prob_neg": 0.0,
        "candidate_top1_soft": 0.0,
        "candidate_confidence": 0.0,
        "num_found_valid": 0.0,
        "num_found_pos": 0.0,
        "num_found_neg": 0.0,
        "dist": 0.0,
        "stop_correct": 0.0,
        "stop_pos_correct": 0.0,
        "stop_neg_correct": 0.0,
        "stop_prob_pos": 0.0,
        "stop_prob_neg": 0.0,
        "pred_step_len": 0.0,
        "gt_step_len": 0.0,
        "pred_dur": 0.0,
        "gt_dur": 0.0,
        "num_steps": 0.0,
        "num_stop_valid": 0.0,
        "num_pos": 0.0,
        "num_neg": 0.0,
        "num_trials": 0.0,
        "grad_norm": 0.0,
        "grad_updates": 0.0,
    }

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device)
        cue_id = batch["cue_id"].to(device)
        target_crop_image = batch["target_crop_image"].to(device)
        scanpath_xydur_raw = batch["scanpath_xydur"].to(device)
        scanpath_xydur = prepare_model_scanpath_xydur(scanpath_xydur_raw, cfg)
        scanpath_mask = batch["scanpath_mask"].to(device)
        scanpath_len = batch["scanpath_len"].to(device)
        is_truncated = batch["is_truncated"].to(device)
        target_bbox_norm = batch["target_bbox_norm"].to(device)

        ui_geom = batch["ui_geom"].to(device)
        ui_type_id = batch["ui_type_id"].to(device)
        ui_mask = batch["ui_mask"].to(device)
        ui_crop_images = batch["ui_crop_images"].to(device)

        B, L, _ = scanpath_xydur.shape
        totals["num_trials"] += float(B)

        bbox_expanded = expand_bbox_torch(target_bbox_norm, target_margin)
        first_hit_idx = compute_first_hit_indices(scanpath_xydur, scanpath_mask, bbox_expanded)
        candidate_soft_targets = soft_ui_candidate_targets(
            ui_geom=ui_geom,
            ui_mask=ui_mask,
            target_bbox_norm=target_bbox_norm,
            tau=float(cfg.get("candidate_target_tau", 0.08)),
        )

        if train:
            optimizer.zero_grad(set_to_none=True)

        generated_history = scanpath_xydur[:, :1, :].clone()
        active_count_total = scanpath_mask[:, 1:].sum().clamp_min(1.0)

        batch_active_count = 0.0
        batch_loss_sum_float = 0.0
        batch_dist_sum_float = 0.0

        for t in range(1, L):
            active = scanpath_mask[:, t] > 0
            if not active.any():
                break

            history_xydur, history_mask = make_history_window(generated_history, history_len)
            step_idx = torch.full((B,), int(t), dtype=torch.long, device=device)

            with torch.set_grad_enabled(train):
                pred_xy, pred_dur, pred_stop_logit, aux = model(
                    image=image,
                    cue_id=cue_id,
                    target_crop_image=target_crop_image,
                    history_xydur=history_xydur,
                    ui_geom=ui_geom,
                    ui_type_id=ui_type_id,
                    ui_mask=ui_mask,
                    ui_crop_images=ui_crop_images,
                    target_bbox_norm=target_bbox_norm,
                    history_mask=history_mask,
                    step_idx=step_idx,
                    return_aux=True,
                )

                if pred_stop_logit.ndim == 2 and pred_stop_logit.shape[-1] == 1:
                    pred_stop_logit = pred_stop_logit.squeeze(-1)
                if pred_dur.ndim == 2 and pred_dur.shape[-1] == 1:
                    pred_dur = pred_dur.squeeze(-1)
                if pred_xy.shape != (B, 2):
                    raise RuntimeError(f"pred_xy should have shape {(B, 2)}, got {tuple(pred_xy.shape)}")
                if pred_dur.shape != (B,):
                    raise RuntimeError(f"pred_dur should have shape {(B,)}, got {tuple(pred_dur.shape)}")
                if pred_stop_logit.shape != (B,):
                    raise RuntimeError(f"pred_stop_logit should have shape {(B,)}, got {tuple(pred_stop_logit.shape)}")

                if cfg.get("strict_finite_check", True):
                    require_finite("pred_xy", pred_xy, f"batch={batch_idx}, t={t}, train={train}")
                    require_finite("pred_dur", pred_dur, f"batch={batch_idx}, t={t}, train={train}")
                    require_finite("pred_stop_logit", pred_stop_logit, f"batch={batch_idx}, t={t}, train={train}")

                target_xy = scanpath_xydur[:, t, 0:2]
                target_xydur = scanpath_xydur[:, t, :]
                target_dur = target_xydur[:, 2]
                prev_gt_xy = scanpath_xydur[:, t - 1, 0:2]
                last_gen_xy = generated_history[:, -1, 0:2]

                target_stop, stop_valid = make_target_stop_for_step(
                    t=t,
                    active=active,
                    scanpath_len=scanpath_len,
                    is_truncated=is_truncated,
                    first_hit_idx=first_hit_idx,
                    cfg=cfg,
                )

                coord_loss_all = reduce_coord_loss(coord_criterion(pred_xy, target_xy))

                pred_delta = pred_xy - last_gen_xy
                # Coordinate correction target from the current generated rollout state.
                # Stage 4D keeps Stage 4D movement fix:
                #   In Stage 4, delta loss used correction_delta, but step-length and
                #   direction losses used human_delta = target_xy - prev_gt_xy.
                #   During scheduled-sampling rollout, last_gen_xy can differ from prev_gt_xy,
                #   so those losses may supervise a direction/length that is inconsistent
                #   with the actual generated state. Here, all movement losses are aligned
                #   to the same correction vector from generated state -> human next fixation.
                correction_delta = target_xy - last_gen_xy
                human_delta = target_xy - prev_gt_xy  # kept only for optional diagnostics/reference

                delta_loss_all = reduce_coord_loss(coord_criterion(pred_delta, correction_delta))

                pred_step_len = torch.linalg.norm(pred_delta, dim=1)
                correction_step_len = torch.linalg.norm(correction_delta, dim=1)
                gt_step_len = correction_step_len
                step_len_loss_all = nn.functional.smooth_l1_loss(
                    pred_step_len,
                    correction_step_len,
                    beta=float(cfg.get("step_len_beta", 0.05)),
                    reduction="none",
                )

                dur_loss_all = duration_loss_all(pred_dur, target_dur, cfg)
                revisit_loss_all = revisit_loss_from_history(pred_xy, generated_history, active, cfg)
                anti_reversal_loss_all = anti_reversal_loss_from_history(pred_xy, generated_history, active)

                dir_loss_all, dir_valid = cosine_direction_loss(pred_delta, correction_delta)
                dir_effective = active & dir_valid

                stop_loss_all_raw = stop_criterion(pred_stop_logit, target_stop)
                stop_loss_all = torch.zeros_like(stop_loss_all_raw)
                stop_loss_all[stop_valid] = stop_loss_all_raw[stop_valid]

                stop_prob = torch.sigmoid(pred_stop_logit)
                bbox_dist = point_to_bbox_distance(pred_xy, bbox_expanded)
                # Penalize confident STOP while predicted point is still far from target.
                stop_far_loss_all = stop_prob * bbox_dist

                candidate_loss_all = soft_candidate_ce_loss(aux["candidate_logits"], candidate_soft_targets, ui_mask)
                found_logit = aux.get("found_logit", pred_stop_logit)
                found_target, found_valid = make_found_target_for_step(
                    mode=cfg.get("found_target_mode", "stop_label"),
                    active=active,
                    target_stop=target_stop,
                    stop_valid=stop_valid,
                    pred_xy=pred_xy,
                    target_xy=target_xy,
                    bbox_expanded=bbox_expanded,
                )
                found_loss_raw = stop_criterion(found_logit, found_target)
                found_loss_all = torch.zeros_like(found_loss_raw)
                found_loss_all[found_valid] = found_loss_raw[found_valid]

                loss_all = (
                    coord_loss_all
                    + delta_loss_weight * delta_loss_all
                    + step_len_loss_weight * step_len_loss_all
                    + direction_loss_weight * torch.where(dir_valid, dir_loss_all, torch.zeros_like(dir_loss_all))
                    + duration_loss_weight * dur_loss_all
                    + revisit_loss_weight * revisit_loss_all
                    + anti_reversal_loss_weight * anti_reversal_loss_all
                    + stop_loss_weight * stop_loss_all
                    + stop_far_loss_weight * stop_far_loss_all
                    + candidate_loss_weight * candidate_loss_all
                    + found_loss_weight * found_loss_all
                )

                if cfg.get("strict_finite_check", True):
                    require_finite("loss_all", loss_all, f"batch={batch_idx}, t={t}, train={train}")

                # Direction loss is only meaningful when both vectors are non-zero.
                active_float = active.float()
                loss_step_sum = loss_all[active].sum()
                coord_step_sum = coord_loss_all[active].sum()
                delta_step_sum = delta_loss_all[active].sum()
                step_len_step_sum = step_len_loss_all[active].sum()
                direction_step_sum = dir_loss_all[dir_effective].sum() if dir_effective.any() else torch.zeros((), device=device)
                duration_step_sum = dur_loss_all[active].sum()
                revisit_step_sum = revisit_loss_all[active].sum()
                anti_reversal_step_sum = anti_reversal_loss_all[active].sum()
                stop_step_sum = stop_loss_all[stop_valid].sum() if stop_valid.any() else torch.zeros((), device=device)
                stop_far_step_sum = stop_far_loss_all[active].sum()
                candidate_step_sum = candidate_loss_all[active].sum()
                found_step_sum = found_loss_all[found_valid].sum() if found_valid.any() else torch.zeros((), device=device)
                active_count = active_float.sum()

                loss_step_scaled = loss_step_sum / active_count_total
                if train:
                    loss_step_scaled.backward()

            with torch.no_grad():
                dist = euclidean_dist(pred_xy, target_xy)
                stop_prob = torch.sigmoid(pred_stop_logit)
                stop_pred = (stop_prob > 0.5).float()
                pos_mask = stop_valid & (target_stop > 0.5)
                neg_mask = stop_valid & (target_stop <= 0.5)

                totals["num_steps"] += float(active_count.item())
                totals["num_stop_valid"] += float(stop_valid.float().sum().item())
                totals["loss"] += float(loss_step_sum.detach().item())
                totals["coord_loss"] += float(coord_step_sum.detach().item())
                totals["delta_loss"] += float(delta_step_sum.detach().item())
                totals["step_len_loss"] += float(step_len_step_sum.detach().item())
                totals["direction_loss"] += float(direction_step_sum.detach().item())
                totals["stop_loss"] += float(stop_step_sum.detach().item())
                totals["stop_far_loss"] += float(stop_far_step_sum.detach().item())
                totals["duration_loss"] += float(duration_step_sum.detach().item())
                totals["revisit_loss"] += float(revisit_step_sum.detach().item())
                totals["anti_reversal_loss"] += float(anti_reversal_step_sum.detach().item())
                totals["candidate_loss"] += float(candidate_step_sum.detach().item())
                totals["found_loss"] += float(found_step_sum.detach().item())
                totals["candidate_confidence"] += float(aux["candidate_confidence"][active].sum().detach().item())
                with torch.no_grad():
                    cand_top = aux["candidate_probs"].argmax(dim=1)
                    cand_soft_at_top = candidate_soft_targets.gather(1, cand_top.view(-1, 1)).squeeze(1)
                    totals["candidate_top1_soft"] += float(cand_soft_at_top[active].sum().detach().item())
                    found_prob = torch.sigmoid(found_logit)
                    found_pred = (found_prob > 0.5).float()
                    found_pos_mask = found_valid & (found_target > 0.5)
                    found_neg_mask = found_valid & (found_target <= 0.5)
                    if found_valid.any():
                        totals["found_correct"] += float((found_pred[found_valid] == found_target[found_valid]).float().sum().item())
                        totals["num_found_valid"] += float(found_valid.float().sum().item())
                    if found_pos_mask.any():
                        totals["num_found_pos"] += float(found_pos_mask.float().sum().item())
                        totals["found_prob_pos"] += float(found_prob[found_pos_mask].sum().item())
                    if found_neg_mask.any():
                        totals["num_found_neg"] += float(found_neg_mask.float().sum().item())
                        totals["found_prob_neg"] += float(found_prob[found_neg_mask].sum().item())
                totals["dist"] += float(dist[active].sum().detach().item())
                totals["pred_step_len"] += float(pred_step_len[active].sum().detach().item())
                totals["gt_step_len"] += float(gt_step_len[active].sum().detach().item())
                totals["pred_dur"] += float(pred_dur[active].sum().detach().item())
                totals["gt_dur"] += float(target_dur[active].sum().detach().item())

                if stop_valid.any():
                    totals["stop_correct"] += float((stop_pred[stop_valid] == target_stop[stop_valid]).float().sum().item())
                if pos_mask.any():
                    totals["num_pos"] += float(pos_mask.float().sum().item())
                    totals["stop_pos_correct"] += float((stop_pred[pos_mask] == 1).float().sum().item())
                    totals["stop_prob_pos"] += float(stop_prob[pos_mask].sum().item())
                if neg_mask.any():
                    totals["num_neg"] += float(neg_mask.float().sum().item())
                    totals["stop_neg_correct"] += float((stop_pred[neg_mask] == 0).float().sum().item())
                    totals["stop_prob_neg"] += float(stop_prob[neg_mask].sum().item())

                batch_active_count += float(active_count.item())
                batch_loss_sum_float += float(loss_step_sum.detach().item())
                batch_dist_sum_float += float(dist[active].sum().detach().item())

            generated_history = append_next_history(
                generated_history=generated_history,
                pred_xy=pred_xy,
                pred_dur=pred_dur,
                gt_next_xydur=target_xydur,
                train=train,
                cfg=cfg,
            )

        if train and batch_active_count > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            totals["grad_updates"] += 1.0

        if batch_idx % int(cfg.get("print_every_batches", 10)) == 0:
            mean_loss = batch_loss_sum_float / max(1.0, batch_active_count)
            mean_dist = batch_dist_sum_float / max(1.0, batch_active_count)
            print(f"    batch {batch_idx}/{len(loader)} loss={mean_loss:.6f} dist={mean_dist:.6f}")

    denom = max(1.0, totals["num_steps"])
    stop_denom = max(1.0, totals["num_stop_valid"])
    found_denom = max(1.0, totals["num_found_valid"])
    found_pos_denom = max(1.0, totals["num_found_pos"])
    found_neg_denom = max(1.0, totals["num_found_neg"])
    metrics = {
        "loss": totals["loss"] / denom,
        "coord_loss": totals["coord_loss"] / denom,
        "stop_loss": totals["stop_loss"] / stop_denom,
        "delta_loss": totals["delta_loss"] / denom,
        "step_len_loss": totals["step_len_loss"] / denom,
        "direction_loss": totals["direction_loss"] / denom,
        "stop_far_loss": totals["stop_far_loss"] / denom,
        "duration_loss": totals["duration_loss"] / denom,
        "revisit_loss": totals["revisit_loss"] / denom,
        "anti_reversal_loss": totals["anti_reversal_loss"] / denom,
        "candidate_loss": totals["candidate_loss"] / denom,
        "found_loss": totals["found_loss"] / found_denom,
        "found_acc": totals["found_correct"] / found_denom,
        "mean_found_prob_pos": totals["found_prob_pos"] / found_pos_denom,
        "mean_found_prob_neg": totals["found_prob_neg"] / found_neg_denom,
        "num_found_valid": int(totals["num_found_valid"]),
        "num_found_pos": int(totals["num_found_pos"]),
        "num_found_neg": int(totals["num_found_neg"]),
        "candidate_top1_soft_mean": totals["candidate_top1_soft"] / denom,
        "candidate_confidence_mean": totals["candidate_confidence"] / denom,
        "dist": totals["dist"] / denom,
        "stop_acc": totals["stop_correct"] / stop_denom,
        "stop_pos_acc": totals["stop_pos_correct"] / max(1.0, totals["num_pos"]),
        "stop_neg_acc": totals["stop_neg_correct"] / max(1.0, totals["num_neg"]),
        "mean_stop_prob_pos": totals["stop_prob_pos"] / max(1.0, totals["num_pos"]),
        "mean_stop_prob_neg": totals["stop_prob_neg"] / max(1.0, totals["num_neg"]),
        "pred_step_len_mean": totals["pred_step_len"] / denom,
        "gt_step_len_mean": totals["gt_step_len"] / denom,
        "pred_dur_mean": totals["pred_dur"] / denom,
        "gt_dur_mean": totals["gt_dur"] / denom,
        "num_pos": int(totals["num_pos"]),
        "num_neg": int(totals["num_neg"]),
        "num_steps": int(totals["num_steps"]),
        "num_stop_valid": int(totals["num_stop_valid"]),
        "num_trials": int(totals["num_trials"]),
        "grad_norm_mean": totals["grad_norm"] / max(1.0, totals["grad_updates"]),
    }
    return metrics


def save_checkpoint(path, model, cfg, cue_vocab, ui_type_vocab, epoch, current_val_metrics, best_val_loss, best_val_dist):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "cue_vocab": cue_vocab,
        "ui_type_vocab": ui_type_vocab,
        "epoch": epoch,
        "current_val_metrics": current_val_metrics,
        "best_val_loss": best_val_loss,
        "best_val_dist": best_val_dist,
        "model_class": "HybridTemporalTargetDurationDecoderModel",
        "stage": "ScanUIFormer",
    }, path)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.alpha is not None:
        cfg["ui_memory_scale"] = args.alpha
    else:
        cfg["ui_memory_scale"] = cfg.get("ui_memory_scale", 1.0)

    if args.freeze_patch and args.no_freeze_patch:
        raise ValueError("Cannot use both --freeze_patch and --no_freeze_patch")
    if args.freeze_patch:
        cfg["freeze_patch_backbone"] = True
    elif args.no_freeze_patch:
        cfg["freeze_patch_backbone"] = False
    else:
        cfg["freeze_patch_backbone"] = cfg.get("freeze_patch_backbone", False)

    cfg.setdefault("max_scanpath_len", cfg.get("max_rollout_steps", 20))
    cfg.setdefault("target_crop_size", 48)
    cfg.setdefault("max_delta", 0.65)
    cfg.setdefault("detach_rollout_history", True)
    cfg.setdefault("rollout_teacher_forcing_ratio", 0.0)
    cfg.setdefault("stop_loss_weight", cfg.get("lambda_stop", 0.2))
    cfg.setdefault("delta_loss_weight", 0.5)
    cfg.setdefault("step_len_loss_weight", 0.2)
    cfg.setdefault("direction_loss_weight", 0.1)
    cfg.setdefault("stop_far_loss_weight", 0.1)
    cfg.setdefault("duration_loss_weight", 0.2)
    cfg.setdefault("duration_loss", "smooth_l1")
    cfg.setdefault("duration_target", "log1p")
    cfg.setdefault("duration_scale", 1.0)
    cfg.setdefault("duration_output", "softplus")
    cfg.setdefault("duration_feature_clip", 8.0)
    cfg.setdefault("revisit_loss_weight", 0.0)
    cfg.setdefault("revisit_radius", 0.06)
    cfg.setdefault("revisit_recent_k", 6)
    cfg.setdefault("anti_reversal_loss_weight", 0.0)
    cfg.setdefault("candidate_loss_weight", 0.2)
    cfg.setdefault("found_loss_weight", 0.05)
    cfg.setdefault("candidate_target_tau", 0.08)
    cfg.setdefault("use_candidate_stop", True)
    cfg.setdefault("found_target_mode", "stop_label")
    cfg.setdefault("use_target_aware_stop", False)
    cfg.setdefault("target_stop_margin", cfg.get("target_margin", 0.05))
    cfg.setdefault("min_stop_step", 2)
    cfg.setdefault("train_min_stop_step", cfg.get("min_stop_step", 2))
    cfg.setdefault("stop_pos_weight_cap", 10.0)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("target_present_only", True)
    cfg.setdefault("coord_loss", "smooth_l1")
    cfg.setdefault("smooth_l1_beta", 0.05)
    cfg.setdefault("strict_finite_check", True)
    cfg.setdefault("print_every_batches", 10)

    print("config_path =", args.config)
    print("output_dir =", cfg["output_dir"])
    print("ui_memory_scale =", cfg["ui_memory_scale"])
    print("freeze_patch_backbone =", cfg["freeze_patch_backbone"])
    print("max_scanpath_len =", cfg["max_scanpath_len"])
    print("target_crop_size =", cfg["target_crop_size"])
    print("target_stop_margin =", cfg["target_stop_margin"])
    print("loss weights:", {
        "stop": cfg["stop_loss_weight"],
        "delta": cfg["delta_loss_weight"],
        "step_len": cfg["step_len_loss_weight"],
        "direction": cfg["direction_loss_weight"],
        "stop_far": cfg["stop_far_loss_weight"],
        "duration": cfg["duration_loss_weight"],
        "revisit": cfg["revisit_loss_weight"],
        "anti_reversal": cfg["anti_reversal_loss_weight"],
        "candidate": cfg["candidate_loss_weight"],
        "found": cfg["found_loss_weight"],
    })
    print("candidate settings:", {
        "candidate_target_tau": cfg.get("candidate_target_tau", 0.08),
        "found_target_mode": cfg.get("found_target_mode", "stop_label"),
        "use_candidate_stop": cfg.get("use_candidate_stop", True),
        "use_target_aware_stop": cfg.get("use_target_aware_stop", False),
    })
    print("IOR / anti-oscillation settings:", {
        "revisit_radius": cfg["revisit_radius"],
        "revisit_recent_k": cfg["revisit_recent_k"],
    })
    print("duration settings:", {
        "duration_target": cfg["duration_target"],
        "duration_scale": cfg["duration_scale"],
        "duration_output": cfg["duration_output"],
        "duration_feature_clip": cfg["duration_feature_clip"],
    })

    set_seed(int(cfg["seed"]))
    ensure_dir(cfg["output_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    trials = read_jsonl_gz(cfg["trials_path"])
    print("loaded trials =", len(trials))
    if cfg.get("target_present_only", True):
        trials = filter_target_present_trials(trials)
        print("trials after target-present filtering =", len(trials))

    trials = filter_trials_by_cue(trials, cfg)

    train_trials, val_trials, test_trials, split_meta = split_trials_with_seekui_holdout(trials, cfg)
    print("split meta =", json.dumps(split_meta, indent=2))
    save_json({
        "train_trial_ids": [t["trial_id"] for t in train_trials],
        "val_trial_ids": [t["trial_id"] for t in val_trials],
        "test_trial_ids": [t["trial_id"] for t in test_trials],
        "split_meta": split_meta,
        "train_seekui_like_ids": [make_seekui_like_id(t) for t in train_trials],
        "val_seekui_like_ids": [make_seekui_like_id(t) for t in val_trials],
        "test_seekui_like_ids": [make_seekui_like_id(t) for t in test_trials],
    }, os.path.join(cfg["output_dir"], "split.json"))

    train_trial_index = build_trial_index(train_trials)
    val_trial_index = build_trial_index(val_trials)
    test_trial_index = build_trial_index(test_trials)

    seg_index = build_seg_index(cfg["seg_root"])
    print("n_seg_files =", len(seg_index))
    cue_vocab = build_cue_vocab(train_trials)
    ui_type_vocab = build_ui_type_vocab(train_trials, seg_index)
    save_json(cue_vocab, os.path.join(cfg["output_dir"], "cue_vocab.json"))
    save_json(ui_type_vocab, os.path.join(cfg["output_dir"], "ui_type_vocab.json"))

    train_samples = build_hybrid_trial_samples(train_trials)
    val_samples = build_hybrid_trial_samples(val_trials)
    test_samples = build_hybrid_trial_samples(test_trials)

    if cfg.get("max_train_samples") is not None:
        train_samples = train_samples[:int(cfg["max_train_samples"])]
    if cfg.get("max_val_samples") is not None:
        val_samples = val_samples[:int(cfg["max_val_samples"])]
    if cfg.get("max_test_samples") is not None:
        test_samples = test_samples[:int(cfg["max_test_samples"])]

    print("train trial samples =", len(train_samples))
    print("val trial samples   =", len(val_samples))
    print("test trial samples  =", len(test_samples))
    check_hybrid_trial_samples(train_samples, "train", cfg["max_scanpath_len"])
    check_hybrid_trial_samples(val_samples, "val", cfg["max_scanpath_len"])
    check_hybrid_trial_samples(test_samples, "test", cfg["max_scanpath_len"])

    common_ds_kwargs = dict(
        cue_vocab=cue_vocab,
        ui_type_vocab=ui_type_vocab,
        seg_index=seg_index,
        image_dir=cfg["image_dir"],
        image_size=cfg["image_size"],
        max_ui_tokens=cfg["max_ui_tokens"],
        drop_full_screen_root=cfg.get("drop_full_screen_root", False),
        crop_size=cfg.get("crop_size", 32),
        max_scanpath_len=cfg["max_scanpath_len"],
        target_crop_size=cfg["target_crop_size"],
    )
    train_ds = HybridTrialDecoderTargetCropDataset(train_samples, train_trial_index, **common_ds_kwargs)
    val_ds = HybridTrialDecoderTargetCropDataset(val_samples, val_trial_index, **common_ds_kwargs)
    test_ds = HybridTrialDecoderTargetCropDataset(test_samples, test_trial_index, **common_ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = HybridTemporalTargetDurationDecoderModel(
        vit_name=cfg["vit_name"],
        pretrained=cfg["pretrained"],
        cue_vocab_size=len(cue_vocab),
        ui_type_vocab_size=len(ui_type_vocab),
        history_len=cfg["history_len"],
        max_scanpath_len=cfg["max_scanpath_len"],
        ui_geom_dim=cfg["ui_geom_dim"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        ui_memory_scale=cfg["ui_memory_scale"],
        freeze_patch_backbone=cfg["freeze_patch_backbone"],
        target_crop_size=cfg["target_crop_size"],
        max_delta=cfg["max_delta"],
        duration_output=cfg.get("duration_output", "softplus"),
        use_patch_memory=cfg.get("use_patch_memory", True),
        use_target_aware_stop=cfg.get("use_target_aware_stop", False),
        use_ui_target_similarity=cfg.get("use_ui_target_similarity", True),
        use_candidate_stop=cfg.get("use_candidate_stop", True),
        ui_num_layers=cfg.get("ui_num_layers", 2),
        patch_num_layers=cfg.get("patch_num_layers", 1),
        state_refine_layers=cfg.get("state_refine_layers", 2),
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("trainable_params =", trainable_params)
    print("total_params     =", total_params)

    init_checkpoint = cfg.get("init_checkpoint", None)
    if init_checkpoint:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"init_checkpoint not found: {init_checkpoint}")
        print(f"Loading init checkpoint for fine-tuning: {init_checkpoint}")
        ckpt = torch.load(init_checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded init checkpoint.")
        print("  missing keys:", len(missing))
        if len(missing) > 0:
            print("  first missing keys:", missing[:10])
        print("  unexpected keys:", len(unexpected))
        if len(unexpected) > 0:
            print("  first unexpected keys:", unexpected[:10])

    coord_criterion = build_coord_criterion(cfg)
    stop_pos_weight_value = compute_stop_pos_weight_trial(train_samples, cfg)
    print("STOP pos_weight =", stop_pos_weight_value)
    stop_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([stop_pos_weight_value], device=device),
        reduction="none",
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    save_json(cfg, os.path.join(cfg["output_dir"], "config.json"))
    log_path = os.path.join(cfg["output_dir"], "train_log.csv")
    history_json_path = os.path.join(cfg["output_dir"], "metrics_history.json")

    fieldnames = [
        "epoch",
        "train_loss", "train_coord_loss", "train_stop_loss", "train_delta_loss", "train_step_len_loss", "train_direction_loss", "train_duration_loss", "train_revisit_loss", "train_anti_reversal_loss", "train_stop_far_loss",
        "train_candidate_loss", "train_found_loss", "train_found_acc", "train_mean_found_prob_pos", "train_mean_found_prob_neg", "train_candidate_top1_soft_mean", "train_candidate_confidence_mean", "train_num_found_valid", "train_num_found_pos", "train_num_found_neg", "train_dist",
        "train_stop_acc", "train_stop_pos_acc", "train_stop_neg_acc", "train_mean_stop_prob_pos", "train_mean_stop_prob_neg",
        "train_pred_step_len_mean", "train_gt_step_len_mean", "train_pred_dur_mean", "train_gt_dur_mean", "train_grad_norm_mean", "train_num_pos", "train_num_neg", "train_num_steps", "train_num_stop_valid",
        "val_loss", "val_coord_loss", "val_stop_loss", "val_delta_loss", "val_step_len_loss", "val_direction_loss", "val_duration_loss", "val_revisit_loss", "val_anti_reversal_loss", "val_stop_far_loss",
        "val_candidate_loss", "val_found_loss", "val_found_acc", "val_mean_found_prob_pos", "val_mean_found_prob_neg", "val_candidate_top1_soft_mean", "val_candidate_confidence_mean", "val_num_found_valid", "val_num_found_pos", "val_num_found_neg", "val_dist",
        "val_stop_acc", "val_stop_pos_acc", "val_stop_neg_acc", "val_mean_stop_prob_pos", "val_mean_stop_prob_neg",
        "val_pred_step_len_mean", "val_gt_step_len_mean", "val_pred_dur_mean", "val_gt_dur_mean", "val_num_pos", "val_num_neg", "val_num_steps", "val_num_stop_valid",
    ]

    best_val_loss = math.inf
    best_val_dist = math.inf
    best_path = os.path.join(cfg["output_dir"], "best.pt")
    best_loss_path = os.path.join(cfg["output_dir"], "best_loss.pt")
    best_dist_path = os.path.join(cfg["output_dir"], "best_dist.pt")
    last_path = os.path.join(cfg["output_dir"], "last.pt")
    metrics_history = []

    print("starting Stage4G uionly3 SAFE: candidate/found logging + SeekUI-test holdout split...")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, int(cfg["epochs"]) + 1):
            print(f"\n===== Epoch {epoch:02d}/{cfg['epochs']} =====")
            train_metrics = run_epoch_rollout(
                model, train_loader, optimizer, coord_criterion, stop_criterion, device, cfg, train=True,
                debug_print=cfg.get("debug_print", False),
            )
            val_metrics = run_epoch_rollout(
                model, val_loader, None, coord_criterion, stop_criterion, device, cfg, train=False,
                debug_print=cfg.get("debug_print", False),
            )

            row = {"epoch": epoch}
            for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
                for key in [
                    "loss", "coord_loss", "stop_loss", "delta_loss", "step_len_loss", "direction_loss", "duration_loss", "revisit_loss", "anti_reversal_loss", "stop_far_loss",
                    "candidate_loss", "found_loss", "found_acc", "mean_found_prob_pos", "mean_found_prob_neg", "candidate_top1_soft_mean", "candidate_confidence_mean", "num_found_valid", "num_found_pos", "num_found_neg", "dist",
                    "stop_acc", "stop_pos_acc", "stop_neg_acc", "mean_stop_prob_pos", "mean_stop_prob_neg",
                    "pred_step_len_mean", "gt_step_len_mean", "pred_dur_mean", "gt_dur_mean", "num_pos", "num_neg", "num_steps", "num_stop_valid",
                ]:
                    row[f"{prefix}_{key}"] = metrics[key]
            row["train_grad_norm_mean"] = train_metrics["grad_norm_mean"]
            writer.writerow(row)
            f.flush()

            metrics_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            save_json(metrics_history, history_json_path)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.6f} train_dist={train_metrics['dist']:.6f} "
                f"train_delta={train_metrics['delta_loss']:.6f} train_dur={train_metrics['duration_loss']:.6f} "
                f"train_revisit={train_metrics['revisit_loss']:.6f} train_rev={train_metrics['anti_reversal_loss']:.6f} train_stop_pos_acc={train_metrics['stop_pos_acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.6f} val_dist={val_metrics['dist']:.6f} "
                f"val_delta={val_metrics['delta_loss']:.6f} val_dur={val_metrics['duration_loss']:.6f} "
                f"val_revisit={val_metrics['revisit_loss']:.6f} val_rev={val_metrics['anti_reversal_loss']:.6f} val_stop_pos_acc={val_metrics['stop_pos_acc']:.4f}"
            )

            epoch_ckpt_path = os.path.join(cfg["output_dir"], f"epoch_{epoch:02d}.pt")
            save_checkpoint(epoch_ckpt_path, model, cfg, cue_vocab, ui_type_vocab, epoch, val_metrics, best_val_loss, best_val_dist)
            print(f"  saved epoch checkpoint -> {epoch_ckpt_path}")

            if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(best_path, model, cfg, cue_vocab, ui_type_vocab, epoch, val_metrics, best_val_loss, best_val_dist)
                save_checkpoint(best_loss_path, model, cfg, cue_vocab, ui_type_vocab, epoch, val_metrics, best_val_loss, best_val_dist)
                print(f"  saved best checkpoint (by val_loss) -> {best_path}")

            if math.isfinite(val_metrics["dist"]) and val_metrics["dist"] < best_val_dist:
                best_val_dist = val_metrics["dist"]
                save_checkpoint(best_dist_path, model, cfg, cue_vocab, ui_type_vocab, epoch, val_metrics, best_val_loss, best_val_dist)
                print(f"  saved best_dist checkpoint -> {best_dist_path}")

    save_checkpoint(last_path, model, cfg, cue_vocab, ui_type_vocab, int(cfg["epochs"]), val_metrics, best_val_loss, best_val_dist)
    print(f"\nsaved last checkpoint -> {last_path}")

    selected_path = best_path if os.path.exists(best_path) else last_path
    selection_rule = "best_val_loss" if os.path.exists(best_path) else "last_fallback_no_best"
    ckpt = torch.load(selected_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = run_epoch_rollout(model, test_loader, None, coord_criterion, stop_criterion, device, cfg, train=False)
    result = {
        "selection_rule": selection_rule,
        "selected_checkpoint": selected_path,
        "selected_epoch": ckpt.get("epoch", None),
        "best_val_loss": best_val_loss,
        "best_val_dist": best_val_dist,
        "test": test_metrics,
        "split_meta": split_meta,
        "train_trials_used": len(train_samples),
        "val_trials_used": len(val_samples),
        "test_trials_used": len(test_samples),
    }
    save_json(result, os.path.join(cfg["output_dir"], "test_result.json"))
    print("\nFINAL TEST RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
