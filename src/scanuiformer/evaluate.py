"""
Evaluation + visualization for the temporal target-crop movement scanpath model.

This script is designed for checkpoints trained with:
  - dataset.py
  - model.py
  - train.py

It evaluates the SAME generated rollout records with multiple metric families:
  1) spatial accuracy: ADE, RMSE, FDE
  2) target success: any-hit, final-hit, first-hit step, time-to-target
  3) STOP / length behavior: pred length, GT length, early stop, timeout, stop reasons
  4) scanpath similarity: ScanMatch-lite, SED, DTW similarity, MultiMatch-lite

It also saves visualizations directly from the generated records, so the metrics
and the drawn scanpaths always correspond to the same rollout.

Example:
py evaluate.py --config outputs/run/config.json --checkpoint outputs/run/best.pt --split val --output_dir outputs/run/eval_v2 --visualize
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from dataset import (
    read_jsonl_gz,
    split_trials,
    build_trial_index,
    build_seg_index,
    build_hybrid_trial_samples,
    HybridTrialDecoderTargetCropDataset,
    filter_target_present_trials,
)
from model import HybridTemporalTargetDurationDecoderModel


# ============================================================================
# Basic IO
# ============================================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_ints_or_none(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def safe_float_name(x: float) -> str:
    return f"{float(x):.2f}".replace(".", "_")


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ============================================================================
# Dataset / model loading
# ============================================================================

def load_checkpoint_state(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    # allow raw state dict as fallback
    if isinstance(ckpt, dict):
        return ckpt, {"model_state_dict": ckpt}
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def merge_checkpoint_config(user_cfg: Dict[str, Any], ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Use checkpoint hyperparameters but keep current data/output paths."""
    cfg = dict(user_cfg)
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        keep_from_user = {"trials_path", "image_dir", "seg_root", "output_dir"}
        for k, v in ckpt_cfg.items():
            if k not in keep_from_user:
                cfg[k] = v
    cfg.setdefault("max_scanpath_len", cfg.get("max_rollout_steps", 20))
    cfg.setdefault("target_crop_size", 48)
    cfg.setdefault("target_present_only", True)
    cfg.setdefault("drop_full_screen_root", False)
    cfg.setdefault("crop_size", 32)
    cfg.setdefault("ui_memory_scale", 1.0)
    cfg.setdefault("freeze_patch_backbone", False)
    cfg.setdefault("max_delta", 0.65)
    cfg.setdefault("duration_target", "log1p")
    cfg.setdefault("duration_scale", 1.0)
    cfg.setdefault("duration_output", "softplus")
    cfg.setdefault("duration_feature_clip", 8.0)
    return cfg


def prepare_dataset_and_loader(
    cfg: Dict[str, Any],
    ckpt: Dict[str, Any],
    split_name: str,
    batch_size: int,
    num_workers: int,
    max_eval_samples: Optional[int] = None,
):
    if "cue_vocab" not in ckpt or "ui_type_vocab" not in ckpt:
        raise KeyError("Checkpoint must contain cue_vocab and ui_type_vocab from training.")

    trials = read_jsonl_gz(cfg["trials_path"])
    if cfg.get("target_present_only", True):
        trials = filter_target_present_trials(trials)

    train_trials, val_trials, test_trials = split_trials(
        trials,
        cfg.get("train_ratio", 0.8),
        cfg.get("val_ratio", 0.1),
        cfg.get("test_ratio", 0.1),
        cfg.get("seed", 42),
    )
    if split_name == "train":
        eval_trials = train_trials
    elif split_name == "val":
        eval_trials = val_trials
    elif split_name == "test":
        eval_trials = test_trials
    else:
        raise ValueError(f"Unknown split: {split_name}")

    seg_index = build_seg_index(cfg["seg_root"])
    trial_index = build_trial_index(eval_trials)
    samples = build_hybrid_trial_samples(eval_trials)
    if max_eval_samples is not None and max_eval_samples > 0:
        samples = samples[:max_eval_samples]

    dataset = HybridTrialDecoderTargetCropDataset(
        samples=samples,
        trial_index=trial_index,
        cue_vocab=ckpt["cue_vocab"],
        ui_type_vocab=ckpt["ui_type_vocab"],
        seg_index=seg_index,
        image_dir=cfg["image_dir"],
        image_size=cfg.get("image_size", 224),
        max_ui_tokens=cfg.get("max_ui_tokens", 64),
        drop_full_screen_root=cfg.get("drop_full_screen_root", False),
        crop_size=cfg.get("crop_size", 32),
        max_scanpath_len=cfg.get("max_scanpath_len", 20),
        target_crop_size=cfg.get("target_crop_size", 48),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, samples


def build_model_from_config(cfg: Dict[str, Any], ckpt: Dict[str, Any]) -> HybridTemporalTargetDurationDecoderModel:
    return HybridTemporalTargetDurationDecoderModel(
        vit_name=cfg["vit_name"],
        pretrained=False,
        cue_vocab_size=len(ckpt["cue_vocab"]),
        ui_type_vocab_size=len(ckpt["ui_type_vocab"]),
        history_len=cfg["history_len"],
        max_scanpath_len=cfg.get("max_scanpath_len", cfg.get("max_rollout_steps", 20)),
        ui_geom_dim=cfg.get("ui_geom_dim", 4),
        d_model=cfg.get("d_model", 192),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        ff_dim=cfg.get("ff_dim", 384),
        dropout=cfg.get("dropout", 0.1),
        ui_memory_scale=cfg.get("ui_memory_scale", 1.0),
        freeze_patch_backbone=cfg.get("freeze_patch_backbone", False),
        target_crop_size=cfg.get("target_crop_size", 48),
        max_delta=cfg.get("max_delta", 0.65),
        duration_output=cfg.get("duration_output", "softplus"),
        use_patch_memory=cfg.get("use_patch_memory", True),
        use_target_aware_stop=cfg.get("use_target_aware_stop", False),
        use_ui_target_similarity=cfg.get("use_ui_target_similarity", True),
        use_candidate_stop=cfg.get("use_candidate_stop", True),
    )


# ============================================================================
# Duration feature transform
# ============================================================================

def transform_duration_feature(dur: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    dur = torch.nan_to_num(dur.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    scale = float(cfg.get("duration_scale", 1.0))
    scale = max(scale, 1e-8)
    mode = str(cfg.get("duration_target", "log1p")).lower()
    if mode in ["log1p", "log"]:
        return torch.log1p(dur / scale)
    if mode == "sqrt":
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


def append_pred_xydur(history: torch.Tensor, pred_xy: torch.Tensor, pred_dur: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    B = pred_xy.size(0)
    device = pred_xy.device
    dtype = pred_xy.dtype
    pred_xy = pred_xy.clamp(0.0, 1.0)
    pred_dur = pred_dur.clamp_min(0.0)
    max_feat = cfg.get("duration_feature_clip", None)
    if max_feat is not None:
        pred_dur = pred_dur.clamp(0.0, float(max_feat))
    nxt = torch.cat([pred_xy, pred_dur.to(dtype=dtype).view(B, 1)], dim=1).unsqueeze(1)
    return torch.cat([history, nxt], dim=1)

# ============================================================================
# Rollout generation
# ============================================================================

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


def append_pred_xy(history: torch.Tensor, pred_xy: torch.Tensor) -> torch.Tensor:
    B = pred_xy.size(0)
    device = pred_xy.device
    dtype = pred_xy.dtype
    pred_xy = pred_xy.clamp(0.0, 1.0)
    dur = torch.zeros(B, 1, device=device, dtype=dtype)
    nxt = torch.cat([pred_xy, dur], dim=1).unsqueeze(1)
    return torch.cat([history, nxt], dim=1)


def expand_bbox_np(bbox: Sequence[float], margin: float) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return [
        max(0.0, x0 - margin),
        max(0.0, y0 - margin),
        min(1.0, x1 + margin),
        min(1.0, y1 + margin),
    ]


def point_in_bbox_np(x: float, y: float, bbox: Sequence[float]) -> bool:
    x0, y0, x1, y1 = bbox
    return x >= x0 and x <= x1 and y >= y0 and y <= y1


def points_any_hit_np(points: Sequence[Sequence[float]], bbox: Sequence[float]) -> bool:
    return any(point_in_bbox_np(float(x), float(y), bbox) for x, y in points)


def first_hit_step_np(points: Sequence[Sequence[float]], bbox: Sequence[float]) -> int:
    for i, (x, y) in enumerate(points):
        if point_in_bbox_np(float(x), float(y), bbox):
            return i
    return -1


def final_hit_np(points: Sequence[Sequence[float]], bbox: Sequence[float]) -> bool:
    if len(points) == 0:
        return False
    x, y = points[-1]
    return point_in_bbox_np(float(x), float(y), bbox)


@torch.no_grad()
def generate_rollout_records(
    model: HybridTemporalTargetDurationDecoderModel,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    mode: str,
    threshold: float,
    learned_min_stop_step: int,
    heuristic_min_steps: int,
    heuristic_target_margin: float,
    heuristic_stop_on_entry_count: int,
    heuristic_inside_patience: int,
) -> List[Dict[str, Any]]:
    if mode not in {"pure", "hybrid"}:
        raise ValueError("mode must be 'pure' or 'hybrid'")

    model.eval()
    records: List[Dict[str, Any]] = []
    max_scanpath_len = int(cfg.get("max_scanpath_len", cfg.get("max_rollout_steps", 20)))
    history_len = int(cfg["history_len"])
    global_index = 0

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device)
        cue_id = batch["cue_id"].to(device)
        target_crop_image = batch["target_crop_image"].to(device)
        scanpath_xydur_raw = batch["scanpath_xydur"].to(device)
        scanpath_xydur = prepare_model_scanpath_xydur(scanpath_xydur_raw, cfg)
        scanpath_len = batch["scanpath_len"].to(device)
        target_bbox_norm = batch["target_bbox_norm"].to(device)
        ui_geom = batch["ui_geom"].to(device)
        ui_type_id = batch["ui_type_id"].to(device)
        ui_mask = batch["ui_mask"].to(device)
        ui_crop_images = batch["ui_crop_images"].to(device)

        B = image.size(0)
        generated_history = scanpath_xydur[:, :1, :].clone()  # start from GT f0
        active = torch.ones(B, dtype=torch.bool, device=device)
        stopped = torch.zeros(B, dtype=torch.bool, device=device)
        stop_step = torch.full((B,), -1, dtype=torch.long, device=device)
        stop_reason = ["max_steps" for _ in range(B)]
        stop_probs: List[List[float]] = [[] for _ in range(B)]
        inside_count = torch.zeros(B, dtype=torch.long, device=device)
        inside_streak = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(1, max_scanpath_len):
            if not active.any():
                break

            history_xydur, history_mask = make_history_window(generated_history, history_len)
            step_idx = torch.full((B,), t, dtype=torch.long, device=device)
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
            pred_xy = pred_xy.clamp(0.0, 1.0)
            stop_prob = torch.sigmoid(pred_stop_logit)

            # Always append predictions for active and inactive samples to keep tensor shape.
            generated_history = append_pred_xydur(generated_history, pred_xy, pred_dur, cfg)

            bbox_np = target_bbox_norm.detach().cpu().numpy()
            pred_np = pred_xy.detach().cpu().numpy()
            heuristic_stop = torch.zeros(B, dtype=torch.bool, device=device)
            learned_stop = (stop_prob >= threshold) & (t >= learned_min_stop_step)

            for i in range(B):
                if not active[i]:
                    continue
                stop_probs[i].append(float(stop_prob[i].detach().cpu().item()))
                bbox_m = expand_bbox_np(bbox_np[i].tolist(), heuristic_target_margin)
                inside = point_in_bbox_np(float(pred_np[i, 0]), float(pred_np[i, 1]), bbox_m)
                if inside:
                    inside_count[i] += 1
                    inside_streak[i] += 1
                else:
                    inside_streak[i] = 0

                if mode == "hybrid" and t >= heuristic_min_steps:
                    if int(inside_count[i].item()) >= heuristic_stop_on_entry_count:
                        heuristic_stop[i] = True
                    if int(inside_streak[i].item()) >= heuristic_inside_patience:
                        heuristic_stop[i] = True

            should_stop = active & (learned_stop | heuristic_stop)
            for i in range(B):
                if bool(should_stop[i].item()):
                    stopped[i] = True
                    stop_step[i] = t
                    if bool(learned_stop[i].item()):
                        stop_reason[i] = "learned_stop"
                    elif bool(heuristic_stop[i].item()):
                        if int(inside_streak[i].item()) >= heuristic_inside_patience:
                            stop_reason[i] = "heuristic_inside_streak"
                        else:
                            stop_reason[i] = "heuristic_target_entry_count"
            active = active & (~should_stop)

        generated = generated_history.detach().cpu().numpy()
        gt = scanpath_xydur.detach().cpu().numpy()
        lens = scanpath_len.detach().cpu().numpy()
        bbox = target_bbox_norm.detach().cpu().numpy()

        for i in range(B):
            true_len = int(lens[i])
            pred_len = int(stop_step[i].item()) + 1 if bool(stopped[i].item()) else generated.shape[1]
            pred_points = generated[i, :pred_len, :2].astype(float).tolist()
            gt_points = gt[i, :true_len, :2].astype(float).tolist()
            pred_durations = generated[i, :pred_len, 2].astype(float).tolist()
            gt_durations = gt[i, :true_len, 2].astype(float).tolist()
            records.append({
                "global_index": global_index,
                "batch_index": batch_idx,
                "mode": mode,
                "threshold": float(threshold),
                "gt": gt_points,
                "pred": pred_points,
                "gt_dur_feature": gt_durations,
                "pred_dur_feature": pred_durations,
                "bbox": bbox[i].astype(float).tolist(),
                "gt_len": true_len,
                "pred_len": len(pred_points),
                "stopped": bool(stopped[i].item()),
                "stop_step": int(stop_step[i].item()),
                "stop_reason": stop_reason[i],
                "stop_probs": stop_probs[i],
            })
            global_index += 1

    return records


# ============================================================================
# Metrics
# ============================================================================

def euclidean_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=-1) + 1e-12)


def aligned_points(gt: Sequence[Sequence[float]], pred: Sequence[Sequence[float]], exclude_initial: bool):
    g = np.asarray(gt, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if exclude_initial:
        g = g[1:] if len(g) > 1 else g[:0]
        p = p[1:] if len(p) > 1 else p[:0]
    n = min(len(g), len(p))
    if n <= 0:
        return g[:0], p[:0]
    return g[:n], p[:n]


def sequence_to_cells(points: Sequence[Sequence[float]], grid_size: int, exclude_initial: bool, collapse_repeats: bool = True) -> List[int]:
    pts = list(points)
    if exclude_initial and len(pts) > 0:
        pts = pts[1:]
    cells: List[int] = []
    for x, y in pts:
        cx = int(np.clip(float(x), 0.0, 0.999999) * grid_size)
        cy = int(np.clip(float(y), 0.0, 0.999999) * grid_size)
        cell = cy * grid_size + cx
        if collapse_repeats and cells and cells[-1] == cell:
            continue
        cells.append(cell)
    return cells


def levenshtein_distance(a: Sequence[int], b: Sequence[int]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[n]


def scanmatch_lite(gt, pred, grid_size: int, exclude_initial: bool) -> Dict[str, float]:
    a = sequence_to_cells(gt, grid_size=grid_size, exclude_initial=exclude_initial, collapse_repeats=True)
    b = sequence_to_cells(pred, grid_size=grid_size, exclude_initial=exclude_initial, collapse_repeats=True)
    denom = max(1, max(len(a), len(b)))
    sed = levenshtein_distance(a, b)
    sed_norm = sed / denom
    return {
        "scanmatch_lite": float(max(0.0, 1.0 - sed_norm)),
        "sed": float(sed),
        "sed_norm": float(sed_norm),
        "scanmatch_gt_symbols": len(a),
        "scanmatch_pred_symbols": len(b),
    }


def dtw_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    if len(gt) == 0 and len(pred) == 0:
        return 0.0
    if len(gt) == 0 or len(pred) == 0:
        return math.sqrt(2.0)
    m, n = len(gt), len(pred)
    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = float(np.linalg.norm(gt[i - 1] - pred[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[m, n] / max(1, m + n))


def resample_by_index(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((n, 2), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points[:1], n, axis=0)
    old_x = np.linspace(0.0, 1.0, len(points))
    new_x = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 2), dtype=np.float64)
    out[:, 0] = np.interp(new_x, old_x, points[:, 0])
    out[:, 1] = np.interp(new_x, old_x, points[:, 1])
    return out


def multimatch_lite(gt, pred, exclude_initial: bool, resample_n: int = 20) -> Dict[str, float]:
    g = np.asarray(gt, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if exclude_initial:
        g = g[1:] if len(g) > 1 else g[:0]
        p = p[1:] if len(p) > 1 else p[:0]

    if len(g) == 0 or len(p) == 0:
        return {
            "multimatch_vector": 0.0,
            "multimatch_direction": 0.0,
            "multimatch_length": 0.0,
            "multimatch_position": 0.0,
            "multimatch_mean": 0.0,
        }

    n = max(2, int(resample_n))
    gr = resample_by_index(g, n)
    pr = resample_by_index(p, n)

    max_dist = math.sqrt(2.0)
    position_sim = 1.0 - float(np.mean(euclidean_np(gr, pr)) / max_dist)
    position_sim = float(np.clip(position_sim, 0.0, 1.0))

    gv = np.diff(gr, axis=0)
    pv = np.diff(pr, axis=0)
    g_len = np.linalg.norm(gv, axis=1)
    p_len = np.linalg.norm(pv, axis=1)

    vector_diff = np.linalg.norm(gv - pv, axis=1)
    vector_sim = 1.0 - float(np.mean(vector_diff) / (2.0 * max_dist))
    vector_sim = float(np.clip(vector_sim, 0.0, 1.0))

    length_sim = 1.0 - float(np.mean(np.abs(g_len - p_len)) / max_dist)
    length_sim = float(np.clip(length_sim, 0.0, 1.0))

    valid = (g_len > 1e-8) & (p_len > 1e-8)
    if valid.any():
        cos = np.sum(gv[valid] * pv[valid], axis=1) / (g_len[valid] * p_len[valid] + 1e-12)
        direction_sim = float(np.mean((np.clip(cos, -1.0, 1.0) + 1.0) / 2.0))
    else:
        direction_sim = 0.0

    mm = float(np.mean([vector_sim, direction_sim, length_sim, position_sim]))
    return {
        "multimatch_vector": vector_sim,
        "multimatch_direction": direction_sim,
        "multimatch_length": length_sim,
        "multimatch_position": position_sim,
        "multimatch_mean": mm,
    }


def compute_per_record_metrics(
    records: List[Dict[str, Any]],
    margins: List[float],
    grid_size: int,
    exclude_initial: bool,
    multimatch_resample_n: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        gt = r["gt"]
        pred = r["pred"]
        bbox = r["bbox"]
        g_aligned, p_aligned = aligned_points(gt, pred, exclude_initial=exclude_initial)

        if len(g_aligned) > 0:
            d = euclidean_np(g_aligned, p_aligned)
            ade = float(np.mean(d))
            rmse = float(np.sqrt(np.mean(d ** 2)))
            fde = float(d[-1])
        else:
            ade = rmse = fde = float("nan")

        gt_np = np.asarray(gt, dtype=np.float64)
        pred_np = np.asarray(pred, dtype=np.float64)
        dtw = dtw_distance(
            gt_np[1:] if exclude_initial and len(gt_np) > 1 else gt_np,
            pred_np[1:] if exclude_initial and len(pred_np) > 1 else pred_np,
        )
        dtw_sim = float(np.clip(1.0 - dtw / math.sqrt(2.0), 0.0, 1.0))

        sm = scanmatch_lite(gt, pred, grid_size=grid_size, exclude_initial=exclude_initial)
        mm = multimatch_lite(gt, pred, exclude_initial=exclude_initial, resample_n=multimatch_resample_n)

        row = {
            "global_index": r["global_index"],
            "mode": r["mode"],
            "threshold": r["threshold"],
            "gt_len": r["gt_len"],
            "pred_len": r["pred_len"],
            "length_abs_error": abs(int(r["pred_len"]) - int(r["gt_len"])),
            "stopped": bool(r["stopped"]),
            "timeout": not bool(r["stopped"]),
            "stop_step": int(r["stop_step"]),
            "stop_reason": r.get("stop_reason", "unknown"),
            "ade": ade,
            "rmse": rmse,
            "fde": fde,
            "dtw_distance": dtw,
            "dtw_similarity": dtw_sim,
            **sm,
            **mm,
        }

        for margin in margins:
            bbox_m = expand_bbox_np(bbox, margin)
            gt_first = first_hit_step_np(gt, bbox_m)
            pred_first = first_hit_step_np(pred, bbox_m)
            row[f"gt_any_hit_m{margin:.2f}"] = gt_first >= 0
            row[f"pred_any_hit_m{margin:.2f}"] = pred_first >= 0
            row[f"gt_final_hit_m{margin:.2f}"] = final_hit_np(gt, bbox_m)
            row[f"pred_final_hit_m{margin:.2f}"] = final_hit_np(pred, bbox_m)
            row[f"gt_first_hit_step_m{margin:.2f}"] = gt_first
            row[f"pred_first_hit_step_m{margin:.2f}"] = pred_first
            if pred_first >= 0 and gt_first >= 0:
                row[f"first_hit_step_error_m{margin:.2f}"] = abs(pred_first - gt_first)
            else:
                row[f"first_hit_step_error_m{margin:.2f}"] = float("nan")

        # A simple diagnostic: early stop before target hit at the default 0.05 margin.
        bbox_default = expand_bbox_np(bbox, 0.05)
        pred_first_default = first_hit_step_np(pred, bbox_default)
        row["early_stop_before_target_m0.05"] = bool(r["stopped"]) and pred_first_default < 0
        out.append(row)
    return out


def mean_finite(values: List[Any]) -> float:
    nums = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            nums.append(x)
    if not nums:
        return float("nan")
    return float(np.mean(nums))


def aggregate_metrics(per_record: List[Dict[str, Any]], threshold: float, margins: List[float], mode: str) -> Dict[str, Any]:
    n = len(per_record)
    if n == 0:
        return {"mode": mode, "threshold": threshold, "num_samples": 0}

    result: Dict[str, Any] = {
        "mode": mode,
        "threshold": float(threshold),
        "num_samples": n,
        "pred_len_mean": mean_finite([r["pred_len"] for r in per_record]),
        "gt_len_mean": mean_finite([r["gt_len"] for r in per_record]),
        "length_abs_error_mean": mean_finite([r["length_abs_error"] for r in per_record]),
        "stopped_rate": float(np.mean([bool(r["stopped"]) for r in per_record])),
        "timeout_rate": float(np.mean([bool(r["timeout"]) for r in per_record])),
        "early_stop_before_target_rate_m0.05": float(np.mean([bool(r["early_stop_before_target_m0.05"]) for r in per_record])),
        "ade_mean": mean_finite([r["ade"] for r in per_record]),
        "rmse_mean": mean_finite([r["rmse"] for r in per_record]),
        "fde_mean": mean_finite([r["fde"] for r in per_record]),
        "dtw_distance_mean": mean_finite([r["dtw_distance"] for r in per_record]),
        "dtw_similarity_mean": mean_finite([r["dtw_similarity"] for r in per_record]),
        "scanmatch_lite_mean": mean_finite([r["scanmatch_lite"] for r in per_record]),
        "sed_norm_mean": mean_finite([r["sed_norm"] for r in per_record]),
        "multimatch_vector_mean": mean_finite([r["multimatch_vector"] for r in per_record]),
        "multimatch_direction_mean": mean_finite([r["multimatch_direction"] for r in per_record]),
        "multimatch_length_mean": mean_finite([r["multimatch_length"] for r in per_record]),
        "multimatch_position_mean": mean_finite([r["multimatch_position"] for r in per_record]),
        "multimatch_mean_mean": mean_finite([r["multimatch_mean"] for r in per_record]),
    }

    for margin in margins:
        key = f"m{margin:.2f}"
        result[f"gt_any_hit_rate_{key}"] = float(np.mean([bool(r[f"gt_any_hit_m{margin:.2f}"]) for r in per_record]))
        result[f"any_hit_rate_{key}"] = float(np.mean([bool(r[f"pred_any_hit_m{margin:.2f}"]) for r in per_record]))
        result[f"gt_final_hit_rate_{key}"] = float(np.mean([bool(r[f"gt_final_hit_m{margin:.2f}"]) for r in per_record]))
        result[f"final_hit_rate_{key}"] = float(np.mean([bool(r[f"pred_final_hit_m{margin:.2f}"]) for r in per_record]))
        result[f"first_hit_step_error_mean_{key}"] = mean_finite([r[f"first_hit_step_error_m{margin:.2f}"] for r in per_record])
        hit_steps = [r[f"pred_first_hit_step_m{margin:.2f}"] for r in per_record if int(r[f"pred_first_hit_step_m{margin:.2f}"]) >= 0]
        result[f"pred_first_hit_step_mean_{key}"] = mean_finite(hit_steps)

    reason_counts: Dict[str, int] = {}
    for r in per_record:
        reason = str(r.get("stop_reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items()):
        result[f"stop_reason_count_{reason}"] = int(count)
        result[f"stop_reason_rate_{reason}"] = float(count / max(1, n))

    # A simple single number for threshold selection. Keep individual metrics for reporting.
    hit = result.get("any_hit_rate_m0.05", float("nan"))
    sm = result.get("scanmatch_lite_mean", float("nan"))
    mm = result.get("multimatch_mean_mean", float("nan"))
    dtw = result.get("dtw_similarity_mean", float("nan"))
    early = result.get("early_stop_before_target_rate_m0.05", 0.0)
    timeout = result.get("timeout_rate", 0.0)
    parts = [x for x in [hit, sm, mm, dtw] if math.isfinite(float(x))]
    result["professional_score"] = float(np.mean(parts) - 0.10 * early - 0.05 * timeout) if parts else float("nan")
    return result


# ============================================================================
# Visualization
# ============================================================================

def find_image_path(image_dir: str, img_name: str) -> str:
    path = Path(image_dir) / img_name
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return str(path)


def draw_scanpath_overlay(
    image_path: str,
    gt_points: Sequence[Sequence[float]],
    pred_points: Sequence[Sequence[float]],
    target_bbox_norm: Sequence[float],
    out_path: str,
    title: str,
    margin: float,
    stop_step: int,
    stop_probs: Optional[List[float]] = None,
):
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img)
    h, w = img_np.shape[:2]

    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    if len(gt) > 0:
        gt_plot = gt.copy()
        gt_plot[:, 0] = np.clip(gt_plot[:, 0], 0.0, 1.0) * w
        gt_plot[:, 1] = np.clip(gt_plot[:, 1], 0.0, 1.0) * h
    else:
        gt_plot = np.zeros((0, 2))
    if len(pred) > 0:
        pred_plot = pred.copy()
        pred_plot[:, 0] = np.clip(pred_plot[:, 0], 0.0, 1.0) * w
        pred_plot[:, 1] = np.clip(pred_plot[:, 1], 0.0, 1.0) * h
    else:
        pred_plot = np.zeros((0, 2))

    bbox_exact = [float(v) for v in target_bbox_norm]
    bbox_exp = expand_bbox_np(bbox_exact, margin)

    plt.figure(figsize=(12, 7))
    plt.imshow(img_np)
    plt.title(title, fontsize=9)

    x0, y0, x1, y1 = bbox_exact
    rect = plt.Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, edgecolor="lime", linewidth=2, label="Target exact")
    plt.gca().add_patch(rect)
    ex0, ey0, ex1, ey1 = bbox_exp
    rect2 = plt.Rectangle((ex0 * w, ey0 * h), (ex1 - ex0) * w, (ey1 - ey0) * h, fill=False, edgecolor="yellow", linewidth=2, linestyle="--", label=f"Target + margin {margin}")
    plt.gca().add_patch(rect2)

    if len(gt_plot) > 0:
        plt.plot(gt_plot[:, 0], gt_plot[:, 1], "-o", linewidth=2, markersize=4, label="GT human", color="deepskyblue")
        for i, (x, y) in enumerate(gt_plot):
            plt.text(x + 4, y + 4, f"G{i}", color="deepskyblue", fontsize=8)
    if len(pred_plot) > 0:
        plt.plot(pred_plot[:, 0], pred_plot[:, 1], "-o", linewidth=2, markersize=4, label="Pred rollout", color="orangered")
        for i, (x, y) in enumerate(pred_plot):
            label = f"P{i}"
            if stop_step >= 0 and i == stop_step:
                label += " STOP"
            plt.text(x + 4, y - 6, label, color="orangered", fontsize=8)

    if stop_probs:
        lines = [f"t={i+1}: stop={p:.3f}" for i, p in enumerate(stop_probs[:14])]
        plt.gcf().text(0.02, 0.02, "\n".join(lines), fontsize=8, bbox=dict(facecolor="white", alpha=0.75))

    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def choose_visual_indices(num_items: int, num_visuals: int, seed: int, explicit: Optional[List[int]]) -> List[int]:
    if explicit is not None:
        return [i for i in explicit if 0 <= i < num_items]
    rng = random.Random(seed)
    indices = list(range(num_items))
    if num_visuals >= num_items:
        return indices
    return rng.sample(indices, num_visuals)


def save_visuals(
    records_by_key: Dict[Tuple[str, float], List[Dict[str, Any]]],
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    output_dir: str,
    visual_indices: List[int],
    viz_thresholds: List[float],
    target_margin: float,
):
    root = os.path.join(output_dir, "visuals")
    os.makedirs(root, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    for threshold in viz_thresholds:
        for mode in ["pure", "hybrid"]:
            key = (mode, float(threshold))
            if key not in records_by_key:
                continue
            records = records_by_key[key]
            mode_dir = os.path.join(root, f"{mode}_thr{safe_float_name(threshold)}")
            os.makedirs(mode_dir, exist_ok=True)
            for rank, idx in enumerate(visual_indices):
                if idx < 0 or idx >= len(records) or idx >= len(samples):
                    continue
                r = records[idx]
                sample = samples[idx]
                img_name = sample.get("img_name")
                image_path = find_image_path(cfg["image_dir"], img_name)
                bbox_m = expand_bbox_np(r["bbox"], target_margin)
                gt_hit = points_any_hit_np(r["gt"], bbox_m)
                pred_hit = points_any_hit_np(r["pred"], bbox_m)
                gt_first = first_hit_step_np(r["gt"], bbox_m)
                pred_first = first_hit_step_np(r["pred"], bbox_m)
                title = (
                    f"{mode} thr={threshold:.2f} idx={idx} trial={sample.get('trial_id')} cue={sample.get('cue')} | "
                    f"GT len={r['gt_len']} Pred len={r['pred_len']} stopped={r['stopped']} "
                    f"step={r['stop_step']} reason={r.get('stop_reason')} | "
                    f"GT hit={gt_hit}@{gt_first} Pred hit={pred_hit}@{pred_first}"
                )
                out_png = os.path.join(mode_dir, f"{rank:03d}_idx{idx:05d}_{mode}_thr{safe_float_name(threshold)}.png")
                draw_scanpath_overlay(
                    image_path=image_path,
                    gt_points=r["gt"],
                    pred_points=r["pred"],
                    target_bbox_norm=r["bbox"],
                    out_path=out_png,
                    title=title,
                    margin=target_margin,
                    stop_step=int(r.get("stop_step", -1)),
                    stop_probs=r.get("stop_probs", []),
                )
                rec = {
                    "mode": mode,
                    "threshold": float(threshold),
                    "split_index": idx,
                    "trial_id": sample.get("trial_id"),
                    "img_name": img_name,
                    "cue": sample.get("cue"),
                    "output_png": out_png,
                    "gt_hit_margin": gt_hit,
                    "pred_hit_margin": pred_hit,
                    "gt_first_hit_margin": gt_first,
                    "pred_first_hit_margin": pred_first,
                    **r,
                }
                out_json = out_png.replace(".png", ".json")
                save_json(rec, out_json)
                summary.append({**rec, "output_json": out_json})
                print(f"saved visual: {out_png}")

    save_json({"records": summary}, os.path.join(root, "visualization_summary.json"))


# ============================================================================
# Printing / main
# ============================================================================

def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": row.get("mode"),
        "threshold": row.get("threshold"),
        "num_samples": row.get("num_samples"),
        "professional_score": row.get("professional_score"),
        "pred_len_mean": row.get("pred_len_mean"),
        "gt_len_mean": row.get("gt_len_mean"),
        "length_abs_error_mean": row.get("length_abs_error_mean"),
        "stopped_rate": row.get("stopped_rate"),
        "timeout_rate": row.get("timeout_rate"),
        "early_stop_before_target_rate_m0.05": row.get("early_stop_before_target_rate_m0.05"),
        "ade_mean": row.get("ade_mean"),
        "rmse_mean": row.get("rmse_mean"),
        "fde_mean": row.get("fde_mean"),
        "any_hit_rate_m0.05": row.get("any_hit_rate_m0.05"),
        "final_hit_rate_m0.05": row.get("final_hit_rate_m0.05"),
        "pred_first_hit_step_mean_m0.05": row.get("pred_first_hit_step_mean_m0.05"),
        "scanmatch_lite_mean": row.get("scanmatch_lite_mean"),
        "sed_norm_mean": row.get("sed_norm_mean"),
        "dtw_similarity_mean": row.get("dtw_similarity_mean"),
        "multimatch_vector_mean": row.get("multimatch_vector_mean"),
        "multimatch_direction_mean": row.get("multimatch_direction_mean"),
        "multimatch_length_mean": row.get("multimatch_length_mean"),
        "multimatch_position_mean": row.get("multimatch_position_mean"),
        "multimatch_mean_mean": row.get("multimatch_mean_mean"),
        "stop_reason_rate_learned_stop": row.get("stop_reason_rate_learned_stop"),
        "stop_reason_rate_heuristic_inside_streak": row.get("stop_reason_rate_heuristic_inside_streak"),
        "stop_reason_rate_heuristic_target_entry_count": row.get("stop_reason_rate_heuristic_target_entry_count"),
        "stop_reason_rate_max_steps": row.get("stop_reason_rate_max_steps"),
    }


def print_compact_table(rows: List[Dict[str, Any]]):
    print("\n" + "=" * 160)
    print("PURE vs HYBRID SUMMARY")
    print("=" * 160)
    header = (
        f"{'mode':>7} {'thr':>5} {'score':>7} {'pred':>6} {'gt':>6} {'timeout':>8} "
        f"{'early':>8} {'ADE':>7} {'hit@.05':>8} {'SM':>7} {'DTW':>7} {'MM':>7}"
    )
    print(header)
    for r in rows:
        def f(k, default=0.0):
            v = r.get(k, default)
            try:
                return float(v)
            except Exception:
                return float(default)
        print(
            f"{str(r.get('mode')):>7} "
            f"{f('threshold'):5.2f} "
            f"{f('professional_score'):7.3f} "
            f"{f('pred_len_mean'):6.2f} "
            f"{f('gt_len_mean'):6.2f} "
            f"{f('timeout_rate'):8.3f} "
            f"{f('early_stop_before_target_rate_m0.05'):8.3f} "
            f"{f('ade_mean'):7.3f} "
            f"{f('any_hit_rate_m0.05'):8.3f} "
            f"{f('scanmatch_lite_mean'):7.3f} "
            f"{f('dtw_similarity_mean'):7.3f} "
            f"{f('multimatch_mean_mean'):7.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--modes", type=str, default="pure,hybrid")
    parser.add_argument("--thresholds", type=str, default="0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--margins", type=str, default="0.00,0.03,0.05,0.10")
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--exclude_initial_fixation", action="store_true")
    parser.add_argument("--multimatch_resample_n", type=int, default=20)
    parser.add_argument("--save_per_sample", action="store_true")

    # Generation rules.
    parser.add_argument("--learned_min_stop_step", type=int, default=1)
    parser.add_argument("--heuristic_min_steps", type=int, default=3)
    parser.add_argument("--heuristic_target_margin", type=float, default=0.02)
    parser.add_argument("--heuristic_stop_on_entry_count", type=int, default=2)
    parser.add_argument("--heuristic_inside_patience", type=int, default=3)

    # Visualization.
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_visuals", type=int, default=20)
    parser.add_argument("--visual_seed", type=int, default=123)
    parser.add_argument("--visual_indices", type=str, default=None)
    parser.add_argument("--viz_thresholds", type=str, default="")
    parser.add_argument("--target_margin", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    user_cfg = load_json(args.config)
    state_dict, ckpt = load_checkpoint_state(args.checkpoint)
    cfg = merge_checkpoint_config(user_cfg, ckpt)
    device = get_device(args.device)
    thresholds = parse_floats(args.thresholds)
    margins = parse_floats(args.margins)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in {"pure", "hybrid"}:
            raise ValueError(f"Unknown mode: {m}")
    max_eval_samples = args.max_eval_samples if args.max_eval_samples > 0 else None

    print("=" * 100)
    print("Temporal target-crop movement eval + visualization")
    print("=" * 100)
    print("Config:    ", args.config)
    print("Checkpoint:", args.checkpoint)
    print("Split:     ", args.split)
    print("Modes:     ", modes)
    print("Thresholds:", thresholds)
    print("Margins:   ", margins)
    print("Device:    ", device)
    print("Output:    ", args.output_dir)
    print("Checkpoint epoch:", ckpt.get("epoch"))

    loader, samples = prepare_dataset_and_loader(
        cfg=cfg,
        ckpt=ckpt,
        split_name=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_samples=max_eval_samples,
    )
    print("eval samples:", len(samples))

    model = build_model_from_config(cfg, ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    if missing[:10]:
        print("first missing keys:", missing[:10])
    if unexpected[:10]:
        print("first unexpected keys:", unexpected[:10])
    model.to(device)
    model.eval()

    all_rows: List[Dict[str, Any]] = []
    compact_rows: List[Dict[str, Any]] = []
    records_by_key: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    best_by_mode: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        for threshold in thresholds:
            print("\n" + "=" * 100)
            print(f"Generating mode={mode}, threshold={threshold}")
            print("=" * 100)
            records = generate_rollout_records(
                model=model,
                loader=loader,
                cfg=cfg,
                device=device,
                mode=mode,
                threshold=threshold,
                learned_min_stop_step=args.learned_min_stop_step,
                heuristic_min_steps=args.heuristic_min_steps,
                heuristic_target_margin=args.heuristic_target_margin,
                heuristic_stop_on_entry_count=args.heuristic_stop_on_entry_count,
                heuristic_inside_patience=args.heuristic_inside_patience,
            )
            records_by_key[(mode, float(threshold))] = records

            per_record = compute_per_record_metrics(
                records=records,
                margins=margins,
                grid_size=args.grid_size,
                exclude_initial=args.exclude_initial_fixation,
                multimatch_resample_n=args.multimatch_resample_n,
            )
            result = aggregate_metrics(per_record, threshold=threshold, margins=margins, mode=mode)
            result.update({
                "learned_min_stop_step": args.learned_min_stop_step,
                "heuristic_min_steps": args.heuristic_min_steps if mode == "hybrid" else None,
                "heuristic_target_margin": args.heuristic_target_margin if mode == "hybrid" else None,
                "heuristic_stop_on_entry_count": args.heuristic_stop_on_entry_count if mode == "hybrid" else None,
                "heuristic_inside_patience": args.heuristic_inside_patience if mode == "hybrid" else None,
                "grid_size": args.grid_size,
                "exclude_initial_fixation": bool(args.exclude_initial_fixation),
            })
            print(json.dumps(compact_row(result), indent=2))
            all_rows.append(result)
            compact_rows.append(compact_row(result))

            if args.save_per_sample:
                per_path = os.path.join(args.output_dir, f"{args.split}_{mode}_thr{threshold:.2f}_per_sample_metrics.jsonl")
                save_jsonl(per_record, per_path)
                rec_path = os.path.join(args.output_dir, f"{args.split}_{mode}_thr{threshold:.2f}_rollout_records.jsonl")
                save_jsonl(records, rec_path)

            if mode not in best_by_mode:
                best_by_mode[mode] = result
            else:
                if float(result.get("professional_score", -1e9)) > float(best_by_mode[mode].get("professional_score", -1e9)):
                    best_by_mode[mode] = result

    save_csv(all_rows, os.path.join(args.output_dir, f"{args.split}_full_metrics.csv"))
    save_csv(compact_rows, os.path.join(args.output_dir, f"{args.split}_compact_metrics.csv"))
    save_json({
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split,
        "modes": modes,
        "thresholds": thresholds,
        "margins": margins,
        "generation_rules": {
            "learned_min_stop_step": args.learned_min_stop_step,
            "heuristic_min_steps": args.heuristic_min_steps,
            "heuristic_target_margin": args.heuristic_target_margin,
            "heuristic_stop_on_entry_count": args.heuristic_stop_on_entry_count,
            "heuristic_inside_patience": args.heuristic_inside_patience,
        },
        "duration_settings": {
            "duration_target": cfg.get("duration_target", "log1p"),
            "duration_scale": cfg.get("duration_scale", 1.0),
            "duration_output": cfg.get("duration_output", "softplus"),
            "duration_feature_clip": cfg.get("duration_feature_clip", None),
        },
        "metric_settings": {
            "grid_size": args.grid_size,
            "exclude_initial_fixation": bool(args.exclude_initial_fixation),
            "multimatch_resample_n": args.multimatch_resample_n,
            "scanmatch_note": "ScanMatch-lite discretizes normalized coordinates to grid cells and uses normalized edit distance.",
            "multimatch_note": "MultiMatch-lite reports vector/direction/length/position similarities after index-based resampling; duration is excluded.",
        },
        "best_by_mode_professional_score": best_by_mode,
        "compact_results": compact_rows,
    }, os.path.join(args.output_dir, f"{args.split}_summary.json"))

    print_compact_table(compact_rows)

    if args.visualize:
        explicit = parse_ints_or_none(args.visual_indices)
        visual_indices = choose_visual_indices(len(samples), args.num_visuals, args.visual_seed, explicit)
        if args.viz_thresholds.strip():
            viz_thresholds = parse_floats(args.viz_thresholds)
        else:
            viz_thresholds = sorted(set(float(v["threshold"]) for v in best_by_mode.values()))
        print("\nvisual indices:", visual_indices)
        print("visual thresholds:", viz_thresholds)
        save_visuals(
            records_by_key=records_by_key,
            samples=samples,
            cfg=cfg,
            output_dir=args.output_dir,
            visual_indices=visual_indices,
            viz_thresholds=viz_thresholds,
            target_margin=args.target_margin,
        )

    print("\nSaved files:")
    print(os.path.join(args.output_dir, f"{args.split}_full_metrics.csv"))
    print(os.path.join(args.output_dir, f"{args.split}_compact_metrics.csv"))
    print(os.path.join(args.output_dir, f"{args.split}_summary.json"))
    if args.visualize:
        print(os.path.join(args.output_dir, "visuals"))


if __name__ == "__main__":
    main()
