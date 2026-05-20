#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from dataset import (
    filter_target_present_trials,
    build_hybrid_trial_samples,
)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_mode_threshold(path: Path, split: str) -> Tuple[str, float]:
    # Example: test_hybrid_thr0.50_rollout_records.jsonl
    name = path.name
    m = re.match(rf"{re.escape(split)}_(pure|hybrid)_thr([0-9.]+)_rollout_records\.jsonl$", name)
    if not m:
        return "unknown", float("nan")
    return m.group(1), float(m.group(2))


def load_eval_samples(config_path: str, split: str, max_eval_samples: int = 0):
    cfg = load_json(config_path)

    trials = read_jsonl_gz(cfg["trials_path"])
    if cfg.get("target_present_only", True):
        trials = filter_target_present_trials(trials)

    split_json_path = cfg.get("split_json_path")
    if not split_json_path or not os.path.exists(split_json_path):
        out_dir = cfg.get("output_dir", "")
        candidate = os.path.join(out_dir, "split.json")
        if os.path.exists(candidate):
            split_json_path = candidate

    if not split_json_path or not os.path.exists(split_json_path):
        raise FileNotFoundError(f"Cannot find split_json_path: {split_json_path}")

    split_obj = load_json(split_json_path)
    split_key = f"{split}_trial_ids"
    if split_key not in split_obj:
        raise KeyError(f"{split_key} not found in {split_json_path}")

    wanted = set(split_obj[split_key])
    eval_trials = [t for t in trials if t.get("trial_id") in wanted]
    samples = build_hybrid_trial_samples(eval_trials)

    if max_eval_samples and max_eval_samples > 0:
        samples = samples[:max_eval_samples]

    print("=" * 100)
    print("BRIDGE METRIC SAMPLE CHECK")
    print("config:", config_path)
    print("split_json_path:", split_json_path)
    print("wanted trial ids:", len(wanted))
    print("matched eval trials:", len(eval_trials))
    print("eval samples:", len(samples))
    print("=" * 100)

    return cfg, samples


def image_size_for_sample(cfg: Dict[str, Any], sample: Dict[str, Any]) -> Tuple[int, int]:
    img_name = sample.get("img_name")
    if not img_name:
        return int(cfg.get("image_size", 224)), int(cfg.get("image_size", 224))

    path = Path(cfg["image_dir"]) / img_name
    if not path.exists():
        return int(cfg.get("image_size", 224)), int(cfg.get("image_size", 224))

    with Image.open(path) as img:
        w, h = img.size
    return int(w), int(h)


def point_to_bbox_distance_px(x: float, y: float, bbox: List[float], w: int, h: int) -> float:
    """
    x,y and bbox are normalized.
    Returns pixel distance from point to bbox. 0 if inside.
    """
    px = float(x) * w
    py = float(y) * h
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = x0 * w, y0 * h, x1 * w, y1 * h

    dx = max(bx0 - px, 0.0, px - bx1)
    dy = max(by0 - py, 0.0, py - by1)
    return math.sqrt(dx * dx + dy * dy)


def last3_target_success_32px(pred: List[List[float]], bbox: List[float], w: int, h: int, radius_px: float = 32.0) -> bool:
    if len(pred) < 3:
        return False
    last3 = pred[-3:]
    for x, y in last3:
        d = point_to_bbox_distance_px(float(x), float(y), bbox, w, h)
        if d > radius_px:
            return False
    return True


def add_gaussian_to_map(fmap: np.ndarray, cx: float, cy: float, sigma: float):
    """
    Adds one Gaussian centered at cx,cy in map coordinates.
    Uses a local window for speed.
    """
    h, w = fmap.shape
    if sigma <= 0:
        return

    radius = int(max(3, math.ceil(3.0 * sigma)))
    x0 = max(0, int(math.floor(cx)) - radius)
    x1 = min(w, int(math.floor(cx)) + radius + 1)
    y0 = max(0, int(math.floor(cy)) - radius)
    y1 = min(h, int(math.floor(cy)) + radius + 1)

    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    fmap[y0:y1, x0:x1] += g.astype(np.float32)


def compute_nss(
    pred: List[List[float]],
    gt: List[List[float]],
    w: int,
    h: int,
    sigma_px: float = 32.0,
    max_side: int = 512,
) -> float:
    """
    Approximate NSS:
    - build predicted fixation density map
    - normalize to zero mean/unit std
    - sample at GT fixation locations
    """
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")

    scale = min(1.0, float(max_side) / float(max(w, h)))
    mw = max(8, int(round(w * scale)))
    mh = max(8, int(round(h * scale)))
    sigma = max(1.0, sigma_px * scale)

    fmap = np.zeros((mh, mw), dtype=np.float32)

    for x, y in pred:
        cx = float(np.clip(x, 0.0, 1.0)) * (mw - 1)
        cy = float(np.clip(y, 0.0, 1.0)) * (mh - 1)
        add_gaussian_to_map(fmap, cx, cy, sigma)

    mean = float(fmap.mean())
    std = float(fmap.std())
    if std < 1e-8:
        return float("nan")

    norm = (fmap - mean) / std

    vals = []
    for x, y in gt:
        gx = int(round(float(np.clip(x, 0.0, 1.0)) * (mw - 1)))
        gy = int(round(float(np.clip(y, 0.0, 1.0)) * (mh - 1)))
        vals.append(float(norm[gy, gx]))

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def mean_finite(vals: List[Any]) -> float:
    nums = []
    for v in vals:
        try:
            x = float(v)
            if math.isfinite(x):
                nums.append(x)
        except Exception:
            pass
    if not nums:
        return float("nan")
    return float(np.mean(nums))


def process_rollout_file(
    rollout_path: Path,
    checkpoint: str,
    split: str,
    cfg: Dict[str, Any],
    samples: List[Dict[str, Any]],
    radius_px: float,
    sigma_px: float,
    max_side: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode, threshold = parse_mode_threshold(rollout_path, split)
    records = read_jsonl(rollout_path)

    rows = []

    for r in records:
        idx = int(r.get("global_index", len(rows)))
        sample = samples[idx] if 0 <= idx < len(samples) else {}
        w, h = image_size_for_sample(cfg, sample)

        pred = r.get("pred", [])
        gt = r.get("gt", [])
        bbox = r.get("bbox", [])

        last3 = last3_target_success_32px(pred, bbox, w, h, radius_px=radius_px)
        nss = compute_nss(pred, gt, w, h, sigma_px=sigma_px, max_side=max_side)

        rows.append({
            "checkpoint": checkpoint,
            "mode": mode,
            "threshold": threshold,
            "global_index": idx,
            "trial_id": sample.get("trial_id"),
            "img_name": sample.get("img_name"),
            "gt_len": r.get("gt_len"),
            "pred_len": r.get("pred_len"),
            "last3_target_success_32px": bool(last3),
            "nss": nss,
            "image_w": w,
            "image_h": h,
            "source_rollout": str(rollout_path),
        })

    summary = {
        "checkpoint": checkpoint,
        "mode": mode,
        "threshold": threshold,
        "num_samples": len(rows),
        "last3_target_success_32px_rate": mean_finite([1.0 if r["last3_target_success_32px"] else 0.0 for r in rows]),
        "nss_mean": mean_finite([r["nss"] for r in rows]),
        "pred_len_mean": mean_finite([r["pred_len"] for r in rows]),
        "gt_len_mean": mean_finite([r["gt_len"] for r in rows]),
        "source_rollout": str(rollout_path),
    }

    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--radius_px", type=float, default=32.0)
    parser.add_argument("--sigma_px", type=float, default=32.0)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--max_eval_samples", type=int, default=0)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir) if args.out_dir else eval_root / "seekui_bridge_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, samples = load_eval_samples(args.config, args.split, max_eval_samples=args.max_eval_samples)

    checkpoints = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    ckpt_dirs = sorted([p for p in eval_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if checkpoints:
        keep = set(checkpoints)
        ckpt_dirs = [p for p in ckpt_dirs if p.name in keep]

    all_rows = []
    all_summaries = []

    for ckpt_dir in ckpt_dirs:
        rollout_files = sorted(ckpt_dir.glob(f"{args.split}_*_thr*_rollout_records.jsonl"))
        if not rollout_files:
            print("SKIP no rollout files:", ckpt_dir)
            continue

        for rf in rollout_files:
            print("processing:", rf)
            rows, summary = process_rollout_file(
                rollout_path=rf,
                checkpoint=ckpt_dir.name,
                split=args.split,
                cfg=cfg,
                samples=samples,
                radius_px=args.radius_px,
                sigma_px=args.sigma_px,
                max_side=args.max_side,
            )
            all_rows.extend(rows)
            all_summaries.append(summary)

    if not all_summaries:
        raise RuntimeError(f"No rollout record files found under {eval_root}")

    # rank summaries
    ranked_last3 = sorted(
        all_summaries,
        key=lambda r: (
            float(r.get("last3_target_success_32px_rate", -1)),
            float(r.get("nss_mean", -999)),
        ),
        reverse=True,
    )

    ranked_nss = sorted(
        all_summaries,
        key=lambda r: float(r.get("nss_mean", -999)),
        reverse=True,
    )

    save_csv(all_rows, out_dir / "bridge_per_sample_metrics.csv")
    save_csv(all_summaries, out_dir / "bridge_summary_all.csv")
    save_csv(ranked_last3, out_dir / "ranked_by_last3_success_32px.csv")
    save_csv(ranked_nss, out_dir / "ranked_by_nss.csv")
    save_json({
        "config": args.config,
        "eval_root": args.eval_root,
        "split": args.split,
        "radius_px": args.radius_px,
        "sigma_px": args.sigma_px,
        "max_side": args.max_side,
        "num_summary_rows": len(all_summaries),
        "notes": {
            "last3_target_success_32px": "Success if the last three predicted fixations are within 32 pixels of the target bounding box.",
            "nss": "Predicted fixations are converted to a Gaussian map, normalized, and sampled at GT fixation locations. Map max side is capped for speed.",
            "comparison": "These are SeekUI-style bridge metrics, not an exact reproduction of the full SeekUI evaluation protocol."
        }
    }, out_dir / "bridge_metric_info.json")

    print("\n" + "=" * 120)
    print("TOP BY LAST3 TARGET SUCCESS @ 32PX")
    print("=" * 120)
    for r in ranked_last3[:20]:
        print(
            f"{r['checkpoint']:>10} {r['mode']:>6} thr={float(r['threshold']):.2f} "
            f"last3={float(r['last3_target_success_32px_rate']):.4f} "
            f"NSS={float(r['nss_mean']):.4f} "
            f"pred_len={float(r['pred_len_mean']):.2f} gt_len={float(r['gt_len_mean']):.2f}"
        )

    print("\n" + "=" * 120)
    print("TOP BY NSS")
    print("=" * 120)
    for r in ranked_nss[:20]:
        print(
            f"{r['checkpoint']:>10} {r['mode']:>6} thr={float(r['threshold']):.2f} "
            f"NSS={float(r['nss_mean']):.4f} "
            f"last3={float(r['last3_target_success_32px_rate']):.4f} "
            f"pred_len={float(r['pred_len_mean']):.2f} gt_len={float(r['gt_len_mean']):.2f}"
        )

    print("\nSaved:")
    print(out_dir / "bridge_per_sample_metrics.csv")
    print(out_dir / "bridge_summary_all.csv")
    print(out_dir / "ranked_by_last3_success_32px.csv")
    print(out_dir / "ranked_by_nss.csv")
    print(out_dir / "bridge_metric_info.json")


if __name__ == "__main__":
    main()