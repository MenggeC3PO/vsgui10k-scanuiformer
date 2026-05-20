#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Metrics where larger is better
HIGHER_IS_BETTER = {
    "professional_score",
    "any_hit_rate_m0.00",
    "any_hit_rate_m0.03",
    "any_hit_rate_m0.05",
    "any_hit_rate_m0.10",
    "final_hit_rate_m0.00",
    "final_hit_rate_m0.03",
    "final_hit_rate_m0.05",
    "final_hit_rate_m0.10",
    "gt_any_hit_rate_m0.00",
    "gt_any_hit_rate_m0.03",
    "gt_any_hit_rate_m0.05",
    "gt_any_hit_rate_m0.10",
    "gt_final_hit_rate_m0.00",
    "gt_final_hit_rate_m0.03",
    "gt_final_hit_rate_m0.05",
    "gt_final_hit_rate_m0.10",
    "stopped_rate",
    "dtw_similarity_mean",
    "scanmatch_lite_mean",
    "multimatch_vector_mean",
    "multimatch_direction_mean",
    "multimatch_length_mean",
    "multimatch_position_mean",
    "multimatch_mean_mean",
}

# Metrics where smaller is better
LOWER_IS_BETTER = {
    "timeout_rate",
    "early_stop_before_target_rate_m0.05",
    "ade_mean",
    "rmse_mean",
    "fde_mean",
    "dtw_distance_mean",
    "sed_norm_mean",
    "length_abs_error_mean",
    "first_hit_step_error_mean_m0.00",
    "first_hit_step_error_mean_m0.03",
    "first_hit_step_error_mean_m0.05",
    "first_hit_step_error_mean_m0.10",
}


def safe_float(x):
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return None
    except Exception:
        return None


def read_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_metric_files(ckpt_dir: Path, split: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    compact = ckpt_dir / f"{split}_compact_metrics.csv"
    full = ckpt_dir / f"{split}_full_metrics.csv"
    summary = ckpt_dir / f"{split}_summary.json"
    return (
        compact if compact.exists() else None,
        full if full.exists() else None,
        summary if summary.exists() else None,
    )


def load_checkpoint_rows(ckpt_dir: Path, split: str) -> pd.DataFrame:
    compact_path, full_path, summary_path = find_metric_files(ckpt_dir, split)

    if compact_path is None and full_path is None:
        return pd.DataFrame()

    # Prefer compact metrics for thesis checkpoint ranking.
    path = compact_path if compact_path is not None else full_path
    df = pd.read_csv(path)

    df.insert(0, "checkpoint", ckpt_dir.name)
    df.insert(1, "source_csv", str(path))

    summary = read_json_if_exists(summary_path) if summary_path else {}
    if summary:
        df["checkpoint_epoch"] = summary.get("checkpoint_epoch", None)
        df["checkpoint_path"] = summary.get("checkpoint", None)
        df["split"] = summary.get("split", split)
    else:
        df["checkpoint_epoch"] = None
        df["checkpoint_path"] = None
        df["split"] = split

    return df


def collect_all(eval_root: Path, split: str, checkpoints: Optional[List[str]]) -> pd.DataFrame:
    if not eval_root.exists():
        raise FileNotFoundError(f"eval_root does not exist: {eval_root}")

    ckpt_dirs = sorted([p for p in eval_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    if checkpoints:
        wanted = set(checkpoints)
        ckpt_dirs = [p for p in ckpt_dirs if p.name in wanted]

    all_rows = []
    for ckpt_dir in ckpt_dirs:
        df = load_checkpoint_rows(ckpt_dir, split)
        if df.empty:
            print(f"SKIP no metrics: {ckpt_dir}")
            continue
        print(f"loaded {len(df):3d} rows from {ckpt_dir.name}")
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"No metric CSV files found under {eval_root}")

    out = pd.concat(all_rows, ignore_index=True)
    return out


def metric_direction(metric: str) -> Optional[str]:
    if metric in HIGHER_IS_BETTER:
        return "max"
    if metric in LOWER_IS_BETTER:
        return "min"

    # Fallback heuristics
    low_words = ["loss", "error", "ade", "rmse", "fde", "timeout", "early", "distance", "sed_norm", "abs_error"]
    high_words = ["hit", "score", "similarity", "scanmatch", "multimatch", "stopped_rate"]

    m = metric.lower()
    if any(w in m for w in low_words):
        return "min"
    if any(w in m for w in high_words):
        return "max"
    return None


def best_rows_for_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []

    for metric in metrics:
        if metric not in df.columns:
            rows.append({
                "metric": metric,
                "direction": "missing",
                "best_value": None,
                "checkpoint": None,
                "mode": None,
                "threshold": None,
                "note": "metric not found",
            })
            continue

        direction = metric_direction(metric)
        if direction is None:
            rows.append({
                "metric": metric,
                "direction": "unknown",
                "best_value": None,
                "checkpoint": None,
                "mode": None,
                "threshold": None,
                "note": "unknown direction; skipped",
            })
            continue

        temp = df.copy()
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
        temp = temp.dropna(subset=[metric])
        if temp.empty:
            rows.append({
                "metric": metric,
                "direction": direction,
                "best_value": None,
                "checkpoint": None,
                "mode": None,
                "threshold": None,
                "note": "no finite values",
            })
            continue

        idx = temp[metric].idxmax() if direction == "max" else temp[metric].idxmin()
        r = temp.loc[idx].to_dict()

        rows.append({
            "metric": metric,
            "direction": direction,
            "best_value": r.get(metric),
            "checkpoint": r.get("checkpoint"),
            "mode": r.get("mode"),
            "threshold": r.get("threshold"),
            "professional_score": r.get("professional_score"),
            "any_hit_rate_m0.05": r.get("any_hit_rate_m0.05"),
            "final_hit_rate_m0.05": r.get("final_hit_rate_m0.05"),
            "any_hit_rate_m0.10": r.get("any_hit_rate_m0.10"),
            "final_hit_rate_m0.10": r.get("final_hit_rate_m0.10"),
            "timeout_rate": r.get("timeout_rate"),
            "pred_len_mean": r.get("pred_len_mean"),
            "gt_len_mean": r.get("gt_len_mean"),
            "ade_mean": r.get("ade_mean"),
            "rmse_mean": r.get("rmse_mean"),
            "fde_mean": r.get("fde_mean"),
            "scanmatch_lite_mean": r.get("scanmatch_lite_mean"),
            "dtw_similarity_mean": r.get("dtw_similarity_mean"),
            "multimatch_mean_mean": r.get("multimatch_mean_mean"),
            "checkpoint_epoch": r.get("checkpoint_epoch"),
            "source_csv": r.get("source_csv"),
            "note": "",
        })

    return pd.DataFrame(rows)


def add_selection_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert useful metrics to numeric
    for c in df.columns:
        if c not in ["checkpoint", "mode", "source_csv", "checkpoint_path", "split"]:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    # Main thesis-oriented score:
    # favors target reaching, final target stopping, scanpath similarity;
    # penalizes timeout and early stopping.
    components = []
    for c in [
        "any_hit_rate_m0.05",
        "final_hit_rate_m0.05",
        "any_hit_rate_m0.10",
        "scanmatch_lite_mean",
        "dtw_similarity_mean",
        "multimatch_mean_mean",
    ]:
        if c in df.columns:
            components.append(pd.to_numeric(df[c], errors="coerce"))

    if components:
        base = sum(components) / len(components)
    else:
        base = pd.Series([float("nan")] * len(df), index=df.index)

    penalty = 0.0
    if "timeout_rate" in df.columns:
        penalty = penalty + 0.05 * pd.to_numeric(df["timeout_rate"], errors="coerce").fillna(0)
    if "early_stop_before_target_rate_m0.05" in df.columns:
        penalty = penalty + 0.10 * pd.to_numeric(df["early_stop_before_target_rate_m0.05"], errors="coerce").fillna(0)

    df["thesis_selection_score"] = base - penalty

    return df


def sort_for_thesis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sort_cols = []
    ascending = []

    for c, asc in [
        ("thesis_selection_score", False),
        ("professional_score", False),
        ("any_hit_rate_m0.05", False),
        ("final_hit_rate_m0.05", False),
        ("timeout_rate", True),
        ("ade_mean", True),
    ]:
        if c in df.columns:
            sort_cols.append(c)
            ascending.append(asc)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last")

    return df


def compact_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "checkpoint",
        "checkpoint_epoch",
        "mode",
        "threshold",
        "thesis_selection_score",
        "professional_score",
        "num_samples",
        "any_hit_rate_m0.05",
        "final_hit_rate_m0.05",
        "any_hit_rate_m0.10",
        "final_hit_rate_m0.10",
        "timeout_rate",
        "stopped_rate",
        "early_stop_before_target_rate_m0.05",
        "pred_len_mean",
        "gt_len_mean",
        "length_abs_error_mean",
        "ade_mean",
        "rmse_mean",
        "fde_mean",
        "scanmatch_lite_mean",
        "sed_norm_mean",
        "dtw_similarity_mean",
        "multimatch_mean_mean",
        "multimatch_vector_mean",
        "multimatch_direction_mean",
        "multimatch_length_mean",
        "multimatch_position_mean",
        "stop_reason_rate_learned_stop",
        "stop_reason_rate_heuristic_inside_streak",
        "stop_reason_rate_heuristic_target_entry_count",
        "stop_reason_rate_max_steps",
        "source_csv",
    ]

    cols = [c for c in wanted if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]


def print_top(df: pd.DataFrame, title: str, n: int = 10):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    show_cols = [
        "checkpoint",
        "mode",
        "threshold",
        "thesis_selection_score",
        "professional_score",
        "any_hit_rate_m0.05",
        "final_hit_rate_m0.05",
        "any_hit_rate_m0.10",
        "final_hit_rate_m0.10",
        "timeout_rate",
        "pred_len_mean",
        "gt_len_mean",
        "ade_mean",
        "scanmatch_lite_mean",
        "dtw_similarity_mean",
        "multimatch_mean_mean",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    with pd.option_context("display.max_columns", 30, "display.width", 220):
        print(df[show_cols].head(n).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", type=str, required=True,
                        help="Folder containing checkpoint subfolders: best/, best_dist/, epoch01/, ...")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="",
                        help="Where to save organized CSVs. Default: eval_root/organized_results")
    parser.add_argument("--checkpoints", type=str, default="",
                        help="Optional comma-separated checkpoint folder names to include.")
    parser.add_argument("--metrics", type=str, default="",
                        help="Optional comma-separated metrics for best-by-metric table.")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir) if args.out_dir else eval_root / "organized_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [x.strip() for x in args.checkpoints.split(",") if x.strip()] if args.checkpoints.strip() else None

    df = collect_all(eval_root=eval_root, split=args.split, checkpoints=checkpoints)
    df = add_selection_scores(df)

    all_path = out_dir / "all_checkpoint_metrics_combined.csv"
    df.to_csv(all_path, index=False)

    compact = compact_columns(df)
    compact_path = out_dir / "all_checkpoint_metrics_compact.csv"
    compact.to_csv(compact_path, index=False)

    ranked_all = sort_for_thesis(compact)
    ranked_all_path = out_dir / "ranked_all_modes_all_thresholds.csv"
    ranked_all.to_csv(ranked_all_path, index=False)

    # Separate pure/hybrid ranking
    if "mode" in ranked_all.columns:
        for mode in sorted(ranked_all["mode"].dropna().unique()):
            sub = ranked_all[ranked_all["mode"] == mode]
            p = out_dir / f"ranked_{mode}_only.csv"
            sub.to_csv(p, index=False)

    # Best row per metric
    default_metrics = [
        "professional_score",
        "thesis_selection_score",
        "any_hit_rate_m0.05",
        "final_hit_rate_m0.05",
        "any_hit_rate_m0.10",
        "final_hit_rate_m0.10",
        "timeout_rate",
        "early_stop_before_target_rate_m0.05",
        "ade_mean",
        "rmse_mean",
        "fde_mean",
        "scanmatch_lite_mean",
        "dtw_similarity_mean",
        "multimatch_mean_mean",
        "pred_len_mean",
        "length_abs_error_mean",
    ]

    # Add custom score direction
    HIGHER_IS_BETTER.add("thesis_selection_score")

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()] if args.metrics.strip() else default_metrics
    best_by_metric = best_rows_for_metrics(df, metrics)
    best_by_metric_path = out_dir / "best_row_by_metric.csv"
    best_by_metric.to_csv(best_by_metric_path, index=False)

    # Best row per checkpoint/mode by thesis_selection_score
    if {"checkpoint", "mode", "thesis_selection_score"}.issubset(df.columns):
        temp = df.copy()
        temp["thesis_selection_score"] = pd.to_numeric(temp["thesis_selection_score"], errors="coerce")
        idx = temp.groupby(["checkpoint", "mode"])["thesis_selection_score"].idxmax()
        best_per_ckpt_mode = compact_columns(temp.loc[idx].sort_values(
            "thesis_selection_score", ascending=False, na_position="last"
        ))
        best_per_ckpt_mode_path = out_dir / "best_threshold_per_checkpoint_and_mode.csv"
        best_per_ckpt_mode.to_csv(best_per_ckpt_mode_path, index=False)
    else:
        best_per_ckpt_mode_path = None

    # Best row per checkpoint regardless of mode
    if {"checkpoint", "thesis_selection_score"}.issubset(df.columns):
        temp = df.copy()
        temp["thesis_selection_score"] = pd.to_numeric(temp["thesis_selection_score"], errors="coerce")
        idx = temp.groupby(["checkpoint"])["thesis_selection_score"].idxmax()
        best_per_ckpt = compact_columns(temp.loc[idx].sort_values(
            "thesis_selection_score", ascending=False, na_position="last"
        ))
        best_per_ckpt_path = out_dir / "best_threshold_mode_per_checkpoint.csv"
        best_per_ckpt.to_csv(best_per_ckpt_path, index=False)
    else:
        best_per_ckpt_path = None

    print_top(ranked_all, "TOP ROWS BY THESIS SELECTION SCORE", n=args.top_n)

    print("\n" + "=" * 120)
    print("BEST ROW BY INDIVIDUAL METRIC")
    print("=" * 120)
    with pd.option_context("display.max_columns", 30, "display.width", 220):
        print(best_by_metric.to_string(index=False))

    print("\nSaved files:")
    print(all_path)
    print(compact_path)
    print(ranked_all_path)
    print(best_by_metric_path)
    if best_per_ckpt_mode_path:
        print(best_per_ckpt_mode_path)
    if best_per_ckpt_path:
        print(best_per_ckpt_path)
    print(out_dir / "ranked_pure_only.csv")
    print(out_dir / "ranked_hybrid_only.csv")


if __name__ == "__main__":
    main()