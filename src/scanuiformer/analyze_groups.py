#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.append(os.path.dirname(__file__))

from dataset import (
    filter_target_present_trials,
    build_hybrid_trial_samples,
)


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Metadata extraction
# -----------------------------

def normalize_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    return None


def find_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def extract_from_nested(obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
    direct = normalize_str(find_first_key(obj, keys))
    if direct:
        return direct

    nested_names = [
        "target",
        "target_element",
        "target_ui",
        "target_obj",
        "target_info",
        "bbox_target",
        "ui_target",
        "metadata",
    ]

    for name in nested_names:
        nested = obj.get(name)
        if isinstance(nested, dict):
            val = normalize_str(find_first_key(nested, keys))
            if val:
                return val

    return None


def extract_target_type(sample: Dict[str, Any], trial: Optional[Dict[str, Any]]) -> str:
    """
    Tries several common fields used in processed GUI/segmentation records.
    If the target type cannot be recovered, returns UNKNOWN.
    """

    target_type_keys = [
        "target_type",
        "target_ui_type",
        "target_element_type",
        "target_component_type",
        "target_category",
        "target_class",
        "target_role",
        "target_tag",
        "ui_type",
        "element_type",
        "component_type",
        "type",
        "role",
        "class",
        "category",
        "label",
        "component_label",
    ]

    val = extract_from_nested(sample, target_type_keys)
    if val:
        return val

    if trial is not None:
        val = extract_from_nested(trial, target_type_keys)
        if val:
            return val

    # Try resolving through target id if available.
    target_id_keys = [
        "target_id",
        "target_ui_id",
        "target_element_id",
        "target_component_id",
        "target_seg_id",
        "target_idx",
        "target_index",
    ]

    target_id = find_first_key(sample, target_id_keys)
    if target_id is None and trial is not None:
        target_id = find_first_key(trial, target_id_keys)

    if target_id is not None and trial is not None:
        candidate_lists = [
            "ui_elements",
            "elements",
            "components",
            "objects",
            "segments",
            "segmentation",
            "annotations",
            "children",
            "compos",
        ]
        id_keys = ["id", "ui_id", "element_id", "component_id", "seg_id", "idx", "index"]

        for list_name in candidate_lists:
            items = trial.get(list_name)
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_id = find_first_key(item, id_keys)
                    if str(item_id) == str(target_id):
                        val = extract_from_nested(item, target_type_keys)
                        if val:
                            return val

    return "UNKNOWN"


def length_bin(gt_len: Any) -> str:
    try:
        n = int(gt_len)
    except Exception:
        return "UNKNOWN"

    if n <= 4:
        return "01_04"
    if n <= 8:
        return "05_08"
    if n <= 12:
        return "09_12"
    if n <= 16:
        return "13_16"
    return "17_plus"


def load_metadata(config_path: str, split: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

    trial_by_id = {t.get("trial_id"): t for t in eval_trials}

    rows = []
    unknown = 0

    for i, s in enumerate(samples):
        trial_id = s.get("trial_id")
        trial = trial_by_id.get(trial_id)
        ttype = extract_target_type(s, trial)
        if ttype == "UNKNOWN":
            unknown += 1

        rows.append({
            "global_index": i,
            "trial_id": trial_id,
            "img_name": s.get("img_name"),
            "cue": s.get("cue"),
            "target_type": ttype,
        })

    info = {
        "config_path": config_path,
        "split_json_path": split_json_path,
        "split": split,
        "wanted_trial_ids": len(wanted),
        "matched_eval_trials": len(eval_trials),
        "eval_samples": len(samples),
        "unknown_target_type": unknown,
    }

    print("=" * 100)
    print("METADATA CHECK")
    print("config:", config_path)
    print("split_json_path:", split_json_path)
    print("wanted trial ids:", len(wanted))
    print("matched eval trials:", len(eval_trials))
    print("eval samples:", len(samples))
    print("UNKNOWN target_type:", unknown)
    print("=" * 100)

    return pd.DataFrame(rows), info


# -----------------------------
# Aggregation
# -----------------------------

def to_numeric_where_possible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in ["checkpoint", "mode", "trial_id", "img_name", "cue", "target_type", "length_bin", "stop_reason"]:
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def aggregate_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    exclude = {
        "global_index",
        "trial_id",
        "img_name",
        "cue",
        "target_type",
        "length_bin",
        "checkpoint",
        "mode",
        "threshold",
        "stop_reason",
        "source_rollout",
    }

    metric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            metric_cols.append(c)

    rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}
        for col, val in zip(group_cols, keys):
            row[col] = val

        row["num_samples"] = int(len(g))

        for c in metric_cols:
            row[c + "_mean"] = float(pd.to_numeric(g[c], errors="coerce").mean())

        rows.append(row)

    return pd.DataFrame(rows)


def compact_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "target_type",
        "length_bin",
        "num_samples",

        "pred_any_hit_m0.05_mean",
        "pred_final_hit_m0.05_mean",
        "pred_any_hit_m0.10_mean",
        "pred_final_hit_m0.10_mean",

        "gt_any_hit_m0.05_mean",
        "gt_final_hit_m0.05_mean",

        "nss_mean",
        "ade_mean",
        "rmse_mean",
        "fde_mean",
        "scanmatch_lite_mean",
        "dtw_similarity_mean",
        "multimatch_mean_mean",

        "pred_len_mean",
        "gt_len_mean",
        "length_abs_error_mean",
        "timeout_mean",
        "stopped_mean",
        "early_stop_before_target_m0.05_mean",
    ]

    cols = [c for c in wanted if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--per_sample_jsonl", required=True)
    parser.add_argument("--nss_per_sample_csv", required=True)
    parser.add_argument("--checkpoint", default="epoch23")
    parser.add_argument("--mode", default="pure")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta_df, meta_info = load_metadata(args.config, args.split)

    metric_rows = read_jsonl(args.per_sample_jsonl)
    metrics_df = pd.DataFrame(metric_rows)

    if "global_index" not in metrics_df.columns:
        metrics_df["global_index"] = list(range(len(metrics_df)))

    # Load NSS and filter selected checkpoint/mode/threshold.
    nss_df = pd.read_csv(args.nss_per_sample_csv)
    nss_df["threshold"] = pd.to_numeric(nss_df["threshold"], errors="coerce").round(4)

    selected_nss = nss_df[
        (nss_df["checkpoint"].astype(str) == str(args.checkpoint))
        & (nss_df["mode"].astype(str) == str(args.mode))
        & (nss_df["threshold"] == round(float(args.threshold), 4))
    ].copy()

    print("=" * 100)
    print("SELECTED SETTING")
    print("checkpoint:", args.checkpoint)
    print("mode:", args.mode)
    print("threshold:", args.threshold)
    print("per-sample metric rows:", len(metrics_df))
    print("selected NSS rows:", len(selected_nss))
    print("=" * 100)

    if len(selected_nss) == 0:
        raise RuntimeError("No matching NSS rows found. Check checkpoint/mode/threshold.")

    keep_nss = ["global_index", "nss"]
    selected_nss = selected_nss[keep_nss]

    merged = metrics_df.merge(meta_df, on="global_index", how="left")
    merged = merged.merge(selected_nss, on="global_index", how="left")

    merged["checkpoint"] = args.checkpoint
    merged["mode"] = args.mode
    merged["threshold"] = args.threshold
    merged["length_bin"] = merged["gt_len"].apply(length_bin)

    merged = to_numeric_where_possible(merged)

    # Save full selected per-sample table.
    full_path = os.path.join(args.out_dir, "selected_per_sample_with_target_type_length_nss.csv")
    merged.to_csv(full_path, index=False)

    # Group tables.
    by_type = aggregate_group(merged, ["target_type"])
    by_len = aggregate_group(merged, ["length_bin"])
    by_type_len = aggregate_group(merged, ["target_type", "length_bin"])

    by_type = compact_columns(by_type)
    by_len = compact_columns(by_len)
    by_type_len = compact_columns(by_type_len)

    # Sort.
    if "num_samples" in by_type.columns:
        by_type = by_type.sort_values("num_samples", ascending=False)
    if "length_bin" in by_len.columns:
        order = {"01_04": 0, "05_08": 1, "09_12": 2, "13_16": 3, "17_plus": 4, "UNKNOWN": 99}
        by_len["_order"] = by_len["length_bin"].map(order).fillna(99)
        by_len = by_len.sort_values("_order").drop(columns=["_order"])

    by_type_path = os.path.join(args.out_dir, "by_target_type_metrics.csv")
    by_len_path = os.path.join(args.out_dir, "by_scanpath_length_metrics.csv")
    by_type_len_path = os.path.join(args.out_dir, "by_target_type_and_length_metrics.csv")

    by_type.to_csv(by_type_path, index=False)
    by_len.to_csv(by_len_path, index=False)
    by_type_len.to_csv(by_type_len_path, index=False)

    save_json({
        "selected_setting": {
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "threshold": args.threshold,
            "split": args.split,
            "per_sample_jsonl": args.per_sample_jsonl,
            "nss_per_sample_csv": args.nss_per_sample_csv,
        },
        "metadata_info": meta_info,
        "outputs": {
            "selected_per_sample": full_path,
            "by_target_type": by_type_path,
            "by_scanpath_length": by_len_path,
            "by_target_type_and_length": by_type_len_path,
        },
        "notes": {
            "target_type": "Extracted from processed sample/raw trial metadata when available. UNKNOWN means no target-type field was recovered.",
            "length_bin": "Based on gt_len: 01_04, 05_08, 09_12, 13_16, 17_plus.",
            "rates": "Boolean columns are averaged, so pred_any_hit_m0.05_mean is AnyHit@0.05 rate.",
            "selected_reason": "Selected as the balanced pure rollout setting for internal all-type analysis."
        }
    }, os.path.join(args.out_dir, "group_analysis_summary.json"))

    print("\nSaved:")
    print(full_path)
    print(by_type_path)
    print(by_len_path)
    print(by_type_len_path)
    print(os.path.join(args.out_dir, "group_analysis_summary.json"))

    print("\nTop target-type rows:")
    show_cols = [
        "target_type", "num_samples",
        "pred_any_hit_m0.05_mean", "pred_final_hit_m0.05_mean",
        "nss_mean", "ade_mean", "scanmatch_lite_mean", "multimatch_mean_mean",
        "timeout_mean", "pred_len_mean", "gt_len_mean",
    ]
    show_cols = [c for c in show_cols if c in by_type.columns]
    print(by_type[show_cols].head(20).to_string(index=False))

    print("\nScanpath-length rows:")
    show_cols = [
        "length_bin", "num_samples",
        "pred_any_hit_m0.05_mean", "pred_final_hit_m0.05_mean",
        "nss_mean", "ade_mean", "scanmatch_lite_mean", "multimatch_mean_mean",
        "timeout_mean", "pred_len_mean", "gt_len_mean",
    ]
    show_cols = [c for c in show_cols if c in by_len.columns]
    print(by_len[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()