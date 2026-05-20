
import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from statistics import mean, median


def read_jsonl_gz(path):
    trials = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def safe_len(x):
    try:
        return len(x)
    except Exception:
        return None


def summarize_structure(trials):
    print("=" * 80)
    print("TRIAL FILE STRUCTURE")
    print("=" * 80)
    print(f"Total trials: {len(trials)}")

    if not trials:
        return

    sample = trials[0]
    print("\nTop-level keys:")
    for k, v in sample.items():
        print(f"  - {k}: {type(v).__name__}")

    print("\nNested structure from first trial:")
    for section in ["key", "geom", "target", "phases", "success", "meta"]:
        if section in sample and isinstance(sample[section], dict):
            print(f"\n[{section}]")
            for k, v in sample[section].items():
                if isinstance(v, dict):
                    print(f"  - {k}: dict -> keys={list(v.keys())}")
                elif isinstance(v, list):
                    extra = ""
                    if v and isinstance(v[0], dict):
                        extra = f" first_item_keys={list(v[0].keys())}"
                    print(f"  - {k}: list len={len(v)}{extra}")
                else:
                    print(f"  - {k}: {type(v).__name__} -> {v}")


def extract_basic_views(trials):
    rows = []
    for t in trials:
        key = t.get("key", {})
        phases = t.get("phases", {})
        search = phases.get("search", {})
        validation = phases.get("validation", {})

        scanpath = search.get("scanpath", [])
        rows.append({
            "trial_id": t.get("trial_id"),
            "pid": key.get("pid"),
            "img_name": key.get("img_name"),
            "tgt_id": key.get("tgt_id"),
            "cue": key.get("cue"),
            "absent": key.get("absent"),
            "search_n_fix": search.get("n_fix"),
            "search_t_end": search.get("t_end"),
            "scanpath_len": len(scanpath) if isinstance(scanpath, list) else None,
            "validation_available": validation.get("available"),
            "validation_n_fix": validation.get("n_fix"),
        })
    return rows


def print_basic_statistics(rows):
    print("\n" + "=" * 80)
    print("BASIC DATA ORGANIZATION")
    print("=" * 80)

    imgs = [r["img_name"] for r in rows]
    pids = [r["pid"] for r in rows]
    tgts = [r["tgt_id"] for r in rows]
    cues = [r["cue"] for r in rows]

    print(f"Unique participants: {len(set(pids))}")
    print(f"Unique images: {len(set(imgs))}")
    print(f"Unique targets: {len(set(tgts))}")
    print(f"Unique cues: {len(set(cues))}")

    img_counter = Counter(imgs)
    img_counts = list(img_counter.values())
    print("\nTrials per image:")
    print(f"  mean   = {mean(img_counts):.2f}")
    print(f"  median = {median(img_counts):.2f}")
    print(f"  min    = {min(img_counts)}")
    print(f"  max    = {max(img_counts)}")

    print("\nMost repeated images:")
    for img, c in img_counter.most_common(10):
        print(f"  {img}: {c} trials")

    pair_img_cue = Counter((r["img_name"], r["cue"]) for r in rows)
    pair_img_tgt = Counter((r["img_name"], r["tgt_id"]) for r in rows)

    repeated_img = sum(1 for _, c in img_counter.items() if c > 1)
    repeated_img_cue = sum(1 for _, c in pair_img_cue.items() if c > 1)
    repeated_img_tgt = sum(1 for _, c in pair_img_tgt.items() if c > 1)

    print("\nRepetition pattern:")
    print(f"  images appearing in >1 trial: {repeated_img}")
    print(f"  (image, cue) appearing in >1 trial: {repeated_img_cue}")
    print(f"  (image, target) appearing in >1 trial: {repeated_img_tgt}")


def print_group_views(rows):
    print("\n" + "=" * 80)
    print("HOW ONE IMAGE EXPANDS INTO MULTIPLE TRIALS")
    print("=" * 80)

    img_to_rows = defaultdict(list)
    for r in rows:
        img_to_rows[r["img_name"]].append(r)

    # choose a few representative images with many trials
    ranked = sorted(img_to_rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
    for img, group in ranked:
        pids = sorted(set(r["pid"] for r in group))
        cues = sorted(set(r["cue"] for r in group))
        tgts = sorted(set(r["tgt_id"] for r in group))
        splens = [r["scanpath_len"] for r in group if r["scanpath_len"] is not None]

        print(f"\nImage: {img}")
        print(f"  number_of_trials      = {len(group)}")
        print(f"  unique_participants   = {len(pids)}")
        print(f"  unique_cues           = {len(cues)} -> {cues[:10]}")
        print(f"  unique_targets        = {len(tgts)}")
        if splens:
            print(f"  scanpath_len_mean     = {mean(splens):.2f}")
            print(f"  scanpath_len_min_max  = {min(splens)} / {max(splens)}")

        print("  first few trials:")
        for r in group[:6]:
            print(
                f"    trial_id={r['trial_id']} "
                f"pid={r['pid']} cue={r['cue']} tgt={r['tgt_id']} "
                f"scanpath_len={r['scanpath_len']} absent={r['absent']}"
            )

def print_same_task_multiple_people(rows, topn=20):
    print("\n" + "=" * 80)
    print("SAME IMAGE + SAME CUE + SAME TARGET ACROSS MULTIPLE PARTICIPANTS")
    print("=" * 80)

    task_map = defaultdict(list)
    for r in rows:
        task_map[(r["img_name"], r["cue"], r["tgt_id"], r["absent"])].append(r)

    multi = [(k, v) for k, v in task_map.items() if len(v) > 1]
    multi.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    print(f"Number of repeated exact tasks: {len(multi)}")
    print("\nTop repeated exact tasks:")
    for (img, cue, tgt, absent), group in multi[:topn]:
        pids = sorted(set(r["pid"] for r in group))
        splens = [r["scanpath_len"] for r in group if r["scanpath_len"] is not None]
        splen_mean = f"{mean(splens):.2f}" if splens else "NA"
        print(
            f"  img={img} cue={cue} tgt={tgt} absent={absent} "
            f"trials={len(group)} unique_pids={len(pids)} "
            f"scanpath_len_mean={splen_mean}"
        )


def show_trial_examples(trials, n=2):
    print("\n" + "=" * 80)
    print("TRIAL EXAMPLE SNIPPETS")
    print("=" * 80)

    for i, t in enumerate(trials[:n]):
        print(f"\n--- Trial example {i+1} ---")
        small = {
            "trial_id": t.get("trial_id"),
            "key": t.get("key"),
            "geom": t.get("geom"),
            "target": t.get("target"),
            "search_summary": {
                "n_fix": t.get("phases", {}).get("search", {}).get("n_fix"),
                "t_end": t.get("phases", {}).get("search", {}).get("t_end"),
                "first_3_scanpath_points": t.get("phases", {}).get("search", {}).get("scanpath", [])[:3],
            },
            "validation_summary": {
                "available": t.get("phases", {}).get("validation", {}).get("available"),
                "n_fix": t.get("phases", {}).get("validation", {}).get("n_fix"),
            },
            "success": t.get("success"),
        }
        print(json.dumps(small, indent=2, ensure_ascii=False))


def write_csvs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    flat_path = os.path.join(out_dir, "trial_flat_summary.csv")
    with open(flat_path, "w", encoding="utf-8") as f:
        headers = [
            "trial_id", "pid", "img_name", "tgt_id", "cue", "absent",
            "search_n_fix", "search_t_end", "scanpath_len",
            "validation_available", "validation_n_fix"
        ]
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = [r.get(h, "") for h in headers]
            vals = [("" if v is None else str(v)).replace(",", ";") for v in vals]
            f.write(",".join(vals) + "\n")

    # image-level view
    img_to_rows = defaultdict(list)
    for r in rows:
        img_to_rows[r["img_name"]].append(r)

    img_path = os.path.join(out_dir, "image_group_summary.csv")
    with open(img_path, "w", encoding="utf-8") as f:
        headers = [
            "img_name", "n_trials", "n_participants", "n_cues",
            "n_targets", "mean_scanpath_len", "min_scanpath_len", "max_scanpath_len"
        ]
        f.write(",".join(headers) + "\n")
        for img, group in sorted(img_to_rows.items()):
            splens = [r["scanpath_len"] for r in group if r["scanpath_len"] is not None]
            line = [
                img,
                len(group),
                len(set(r["pid"] for r in group)),
                len(set(r["cue"] for r in group)),
                len(set(r["tgt_id"] for r in group)),
                f"{mean(splens):.4f}" if splens else "",
                min(splens) if splens else "",
                max(splens) if splens else "",
            ]
            f.write(",".join(map(str, line)) + "\n")

    task_map = defaultdict(list)
    for r in rows:
        task_map[(r["img_name"], r["cue"], r["tgt_id"], r["absent"])].append(r)

    task_path = os.path.join(out_dir, "exact_task_repetitions.csv")
    with open(task_path, "w", encoding="utf-8") as f:
        headers = [
            "img_name", "cue", "tgt_id", "absent",
            "n_trials", "n_participants", "participant_ids"
        ]
        f.write(",".join(headers) + "\n")
        for (img, cue, tgt, absent), group in sorted(task_map.items()):
            if len(group) <= 1:
                continue
            pids = sorted(set(r["pid"] for r in group))
            line = [
                img, cue, tgt, absent,
                len(group), len(pids), "|".join(pids)
            ]
            f.write(",".join(map(str, line)) + "\n")

    print("\nCSV files written to:")
    print(f"  - {flat_path}")
    print(f"  - {img_path}")
    print(f"  - {task_path}")


def main():
    parser = argparse.ArgumentParser(description="Scan trial file structure and repetition patterns.")
    parser.add_argument("--trials", type=str, required=True, help="Path to trials_official_with_validation.jsonl.gz")
    parser.add_argument("--out_dir", type=str, default="trial_scan_output", help="Directory for CSV outputs")
    parser.add_argument("--example_trials", type=int, default=2, help="How many trial examples to print")
    args = parser.parse_args()

    trials = read_jsonl_gz(args.trials)
    rows = extract_basic_views(trials)

    summarize_structure(trials)
    print_basic_statistics(rows)
    print_group_views(rows)
    print_same_task_multiple_people(rows)
    show_trial_examples(trials, n=args.example_trials)
    write_csvs(rows, args.out_dir)

    print("\n" + "=" * 80)
    print("HOW TO INTERPRET THIS FOR MODEL LEARNING")
    print("=" * 80)
    print("1. One trial = one training sample.")
    print("2. The same image can appear in many trials.")
    print("3. Different trials may share the same image but differ in participant, cue, target, or scanpath.")
    print("4. This means the model may learn multiple human scanpaths on the same GUI image.")
    print("5. Therefore, understanding split strategy (trial-level vs image-level) is critical for evaluation.")


if __name__ == "__main__":
    main()
