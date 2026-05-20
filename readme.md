# ScanUIFormer: GUI Visual Search Scanpath Prediction

ScanUIFormer is an autoregressive model for predicting target-directed human scanpaths on graphical user interfaces. This repository contains the VSGUI10K preprocessing scripts, model code, training configs, evaluation scripts, and visualization utilities used in our thesis project.

<p align="center">
  <img src="assets/design-overview.png" width="850"/>
</p>

## Highlights

- Official-style preprocessing pipeline for VSGUI10K visual search trials
- Structured GUI scanpath prediction with UI element features, target cues, fixation history, and STOP prediction
- Evaluation on both an internal all-type split and a SeekUI-aligned 232-sample subset
- Rollout-based metrics and qualitative scanpath visualizations

## Repository Structure

```text
configs/          Training configs, evaluation configs, and split files
preprocessing/    Scripts for generating trial-level VSGUI10K data
scripts/          Example shell and SLURM scripts
src/scanuiformer/ Dataset, model, training, evaluation, and analysis code
assets/           README figures and qualitative examples
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the VSGUI10K dataset. The raw dataset is not included in this repository.

Download VSGUI10K from:

```text
https://osf.io/hmg9b/
```

Place the downloaded files under `data/`:

```text
data/
  vsgui10k_fixations.csv
  vsgui10k_targets.csv
  vsgui10k-images/
  segmentation/
```

The model does not read the raw CSV files directly. They must first be converted into a trial-level file using the preprocessing script.

## Preprocessing

```bash
python preprocessing/preprocess_official_with_validation.py
```

This script performs search-phase extraction, fixation filtering, coordinate normalization, trial segmentation, target bounding-box alignment, and validation checks.

Main generated file:

```text
data/trials_official_with_validation.jsonl.gz
```

If the script writes `trials_official_with_validation.jsonl.gz` to the repository root, move it into `data/` or update the paths in the config files.

## Training

Train from scratch:

```bash
python src/scanuiformer/train.py --config configs/train_from_scratch.json --freeze_patch
```

Continue from checkpoint:

```bash
python src/scanuiformer/train.py --config configs/train_continue_from_checkpoint.json --freeze_patch
```

By default, continuation training expects:

```text
outputs/scanuiformer_from_scratch/last.pt
```

Model checkpoints are not included in this repository.

## Evaluation

Two evaluation configs are provided:

| Config | Purpose |
|---|---|
| `configs/eval_internal_alltype.json` | Internal all-type test split |
| `configs/eval_seekui232.json` | SeekUI-aligned 232-sample subset |

Example evaluation:

```bash
python src/scanuiformer/evaluate.py \
  --config configs/eval_internal_alltype.json \
  --checkpoint outputs/scanuiformer_continue_from_checkpoint/last.pt \
  --split test \
  --output_dir outputs/scanuiformer_continue_from_checkpoint/eval_internal_alltype/last \
  --device auto \
  --modes pure,hybrid \
  --thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --margins 0.00,0.03,0.05,0.10 \
  --save_per_sample \
  --visualize
```

Evaluation outputs include compact metrics, full metrics, per-sample metrics, rollout records, and scanpath visualizations.


## Additional Scripts

```bash
bash scripts/collect_results.sh
bash scripts/add_nss_metrics.sh
bash scripts/add_seekui_bridge_metrics.sh
bash scripts/analyze_groups.sh
```

Example SLURM scripts are also provided in `scripts/`. Edit the account, partition, and environment activation lines before using them on a cluster.

## Notes

The raw dataset, processed trial file, model checkpoints, logs, and experiment outputs are not included in this repository. The provided split files under `configs/splits/` keep internal and SeekUI232 evaluation consistent.

## Citation

This repository uses the VSGUI10K dataset and includes evaluation settings aligned with SeekUI. Please cite the relevant works if you use this code, preprocessing pipeline, or evaluation setup:

```bibtex
@article{putkonen2025vsgui10k,
  title   = {Understanding Visual Search in Graphical User Interfaces},
  author  = {Putkonen, Aini and Jiang, Yue and Zeng, Jingchun and Tammilehto, Olli and Jokinen, Jussi P. P. and Oulasvirta, Antti},
  journal = {International Journal of Human-Computer Studies},
  volume  = {199},
  pages   = {103483},
  year    = {2025},
  doi     = {10.1016/j.ijhcs.2025.103483}
}

@inproceedings{guo2026seekui,
  title     = {SeekUI: Predicting Visual Search Behavior on Graphical User Interfaces with a Reward-Augmented Vision Language Model},
  author    = {Guo, Zixin and Jiang, Yue and Leiva, Luis A. and Oulasvirta, Antti},
  booktitle = {Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  series    = {CHI '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3772318.3791178}
}
```
