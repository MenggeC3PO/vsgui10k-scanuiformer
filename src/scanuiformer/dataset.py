import gzip
import json
import os
import random
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def read_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    trials = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_trials(
    trials: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    rng = random.Random(seed)
    trials_copy = trials[:]
    rng.shuffle(trials_copy)

    n = len(trials_copy)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_trials = trials_copy[:n_train]
    val_trials = trials_copy[n_train:n_train + n_val]
    test_trials = trials_copy[n_train + n_val:]
    return train_trials, val_trials, test_trials


def build_trial_index(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["trial_id"]: t for t in trials}


def build_cue_vocab(trials: List[Dict[str, Any]]) -> Dict[str, int]:
    cues = sorted({str(t["key"]["cue"]) for t in trials})
    vocab = {"<UNK>": 0}
    for i, cue in enumerate(cues, start=1):
        vocab[cue] = i
    return vocab


def get_duration(p: Dict[str, Any]) -> float:
    for k in ["dur", "duration", "dur_s", "dur_sec", "dt"]:
        if k in p and p[k] is not None:
            return float(p[k])
    return 0.0


def get_xy_seg_norm(p: Dict[str, Any], seg_w: float, seg_h: float) -> Tuple[float, float]:
    x = float(p["xy_seg"]["x"]) / float(seg_w)
    y = float(p["xy_seg"]["y"]) / float(seg_h)
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def is_target_present_trial(trial: Dict[str, Any]) -> bool:
    key = trial.get("key", {})
    success = trial.get("success", {})
    if "absent" in key:
        return int(key.get("absent", 0)) == 0
    if "present_trial" in success:
        return int(success.get("present_trial", 1)) == 1
    return True


def filter_target_present_trials(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = [t for t in trials if is_target_present_trial(t)]
    print("target-present filtering")
    print("  input trials:", len(trials))
    print("  kept present trials:", len(kept))
    print("  removed absent trials:", len(trials) - len(kept))
    return kept


def get_target_bbox_norm(trial: Dict[str, Any]) -> List[float]:
    seg_w = float(trial["geom"]["seg_w"])
    seg_h = float(trial["geom"]["seg_h"])
    target = trial.get("target", {})
    bbox = target.get("bbox_seg")

    if not isinstance(bbox, dict):
        raise KeyError(f"Missing target.bbox_seg for trial {trial.get('trial_id')}")

    if all(k in bbox for k in ["x0", "y0", "x1", "y1"]):
        x0 = float(bbox["x0"]) / seg_w
        y0 = float(bbox["y0"]) / seg_h
        x1 = float(bbox["x1"]) / seg_w
        y1 = float(bbox["y1"]) / seg_h
    elif all(k in bbox for k in ["x", "y", "w", "h"]):
        x0 = float(bbox["x"]) / seg_w
        y0 = float(bbox["y"]) / seg_h
        x1 = (float(bbox["x"]) + float(bbox["w"])) / seg_w
        y1 = (float(bbox["y"]) + float(bbox["h"])) / seg_h
    else:
        raise KeyError(f"Unsupported target.bbox_seg format for trial {trial.get('trial_id')}: {bbox}")

    xa, xb = sorted([x0, x1])
    ya, yb = sorted([y0, y1])
    return [
        max(0.0, min(1.0, xa)),
        max(0.0, min(1.0, ya)),
        max(0.0, min(1.0, xb)),
        max(0.0, min(1.0, yb)),
    ]


def normalize_ui_type(raw_type: Any) -> str:
    if raw_type is None:
        return "<UNK>"
    s = str(raw_type).strip()
    return s if s else "<UNK>"


def strip_ext(filename: str) -> str:
    base, _ = os.path.splitext(filename)
    return base


def build_seg_index(seg_root: str) -> Dict[str, str]:
    seg_index = {}
    for root, _, files in os.walk(seg_root):
        for fn in files:
            if fn.lower().endswith(".json"):
                seg_index[os.path.splitext(fn)[0]] = os.path.join(root, fn)
    return seg_index


def find_seg_path(seg_index: Dict[str, str], img_name: str) -> str:
    stem = strip_ext(img_name)
    if stem in seg_index:
        return seg_index[stem]
    raise FileNotFoundError(f"Cannot find segmentation file for image: {img_name}")


def parse_bbox_from_elem(elem: Dict[str, Any]) -> Optional[List[float]]:
    bbox = elem.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
        return [x1, y1, x1 + x2, y1 + y2]

    bbox_xywh = elem.get("bbox_xywh")
    if isinstance(bbox_xywh, list) and len(bbox_xywh) == 4:
        x, y, w, h = [float(v) for v in bbox_xywh]
        return [x, y, x + w, y + h]

    if all(k in elem for k in ["x1", "y1", "x2", "y2"]):
        return [float(elem["x1"]), float(elem["y1"]), float(elem["x2"]), float(elem["y2"])]

    if all(k in elem for k in ["x", "y", "w", "h"]):
        x, y, w, h = float(elem["x"]), float(elem["y"]), float(elem["w"]), float(elem["h"])
        return [x, y, x + w, y + h]

    if all(k in elem for k in ["column_min", "row_min", "column_max", "row_max"]):
        return [float(elem["column_min"]), float(elem["row_min"]), float(elem["column_max"]), float(elem["row_max"])]

    if "position" in elem and "size" in elem:
        pos, size = elem["position"], elem["size"]
        if isinstance(pos, dict) and isinstance(size, dict):
            if all(k in pos for k in ["x", "y"]) and all(k in size for k in ["width", "height"]):
                x, y = float(pos["x"]), float(pos["y"])
                w, h = float(size["width"]), float(size["height"])
                return [x, y, x + w, y + h]
    return None


def load_ui_elements_from_seg(seg_index: Dict[str, str], img_name: str) -> List[Dict[str, Any]]:
    seg_path = find_seg_path(seg_index, img_name)
    data = load_json(seg_path)
    elements = (
        data.get("elements")
        or data.get("ui_elements")
        or data.get("compos")
        or data.get("components")
        or data.get("children")
        or []
    )

    out = []
    for elem in elements:
        if not isinstance(elem, dict):
            continue
        bbox = parse_bbox_from_elem(elem)
        if bbox is None:
            continue
        ui_type = (
            elem.get("class")
            or elem.get("type")
            or elem.get("category")
            or elem.get("label")
            or elem.get("component_label")
            or "<UNK>"
        )
        out.append({"bbox": bbox, "type": normalize_ui_type(ui_type)})
    return out


def build_ui_type_vocab(train_trials: List[Dict[str, Any]], seg_index: Dict[str, str]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    next_id = 2
    for i, t in enumerate(train_trials):
        if i % 500 == 0:
            print(f"  build_ui_type_vocab progress: {i}/{len(train_trials)}")
        img_name = t["key"]["img_name"]
        try:
            elems = load_ui_elements_from_seg(seg_index, img_name)
        except FileNotFoundError:
            elems = []
        for e in elems:
            ui_type = normalize_ui_type(e.get("type", "<UNK>"))
            if ui_type.lower() == "background":
                continue
            if ui_type not in vocab:
                vocab[ui_type] = next_id
                next_id += 1
    return vocab


def should_filter_ui_element(x1, y1, x2, y2, seg_w, seg_h, ui_type, drop_full_screen_root=False) -> bool:
    w = (x2 - x1) / float(seg_w)
    h = (y2 - y1) / float(seg_h)
    if drop_full_screen_root:
        if w > 0.95 and h > 0.95:
            return True
        if normalize_ui_type(ui_type).lower() == "background":
            return True
    return False


def encode_ui_elements(
    elements: List[Dict[str, Any]],
    seg_w: float,
    seg_h: float,
    ui_type_vocab: Dict[str, int],
    max_ui_tokens: int,
    drop_full_screen_root: bool = False,
):
    """Encode UI elements with richer geometry and stable spatial order.

    uionly3 geometry is 10-D:
      [x1, y1, x2, y2, xc, yc, w, h, area, aspect_ratio]
    all normalized to [0,1] except aspect_ratio, clipped to [0,10].
    We sort top-to-bottom then left-to-right before capping, instead of relying
    on raw segmentation JSON order.
    """
    candidates = []
    for elem in elements:
        x1, y1, x2, y2 = elem["bbox"]
        x1 = max(0.0, min(float(seg_w), float(x1)))
        y1 = max(0.0, min(float(seg_h), float(y1)))
        x2 = max(0.0, min(float(seg_w), float(x2)))
        y2 = max(0.0, min(float(seg_h), float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        ui_type = normalize_ui_type(elem.get("type", "<UNK>"))
        if should_filter_ui_element(x1, y1, x2, y2, seg_w, seg_h, ui_type, drop_full_screen_root):
            continue
        xc_abs = (x1 + x2) / 2.0
        yc_abs = (y1 + y2) / 2.0
        candidates.append((yc_abs, xc_abs, x1, y1, x2, y2, ui_type))

    candidates.sort(key=lambda z: (z[0], z[1]))

    feats, type_ids, kept_elements = [], [], []
    for _, _, x1, y1, x2, y2, ui_type in candidates[:max_ui_tokens]:
        x1n = x1 / float(seg_w)
        y1n = y1 / float(seg_h)
        x2n = x2 / float(seg_w)
        y2n = y2 / float(seg_h)
        xc = 0.5 * (x1n + x2n)
        yc = 0.5 * (y1n + y2n)
        w = max(1e-6, x2n - x1n)
        h = max(1e-6, y2n - y1n)
        area = max(0.0, min(1.0, w * h))
        aspect = max(0.0, min(10.0, w / h))
        feats.append([x1n, y1n, x2n, y2n, xc, yc, w, h, area, aspect])
        type_ids.append(ui_type_vocab.get(ui_type, ui_type_vocab["<UNK>"]))
        kept_elements.append({"bbox": [x1, y1, x2, y2], "type": ui_type})

    valid_len = len(feats)
    while len(feats) < max_ui_tokens:
        feats.append([0.0] * 10)
        type_ids.append(ui_type_vocab["<PAD>"])
    mask = [1.0] * valid_len + [0.0] * (max_ui_tokens - valid_len)
    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(type_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
        kept_elements,
    )


class ResizeWithPadding:
    def __init__(self, target_size: int, fill=(128, 128, 128)):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (self.target_size, self.target_size), self.fill)
        scale = min(self.target_size / w, self.target_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        offset_x = (self.target_size - new_w) // 2
        offset_y = (self.target_size - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


def crop_ui_regions(pil_image, kept_elements, seg_w, seg_h, max_ui_tokens, crop_size, crop_transform):
    crops = []
    img_w, img_h = pil_image.size
    for elem in kept_elements:
        x1, y1, x2, y2 = elem["bbox"]
        x1 = x1 / float(seg_w) * img_w
        y1 = y1 / float(seg_h) * img_h
        x2 = x2 / float(seg_w) * img_w
        y2 = y2 / float(seg_h) * img_h
        x1 = int(max(0, min(img_w - 1, round(x1))))
        y1 = int(max(0, min(img_h - 1, round(y1))))
        x2 = int(max(x1 + 1, min(img_w, round(x2))))
        y2 = int(max(y1 + 1, min(img_h, round(y2))))
        crop = crop_transform(pil_image.crop((x1, y1, x2, y2)))
        crops.append(crop)
        if len(crops) >= max_ui_tokens:
            break
    while len(crops) < max_ui_tokens:
        crops.append(torch.zeros(3, crop_size, crop_size, dtype=torch.float32))
    return torch.stack(crops, dim=0)


def crop_target_region(pil_image: Image.Image, bbox_norm: List[float], target_crop_size: int, crop_transform):
    """Crop the target appearance and resize it independently.

    This uses the bbox only to produce the visual cue image. The normalized bbox
    coordinates are still returned separately for supervision/evaluation, but the
    model should not receive them as coordinate input.
    """
    img_w, img_h = pil_image.size
    x0, y0, x1, y1 = bbox_norm
    left = int(max(0, min(img_w - 1, round(x0 * img_w))))
    top = int(max(0, min(img_h - 1, round(y0 * img_h))))
    right = int(max(left + 1, min(img_w, round(x1 * img_w))))
    bottom = int(max(top + 1, min(img_h, round(y1 * img_h))))
    return crop_transform(pil_image.crop((left, top, right, bottom)))


def build_hybrid_trial_samples(trials: List[Dict[str, Any]]):
    samples = []
    for t in trials:
        if not is_target_present_trial(t):
            continue
        img_name = t["key"]["img_name"]
        cue = str(t["key"]["cue"])
        seg_w = float(t["geom"]["seg_w"])
        seg_h = float(t["geom"]["seg_h"])
        scanpath = t["phases"]["search"]["scanpath"]
        if len(scanpath) < 2:
            continue
        try:
            target_bbox_norm = get_target_bbox_norm(t)
        except Exception as e:
            print(f"WARNING: skipping trial without usable target bbox: {t.get('trial_id')} ({e})")
            continue

        scanpath_xydur = []
        for p in scanpath:
            x, y = get_xy_seg_norm(p, seg_w, seg_h)
            dur = get_duration(p)
            scanpath_xydur.append([x, y, dur])

        samples.append({
            "trial_id": t["trial_id"],
            "img_name": img_name,
            "cue": cue,
            "present_trial": 1,
            "target_bbox_norm": target_bbox_norm,
            "scanpath_xydur": scanpath_xydur,
            "scanpath_len": len(scanpath_xydur),
        })
    return samples


def check_hybrid_trial_samples(samples, name="trial_samples", max_scanpath_len: Optional[int] = None):
    lengths = [int(s["scanpath_len"]) for s in samples]
    if not lengths:
        summary = {"name": name, "num_trials": 0}
        print(f"[{name}] no samples")
        return summary
    if max_scanpath_len is None:
        effective_lengths = lengths
        truncated = [False for _ in lengths]
    else:
        effective_lengths = [min(L, int(max_scanpath_len)) for L in lengths]
        truncated = [L > int(max_scanpath_len) for L in lengths]
    summary = {
        "name": name,
        "num_trials": len(samples),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": sum(lengths) / max(1, len(lengths)),
        "num_truncated": sum(1 for x in truncated if x),
        "num_prediction_steps": sum(max(0, L - 1) for L in effective_lengths),
        "num_untruncated_stop_candidates": sum(1 for L, tr in zip(effective_lengths, truncated) if L >= 2 and not tr),
    }
    print(f"[{name}] num_trials: {summary['num_trials']}")
    print(f"[{name}] len min/max/mean: {summary['min_len']} / {summary['max_len']} / {summary['mean_len']:.3f}")
    print(f"[{name}] num_truncated: {summary['num_truncated']}")
    print(f"[{name}] prediction_steps: {summary['num_prediction_steps']}")
    return summary


class HybridTrialDecoderTargetCropDataset(Dataset):
    """One item = one full target-present trial.

    Model inputs include the target visual crop, but not the target bbox
    coordinates. The bbox remains in the batch for target-aware STOP labels,
    STOP-far-from-target loss, and evaluation.
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        trial_index: Dict[str, Dict[str, Any]],
        cue_vocab: Dict[str, int],
        ui_type_vocab: Dict[str, int],
        seg_index: Dict[str, str],
        image_dir: str,
        image_size: int = 224,
        max_ui_tokens: int = 64,
        drop_full_screen_root: bool = False,
        crop_size: int = 32,
        max_scanpath_len: int = 20,
        target_crop_size: int = 48,
    ):
        self.samples = samples
        self.trial_index = trial_index
        self.cue_vocab = cue_vocab
        self.ui_type_vocab = ui_type_vocab
        self.seg_index = seg_index
        self.image_dir = image_dir
        self.image_size = int(image_size)
        self.max_ui_tokens = int(max_ui_tokens)
        self.drop_full_screen_root = bool(drop_full_screen_root)
        self.crop_size = int(crop_size)
        self.max_scanpath_len = int(max_scanpath_len)
        self.target_crop_size = int(target_crop_size)

        if self.max_scanpath_len < 2:
            raise ValueError("max_scanpath_len must be >= 2.")

        self.img_tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        self.crop_tf = transforms.Compose([
            ResizeWithPadding(self.crop_size),
            transforms.ToTensor(),
        ])
        self.target_crop_tf = transforms.Compose([
            ResizeWithPadding(self.target_crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        trial = self.trial_index[s["trial_id"]]
        cue_id = self.cue_vocab.get(s["cue"], 0)

        raw_scanpath = s["scanpath_xydur"]
        raw_len = int(s["scanpath_len"])
        effective_len = min(raw_len, self.max_scanpath_len)
        is_truncated = 1.0 if raw_len > self.max_scanpath_len else 0.0

        scanpath = raw_scanpath[:effective_len]
        scanpath_mask = [1.0] * effective_len
        while len(scanpath) < self.max_scanpath_len:
            scanpath.append([0.0, 0.0, 0.0])
            scanpath_mask.append(0.0)

        img_path = os.path.join(self.image_dir, s["img_name"])
        pil_image = Image.open(img_path).convert("RGB")
        image = self.img_tf(pil_image)

        seg_w = float(trial["geom"]["seg_w"])
        seg_h = float(trial["geom"]["seg_h"])

        ui_elements = load_ui_elements_from_seg(self.seg_index, s["img_name"])
        ui_geom, ui_type_id, ui_mask, kept_elements = encode_ui_elements(
            elements=ui_elements,
            seg_w=seg_w,
            seg_h=seg_h,
            ui_type_vocab=self.ui_type_vocab,
            max_ui_tokens=self.max_ui_tokens,
            drop_full_screen_root=self.drop_full_screen_root,
        )
        ui_crop_images = crop_ui_regions(
            pil_image=pil_image,
            kept_elements=kept_elements,
            seg_w=seg_w,
            seg_h=seg_h,
            max_ui_tokens=self.max_ui_tokens,
            crop_size=self.crop_size,
            crop_transform=self.crop_tf,
        )

        target_bbox_norm = torch.tensor(s["target_bbox_norm"], dtype=torch.float32)
        target_crop_image = crop_target_region(
            pil_image=pil_image,
            bbox_norm=s["target_bbox_norm"],
            target_crop_size=self.target_crop_size,
            crop_transform=self.target_crop_tf,
        )

        return {
            "image": image,
            "cue_id": torch.tensor(cue_id, dtype=torch.long),
            "target_crop_image": target_crop_image,

            "scanpath_xydur": torch.tensor(scanpath, dtype=torch.float32),
            "scanpath_mask": torch.tensor(scanpath_mask, dtype=torch.float32),
            "scanpath_len": torch.tensor(effective_len, dtype=torch.long),
            "raw_scanpath_len": torch.tensor(raw_len, dtype=torch.long),
            "is_truncated": torch.tensor(is_truncated, dtype=torch.float32),
            "trial_id": s["trial_id"],

            "present_trial": torch.tensor(s.get("present_trial", 1), dtype=torch.long),
            "target_bbox_norm": target_bbox_norm,

            "ui_geom": ui_geom,
            "ui_type_id": ui_type_id,
            "ui_mask": ui_mask,
            "ui_crop_images": ui_crop_images,
        }


# Backward-compatible alias if you want to import a familiar name.
HybridTrialDecoderDataset = HybridTrialDecoderTargetCropDataset
