#!/usr/bin/env python3
"""Sample balanced dataset from a JSONL face dataset and copy images.

Reads an input JSONL where each line is a JSON object with at least the
fields: `id`, `img` (relative image path), and `label` (0 or 1). Produces
an output JSONL with the sampled records and copies referenced images into
an output images folder. The output `img` field is updated to point to the
copied image path (relative to the JSONL file location).

Usage example:
    python datapreparation/sample_faces.py -n 100 \
        -i facebook-data/dev.jsonl -r facebook-data \
        -o samples.jsonl -d sampled_images --seed 42

Requirements: only Python standard library (3.7+). The input `--n` must be
even so the script can select 50% label=1 and 50% label=0.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any


# --- Simple config (no CLI required) ---
# Resolve paths relative to the repository root (parent of this script's folder)
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "facebook-data" / "dev.jsonl"
IMAGES_ROOT = BASE / "facebook-data"
# Output folder inside datapreparation: datapreparation/output
OUTPUT_DIR = BASE / "datapreparation" / "output"
OUTPUT_JSONL = OUTPUT_DIR / "facebook-samples.jsonl"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "facebook-images"
DEFAULT_N = 400  # must be even
SEED = 42


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON at {path}:{ln_no}: {e}", file=sys.stderr)
                continue
            items.append(obj)
    return items


def ask_n(default: int) -> int:
    try:
        s = input(f"Number of samples to create (even, default {default}): ").strip()
        if not s:
            return default
        n = int(s)
        return n
    except Exception:
        print("Invalid number, using default.")
        return default


def main() -> None:
    n = ask_n(DEFAULT_N)
    if n % 2 != 0:
        print("Number must be even. Exiting.")
        sys.exit(2)

    rnd = random.Random(SEED)

    if not INPUT.exists():
        print(f"Input file not found: {INPUT}", file=sys.stderr)
        sys.exit(2)

    records = load_jsonl(INPUT)
    by_label = {0: [], 1: []}
    for r in records:
        lab = r.get("label")
        if lab in (0, 1):
            by_label[lab].append(r)

    need_each = n // 2
    if len(by_label[0]) < need_each or len(by_label[1]) < need_each:
        print(
            f"Not enough samples to satisfy 50/50: need {need_each} of each label; "
            f"found {len(by_label[0])} zeros and {len(by_label[1])} ones",
            file=sys.stderr,
        )
        sys.exit(3)

    sample0 = rnd.sample(by_label[0], need_each)
    sample1 = rnd.sample(by_label[1], need_each)
    sampled = sample0 + sample1
    rnd.shuffle(sampled)

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    for rec in sampled:
        img_field = rec.get("img")
        if not img_field:
            missing.append(f"id={rec.get('id')}: no 'img' field")
            continue

        src = (IMAGES_ROOT / Path(img_field)).resolve()
        if not src.exists():
            alt = (IMAGES_ROOT / Path(img_field).name).resolve()
            if alt.exists():
                src = alt

        if not src.exists():
            missing.append(str(src))
            continue

        dst = OUTPUT_IMAGES_DIR / Path(src.name)
        shutil.copy2(src, dst)
        rec["img"] = str(Path(OUTPUT_IMAGES_DIR.name) / src.name)

    if missing:
        print(f"Error: {len(missing)} referenced images were not found. Examples: {missing[:5]}", file=sys.stderr)
        sys.exit(4)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for rec in sampled:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} records to {OUTPUT_JSONL} and copied images to {OUTPUT_IMAGES_DIR}")


if __name__ == "__main__":
    main()
