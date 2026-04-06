#!/usr/bin/env python3
"""Sample balanced dataset from a JSONL face dataset.

Reads an input JSONL where each line is a JSON object with at least the
fields: `id`, `img` (relative image path), and `label` (0 or 1). Produces
an output JSONL with the sampled records. The output `img` field keeps the
original relative image path from the source dataset.

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
import sys
from pathlib import Path
from typing import Dict, List, Any


# --- Simple config (no CLI required) ---
# Resolve paths relative to the repository root (parent of this script's folder)
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "facebook-data" / "train.jsonl"
IMAGES_ROOT = BASE / "facebook-data"
# Output folder inside datapreparation: datapreparation/output
OUTPUT_DIR = BASE / "datapreparation" / "output"
OUTPUT_JSONL = OUTPUT_DIR / "facebook-samples-test-roberta.jsonl"
EXCLUDE_JSONL = OUTPUT_DIR / "facebook-samples-test-rationale-to-caption.jsonl"
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


def load_exclusion_keys(path: Path) -> tuple[set[int], set[str]]:
    excluded_ids: set[int] = set()
    excluded_images: set[str] = set()

    if not path.exists():
        print(f"Exclusion file not found: {path}", file=sys.stderr)
        sys.exit(2)

    for record in load_jsonl(path):
        post_id = record.get("id")
        if post_id is not None:
            try:
                excluded_ids.add(int(post_id))
            except (TypeError, ValueError):
                pass

        img_field = record.get("img")
        if img_field:
            excluded_images.add(Path(str(img_field)).name)

    return excluded_ids, excluded_images


def filter_excluded_records(
    records: List[Dict[str, Any]],
    excluded_ids: set[int],
    excluded_images: set[str],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for record in records:
        post_id = record.get("id")
        if post_id is not None:
            try:
                if int(post_id) in excluded_ids:
                    continue
            except (TypeError, ValueError):
                pass

        img_field = record.get("img")
        if img_field and Path(str(img_field)).name in excluded_images:
            continue

        filtered.append(record)

    return filtered


def group_records_by_label(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_label = {0: [], 1: []}
    for record in records:
        lab = record.get("label")
        if lab in (0, 1):
            by_label[lab].append(record)
    return by_label


def sample_balanced_records(records: List[Dict[str, Any]], n: int, rnd: random.Random) -> List[Dict[str, Any]]:
    by_label = group_records_by_label(records)
    need_each = n // 2
    if len(by_label[0]) < need_each or len(by_label[1]) < need_each:
        print(
            f"Not enough samples to satisfy 50/50: need {need_each} of each label; "
            f"found {len(by_label[0])} zeros and {len(by_label[1])} ones",
            file=sys.stderr,
        )
        sys.exit(3)

    sampled = rnd.sample(by_label[0], need_each) + rnd.sample(by_label[1], need_each)
    rnd.shuffle(sampled)
    return sampled


def validate_sampled_images(sampled: List[Dict[str, Any]]) -> List[str]:
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

    return missing


def write_output(sampled: List[Dict[str, Any]]) -> None:
    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for rec in sampled:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
    excluded_ids, excluded_images = load_exclusion_keys(EXCLUDE_JSONL)
    records = filter_excluded_records(records, excluded_ids, excluded_images)
    sampled = sample_balanced_records(records, n, rnd)
    missing = validate_sampled_images(sampled)

    if missing:
        print(f"Error: {len(missing)} referenced images were not found. Examples: {missing[:5]}", file=sys.stderr)
        sys.exit(4)

    write_output(sampled)

    print(f"Wrote {len(sampled)} records to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
