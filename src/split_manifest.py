"""Deterministic, speaker-group-disjoint 80/10/10 TSV splits.

Does not prevent transcript or acoustic duplication; audit those separately.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def split_rows(rows: list[dict[str, str]], seed: int = 42):
    if len(rows) < 3:
        raise ValueError("Need at least three rows")
    speakers = [(r.get("client_id") or "").strip() for r in rows]
    if any(not s for s in speakers):
        raise ValueError("Missing client_id; cannot guarantee speaker-disjoint splits")
    if len(set(speakers)) < 3:
        raise ValueError("Need at least three distinct speakers")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_val_idx, test_idx = next(outer.split(rows, groups=speakers))
    train_val = [rows[i] for i in train_val_idx]
    tv_groups = [speakers[i] for i in train_val_idx]
    if len(set(tv_groups)) < 2:
        raise ValueError("Not enough speakers for train/validation")
    inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=seed)
    train_idx, val_idx = next(inner.split(train_val, groups=tv_groups))
    result = {
        "train": [train_val[i] for i in train_idx],
        "validation": [train_val[i] for i in val_idx],
        "test": [rows[i] for i in test_idx],
    }
    groups = [{r["client_id"] for r in v} for v in result.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        rows = list(reader)
    output = split_rows(rows, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in output.items():
        with (args.output_dir / f"{name}.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(split)
        print(f"{name}: {len(split)} rows, {len({r['client_id'] for r in split})} speakers")


if __name__ == "__main__":
    main()
