"""Score prediction files with the shared WER/CER code (src/metrics.py).

Refuses to score unless every prediction file covers exactly the frozen test IDs
with the frozen reference text, so E1/E2/E3 are compared on identical data.
Writes results/metrics.csv - never edit that file by hand.

Usage:
python src/score_predictions.py --test-manifest data/splits/test.tsv \
  --predictions results/predictions/E1.tsv results/predictions/E2.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from metrics import evaluate_asr
from run_baseline import ID_COL, REF_COL


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def check_alignment(test_rows: list[dict], pred_rows: list[dict], name: str) -> None:
    gold = {r[ID_COL]: r[REF_COL] for r in test_rows}
    ids = [r["id"] for r in pred_rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate ids")
    missing, extra = set(gold) - set(ids), set(ids) - set(gold)
    if missing or extra:
        raise ValueError(f"{name}: {len(missing)} test ids missing, {len(extra)} unexpected ids")
    changed = [r["id"] for r in pred_rows if r["reference"] != gold[r["id"]]]
    if changed:
        raise ValueError(f"{name}: reference text differs from frozen test set, e.g. {changed[:3]}")


def score(test_rows: list[dict], pred_rows: list[dict], name: str) -> dict:
    check_alignment(test_rows, pred_rows, name)
    result = evaluate_asr([r["reference"] for r in pred_rows], [r["hypothesis"] for r in pred_rows])
    return {"run": name, "n_utterances": len(pred_rows), **{k: round(v, 4) for k, v in result.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-manifest", type=Path, default=Path("data/splits/test.tsv"))
    p.add_argument("--predictions", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=Path("results/metrics.csv"))
    args = p.parse_args()

    test_rows = read_tsv(args.test_manifest)
    table = []
    for pred in args.predictions:
        row = score(test_rows, read_tsv(pred), pred.stem)
        meta = pred.parent.parent / "runs" / f"{pred.stem}.json"
        if meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            row.update(role=info.get("role"), model_id=info.get("model_id"), model_commit=info.get("model_commit"))
        table.append(row)
        print(f"{row['run']:<6} WER {row['wer']:.4f}  CER {row['cer']:.4f}  (n={row['n_utterances']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "role", "model_id", "model_commit", "n_utterances", "wer", "cer"]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
