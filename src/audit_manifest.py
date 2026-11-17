"""Conservative Common Voice TSV metadata audit (not audio cleaning).

Usage:
python src/audit_manifest.py --input data/raw/validated.tsv \
 --audio-root data/raw/clips --output data/cleaned/validated_cleaned.tsv \
 --log data/cleaned/cleaning_log.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def audit_rows(rows: list[dict[str, str]], audio_root: Path):
    """Return accepted metadata and a reason per rejected row.

    This checks metadata and file *presence*, not waveform quality.
    """
    accepted, rejects = [], []
    seen_paths: set[str] = set()
    for index, item in enumerate(rows):
        path = (item.get("path") or "").strip()
        sentence = (item.get("sentence") or "").strip()
        speaker = (item.get("client_id") or "").strip()
        # Reject absolute/traversal paths; dataset archive must stay in audio_root.
        normalized = Path(path)
        reasons = []
        if not sentence:
            reasons.append("empty_transcript")
        if not speaker:
            reasons.append("missing_speaker")
        if not path or normalized.is_absolute() or ".." in normalized.parts:
            reasons.append("invalid_clip_path")
        elif not (audio_root / normalized).is_file():
            reasons.append("missing_audio_file")
        if path in seen_paths:
            reasons.append("duplicate_clip_path")
        if reasons:
            rejects.append({"row_index": index, "path": path, "reason": "|".join(reasons)})
        else:
            accepted.append(dict(item))
            seen_paths.add(path)
    return accepted, rejects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        required = {"path", "sentence", "client_id"}
        if not required.issubset(fields):
            raise ValueError(f"Missing manifest columns: {sorted(required - set(fields))}")
        rows = list(reader)
    accepted, rejects = audit_rows(rows, args.audio_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(accepted)
    with args.log.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "path", "reason"])
        writer.writeheader()
        writer.writerows(rejects)
    print(f"Accepted {len(accepted)} / {len(rows)} rows; excluded {len(rejects)}. Metadata-only audit.")


if __name__ == "__main__":
    main()
