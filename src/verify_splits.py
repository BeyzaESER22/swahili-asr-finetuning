"""T2 - Independent QA of the frozen FLEURS sw_ke splits prepared in T1.

This does NOT re-split the data. T1 preserved the official FLEURS train/dev/test
boundaries (speaker-disjoint according to the FLEURS authors; the TSVs expose no
speaker IDs). This script re-checks T1's claims with code written separately from
`prepare_fleurs.py`, so the split can be trusted by E1/E2/E3.

HARD checks (any failure -> exit code 1):
  H1  required columns present; no repeated filename inside a split
  H2  each split comes only from its official source split (validation <- dev)
  H3  every split row is `accepted` in the audit and is not in the cleaning log
  H4  splits together == data/cleaned/manifest.tsv, row for row
  H5  zero cross-split overlap of filename, sentence_id, normalised transcript, audio SHA-256
  H6  per-clip format: 16 kHz, mono, duration == num_samples / rate, duration <= 30 s
  H7  clip counts and hours match data/audit/summary.json; total <= 10 h
  H8  (with --audio-dir) each WAV exists, its SHA-256 matches the manifest and its
      header says IEEE-float32 mono 16 kHz with the expected sample count
REVIEW items (reported, never failures):
  R1  near-duplicate transcripts across splits (character-trigram Jaccard >= threshold)
  R2  gender balance per split, recordings per sentence
  R3  speaker disjointness cannot be proven by ID (no speaker column in source)

Usage:
python src/verify_splits.py                                   # metadata checks
python src/verify_splits.py --audio-dir data/cleaned/audio     # + audio checks (after `git lfs pull`)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
SOURCE_OF = {"train": "train", "validation": "dev", "test": "test"}
REQUIRED = ("sentence_id", "filename", "raw_transcription", "num_samples", "gender", "source_split",
            "duration_seconds", "sample_rate_hz", "channels", "sha256", "audit_status")
LFS_PREFIX = b"version https://git-lfs"


# ------------------------------------------------------------------ helpers
def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") and c != "'" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def trigrams(text: str) -> set[str]:
    t = f"  {text} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def wav_header(data: bytes) -> dict:
    """Minimal RIFF parser written independently of prepare_fleurs.py."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, n_bytes = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[pos + 8:pos + 24])
        elif cid == b"data":
            n_bytes = size
            break
        pos += 8 + size + (size & 1)
    if fmt is None or n_bytes is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, rate, _, block, bits = fmt
    return {"format_tag": tag, "channels": channels, "rate": rate, "bits": bits,
            "frames": n_bytes // block if block else 0}


# ------------------------------------------------------------------ checks
def verify(splits: dict[str, list[dict]], cleaned: list[dict], log: list[dict], summary: dict | None,
           audio_dir: Path | None = None, near_dup: float = 0.8, max_hours: float = 10.0) -> dict:
    fail: list[str] = []
    checks: dict[str, str] = {}

    def check(code: str, problems: list[str]):
        checks[code] = "PASS" if not problems else "FAIL"
        fail.extend(f"{code}: {p}" for p in problems)

    # H1
    p = []
    for name, rows in splits.items():
        missing = [c for c in REQUIRED if rows and c not in rows[0]]
        if missing:
            p.append(f"{name} missing columns {missing}")
        dup = [f for f, n in Counter(r["filename"] for r in rows).items() if n > 1]
        if dup:
            p.append(f"{name} repeats {len(dup)} filenames, e.g. {dup[:3]}")
        if not rows:
            p.append(f"{name} is empty")
    check("H1_structure", p)
    if p:
        return {"passed": False, "failures": fail, "checks": checks}

    # H2
    check("H2_official_boundaries", [
        f"{name}: {sum(r['source_split'] != SOURCE_OF[name] for r in rows)} rows not from '{SOURCE_OF[name]}'"
        for name, rows in splits.items() if any(r["source_split"] != SOURCE_OF[name] for r in rows)])

    # H3
    rejected = {r["filename"] for r in log}
    p = []
    for name, rows in splits.items():
        bad = [r["filename"] for r in rows if r["audit_status"] != "accepted" or r["filename"] in rejected]
        if bad:
            p.append(f"{name}: {len(bad)} rejected/unaccepted clips, e.g. {bad[:3]}")
    check("H3_only_accepted", p)

    # H4
    by_file = {r["filename"]: r for r in cleaned}
    union = [r for rows in splits.values() for r in rows]
    p = []
    if len(union) != len(cleaned):
        p.append(f"splits have {len(union)} rows, cleaned manifest has {len(cleaned)}")
    not_in = [r["filename"] for r in union if r["filename"] not in by_file]
    differs = [r["filename"] for r in union if r["filename"] in by_file and r != by_file[r["filename"]]]
    if not_in:
        p.append(f"{len(not_in)} split rows absent from cleaned manifest, e.g. {not_in[:3]}")
    if differs:
        p.append(f"{len(differs)} split rows differ from cleaned manifest, e.g. {differs[:3]}")
    check("H4_matches_cleaned_manifest", p)

    # H5
    keys = {
        "filename": lambda r: r["filename"],
        "sentence_id": lambda r: r["sentence_id"],
        "normalized_transcript": lambda r: norm_text(r["raw_transcription"]),
        "audio_sha256": lambda r: r["sha256"],
    }
    overlap, p = {}, []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            pair = overlap.setdefault(f"{a}/{b}", {})
            for key, fn in keys.items():
                shared = {fn(r) for r in splits[a]} & {fn(r) for r in splits[b]}
                pair[key] = len(shared)
                if shared:
                    p.append(f"{a}/{b} share {len(shared)} {key}, e.g. {sorted(shared)[:2]}")
    check("H5_no_cross_split_overlap", p)

    # H6
    p = []
    for name, rows in splits.items():
        for r in rows:
            rate, n, dur = int(r["sample_rate_hz"]), int(r["num_samples"]), float(r["duration_seconds"])
            if rate != 16000 or r["channels"] != "1":
                p.append(f"{name}/{r['filename']}: {rate} Hz, {r['channels']} ch")
            if abs(n / rate - dur) > 1e-3:
                p.append(f"{name}/{r['filename']}: duration {dur} != {n}/{rate}")
            if dur > 30.0 + 1e-9 or dur <= 0:
                p.append(f"{name}/{r['filename']}: duration {dur}s outside (0, 30]")
    check("H6_clip_format", p[:20] + ([f"... {len(p) - 20} more"] if len(p) > 20 else []))

    # H7
    stats = {}
    for name, rows in splits.items():
        stats[name] = {
            "clips": len(rows),
            "hours": round(sum(float(r["duration_seconds"]) for r in rows) / 3600, 6),
            "sentence_ids": len({r["sentence_id"] for r in rows}),
            "recordings_per_sentence": round(len(rows) / len({r["sentence_id"] for r in rows}), 3),
            "gender": dict(Counter(r["gender"] for r in rows)),
            "min_s": min(float(r["duration_seconds"]) for r in rows),
            "max_s": max(float(r["duration_seconds"]) for r in rows),
        }
    total = round(sum(s["hours"] for s in stats.values()), 6)
    p = [] if total <= max_hours else [f"total {total} h exceeds {max_hours} h"]
    if summary:
        for name in SPLITS:
            want = summary["splits"][name]
            if want["selected_rows"] != stats[name]["clips"]:
                p.append(f"{name}: {stats[name]['clips']} clips, summary.json says {want['selected_rows']}")
            if abs(want["selected_hours"] - stats[name]["hours"]) > 1e-3:
                p.append(f"{name}: {stats[name]['hours']} h, summary.json says {want['selected_hours']}")
    check("H7_counts_and_hours", p)

    # H8
    audio = {"checked": 0, "lfs_pointer_not_downloaded": 0, "missing": 0}
    if audio_dir is not None:
        p = []
        for name, rows in splits.items():
            for r in rows:
                path = audio_dir / name / r["filename"]
                if not path.is_file():
                    audio["missing"] += 1
                    p.append(f"missing {path}")
                    continue
                data = path.read_bytes()
                if data.startswith(LFS_PREFIX):
                    audio["lfs_pointer_not_downloaded"] += 1
                    continue
                audio["checked"] += 1
                if hashlib.sha256(data).hexdigest() != r["sha256"]:
                    p.append(f"{name}/{r['filename']}: SHA-256 mismatch")
                    continue
                try:
                    h = wav_header(data)
                except ValueError as exc:
                    p.append(f"{name}/{r['filename']}: {exc}")
                    continue
                if (h["format_tag"], h["channels"], h["rate"], h["bits"]) != (3, 1, 16000, 32) \
                        or h["frames"] != int(r["num_samples"]):
                    p.append(f"{name}/{r['filename']}: header {h} vs num_samples {r['num_samples']}")
        if audio["lfs_pointer_not_downloaded"]:
            p.append(f"{audio['lfs_pointer_not_downloaded']} files are Git LFS pointers - run `git lfs pull`")
        check("H8_audio_files", p[:20] + ([f"... {len(p) - 20} more"] if len(p) > 20 else []))
    else:
        checks["H8_audio_files"] = "SKIPPED (no --audio-dir)"

    # R1 near-duplicate transcripts across splits (different sentence_id, very similar text)
    uniq = {name: {r["sentence_id"]: norm_text(r["raw_transcription"]) for r in rows} for name, rows in splits.items()}
    grams = {name: {sid: trigrams(t) for sid, t in d.items()} for name, d in uniq.items()}
    near = []
    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        for sa, ga in grams[a].items():
            for sb, gb in grams[b].items():
                score = jaccard(ga, gb)
                if score >= near_dup:
                    near.append({"pair": f"{a}/{b}", "score": round(score, 3), a: uniq[a][sa], b: uniq[b][sb]})
    near.sort(key=lambda x: -x["score"])

    return {
        "passed": not fail,
        "failures": fail,
        "checks": checks,
        "splits": stats,
        "total_hours": total,
        "cross_split_overlap": overlap,
        "audio": audio,
        "review": {
            "R1_near_duplicate_transcripts": {"threshold": near_dup, "count": len(near), "examples": near[:20]},
            "R2_gender_per_split": {n: s["gender"] for n, s in stats.items()},
            "R3_speaker_ids": "Not available in FLEURS TSVs. Speaker disjointness relies on the official FLEURS "
                              "split design (Conneau et al., 2022) and cannot be verified by ID here.",
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--audio-dir", type=Path, help="e.g. data/cleaned/audio (needs real audio: git lfs pull)")
    p.add_argument("--near-dup-threshold", type=float, default=0.8)
    p.add_argument("--max-total-hours", type=float, default=10.0)
    p.add_argument("--report", type=Path, help="default: <data-dir>/splits/qa_report.json")
    args = p.parse_args()

    d = args.data_dir
    splits = {name: read_tsv(d / "splits" / f"{name}.tsv") for name in SPLITS}
    summary_path = d / "audit" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    report = verify(splits, read_tsv(d / "cleaned" / "manifest.tsv"), read_tsv(d / "cleaned" / "cleaning_log.tsv"),
                    summary, args.audio_dir, args.near_dup_threshold, args.max_total_hours)
    out = args.report or d / "splits" / "qa_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'split':<11}{'clips':>7}{'hours':>10}{'sentences':>11}{'female':>8}{'male':>6}")
    for name, s in report.get("splits", {}).items():
        print(f"{name:<11}{s['clips']:>7}{s['hours']:>10.4f}{s['sentence_ids']:>11}"
              f"{s['gender'].get('FEMALE', 0):>8}{s['gender'].get('MALE', 0):>6}")
    print(f"{'total':<11}{'':>7}{report.get('total_hours', 0):>10.4f}\n")
    for code, status in report["checks"].items():
        print(f"  {status:<8} {code}")
    r1 = report.get("review", {}).get("R1_near_duplicate_transcripts")
    if r1:
        print(f"\n  REVIEW   {r1['count']} near-duplicate transcript pairs across splits (>= {r1['threshold']})")
    print(f"\nReport: {out}")
    if not report["passed"]:
        print("FAILED:\n  " + "\n  ".join(report["failures"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
