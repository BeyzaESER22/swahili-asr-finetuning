"""Audit and select a reproducible <=10-hour FLEURS ``sw_ke`` subset.

The original FLEURS TSV files have no header. Their seven fields are:
sentence_id, filename, raw_transcription, transcription, characters,
num_samples, gender.  They do *not* expose speaker identifiers.  This tool
therefore preserves the official train/dev/test boundaries (documented as
speaker-disjoint by the FLEURS authors) and never claims to create a new
speaker split.

Waveforms are not denoised or transcript-corrected. Structural failures are
rejected; high silence/clipping ratios are logged as review warnings. The
selected cleaned audio is byte-identical to raw audio when extraction is
requested, making the cleaning operation an auditable selection rather than
an undocumented signal transformation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import tarfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


SOURCE_COLUMNS = (
    "sentence_id",
    "filename",
    "raw_transcription",
    "transcription",
    "characters",
    "num_samples",
    "gender",
)
SPLITS = ("train", "dev", "test")
OUTPUT_SPLIT_NAMES = {"train": "train", "dev": "validation", "test": "test"}
MANIFEST_COLUMNS = SOURCE_COLUMNS + (
    "source_split",
    "archive_member",
    "duration_seconds",
    "sample_rate_hz",
    "channels",
    "bits_per_sample",
    "audio_format",
    "peak_abs",
    "rms",
    "silence_ratio",
    "clipping_ratio",
    "sha256",
    "audit_status",
    "audit_reasons",
)


@dataclass
class WaveStats:
    audio_format: int
    channels: int
    sample_rate_hz: int
    bits_per_sample: int
    block_align: int
    num_frames: int
    data_bytes: int
    peak_abs: float
    rms: float
    silence_ratio: float
    clipping_ratio: float
    finite: bool
    sha256: str


def read_source_tsv(path: Path, split: str) -> list[dict[str, str]]:
    """Read a literal tab-separated FLEURS TSV without CSV quote semantics."""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            values = raw_line.rstrip("\r\n").split("\t")
            if len(values) != len(SOURCE_COLUMNS):
                raise ValueError(
                    f"{path}:{line_number}: expected 7 tab-separated fields, "
                    f"found {len(values)}"
                )
            row = dict(zip(SOURCE_COLUMNS, values, strict=True))
            row["source_split"] = split
            row["source_line"] = str(line_number)
            rows.append(row)
    return rows


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"truncated WAV: wanted {size} bytes, found {len(data)}")
    return data


def inspect_float_wav(handle: BinaryIO, *, chunk_bytes: int = 4 * 1024 * 1024) -> WaveStats:
    """Validate a RIFF/WAVE file and compute exact Float32 signal statistics."""
    digest = hashlib.sha256()

    def take(size: int) -> bytes:
        data = _read_exact(handle, size)
        digest.update(data)
        return data

    if take(4) != b"RIFF":
        raise ValueError("not a little-endian RIFF file")
    _riff_size = struct.unpack("<I", take(4))[0]
    if take(4) != b"WAVE":
        raise ValueError("RIFF payload is not WAVE")

    fmt: tuple[int, int, int, int, int] | None = None
    data_size: int | None = None
    peak = 0.0
    sum_squares = 0.0
    sample_count = 0
    silence_count = 0
    clipping_count = 0
    all_finite = True

    while True:
        header = handle.read(8)
        if not header:
            break
        if len(header) != 8:
            raise ValueError("truncated RIFF chunk header")
        digest.update(header)
        chunk_id, size = struct.unpack("<4sI", header)
        if chunk_id == b"fmt ":
            payload = take(size)
            if size < 16:
                raise ValueError("WAV fmt chunk is shorter than 16 bytes")
            audio_format, channels, rate, _byte_rate, block_align, bits = struct.unpack(
                "<HHIIHH", payload[:16]
            )
            fmt = (audio_format, channels, rate, block_align, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("WAV data chunk appears before fmt chunk")
            audio_format, channels, rate, block_align, bits = fmt
            if audio_format != 3 or bits != 32:
                raise ValueError(
                    f"unsupported WAV encoding format={audio_format}, bits={bits}; "
                    "expected IEEE Float32"
                )
            data_size = size
            remaining = size
            while remaining:
                payload = take(min(chunk_bytes, remaining))
                remaining -= len(payload)
                if len(payload) % 4:
                    raise ValueError("Float32 audio payload is not 4-byte aligned")
                samples = np.frombuffer(payload, dtype="<f4")
                finite_mask = np.isfinite(samples)
                if not bool(finite_mask.all()):
                    all_finite = False
                    samples = samples[finite_mask]
                if samples.size:
                    absolute = np.abs(samples.astype(np.float64, copy=False))
                    peak = max(peak, float(absolute.max()))
                    sum_squares += float(np.dot(absolute, absolute))
                    silence_count += int(np.count_nonzero(absolute < 1e-4))
                    clipping_count += int(np.count_nonzero(absolute >= 0.999))
                    sample_count += int(samples.size)
        else:
            remaining = size
            while remaining:
                payload = take(min(chunk_bytes, remaining))
                remaining -= len(payload)

        if size % 2:
            take(1)

    if fmt is None or data_size is None:
        raise ValueError("WAV is missing fmt or data chunk")
    audio_format, channels, rate, block_align, bits = fmt
    if block_align <= 0 or data_size % block_align:
        raise ValueError("WAV data size is not aligned to complete frames")
    frame_count = data_size // block_align
    denominator = sample_count or 1
    return WaveStats(
        audio_format=audio_format,
        channels=channels,
        sample_rate_hz=rate,
        bits_per_sample=bits,
        block_align=block_align,
        num_frames=frame_count,
        data_bytes=data_size,
        peak_abs=peak,
        rms=math.sqrt(sum_squares / denominator),
        silence_ratio=silence_count / denominator,
        clipping_ratio=clipping_count / denominator,
        finite=all_finite,
        sha256=digest.hexdigest(),
    )


def audit_split(input_dir: Path, split: str) -> list[dict[str, str]]:
    rows = read_source_tsv(input_dir / f"{split}.tsv", split)
    by_filename: dict[str, dict[str, str]] = {}
    duplicate_filenames: set[str] = set()
    for row in rows:
        filename = row["filename"]
        if filename in by_filename:
            duplicate_filenames.add(filename)
        by_filename[filename] = row

    expected_members = {f"{split}/{name}" for name in by_filename}
    seen_members: set[str] = set()
    unexpected_members: list[str] = []
    results: list[dict[str, str]] = []
    archive_path = input_dir / f"{split}.tar"

    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name not in expected_members:
                unexpected_members.append(member.name)
                continue
            if member.name in seen_members:
                raise ValueError(f"{archive_path}: duplicate archive member {member.name}")
            seen_members.add(member.name)
            row = dict(by_filename[Path(member.name).name])
            reasons: list[str] = []
            warnings: list[str] = []
            extracted = archive.extractfile(member)
            if extracted is None:
                reasons.append("unreadable_archive_member")
                stats = None
            else:
                try:
                    stats = inspect_float_wav(extracted)
                except (OSError, ValueError, struct.error) as error:
                    reasons.append(f"invalid_wav:{error}")
                    stats = None

            if not row["raw_transcription"].strip():
                reasons.append("empty_raw_transcription")
            if not row["transcription"].strip():
                reasons.append("empty_normalized_transcription")
            if not row["characters"].strip():
                reasons.append("empty_character_transcription")
            if row["gender"] not in {"FEMALE", "MALE", "OTHER"}:
                warnings.append("unexpected_gender_label")
            if row["filename"] in duplicate_filenames:
                reasons.append("duplicate_filename_in_tsv")
            try:
                declared_samples = int(row["num_samples"])
            except ValueError:
                declared_samples = -1
                reasons.append("invalid_num_samples")

            result = {key: row.get(key, "") for key in SOURCE_COLUMNS}
            result["source_split"] = split
            result["archive_member"] = member.name
            if stats is not None:
                if stats.audio_format != 3:
                    reasons.append("not_ieee_float_wav")
                if stats.channels != 1:
                    reasons.append("not_mono")
                if stats.sample_rate_hz != 16000:
                    reasons.append("sample_rate_not_16000")
                if stats.bits_per_sample != 32:
                    reasons.append("not_float32")
                if stats.num_frames != declared_samples:
                    reasons.append("num_samples_mismatch")
                if stats.num_frames <= 0:
                    reasons.append("empty_audio")
                if stats.num_frames / stats.sample_rate_hz > 30.0:
                    reasons.append("duration_over_30_seconds")
                if not stats.finite:
                    reasons.append("nonfinite_samples")
                if stats.rms <= 1e-8:
                    reasons.append("zero_energy_audio")
                if stats.silence_ratio >= 0.98:
                    warnings.append("silence_ratio_ge_0.98_review")
                if stats.clipping_ratio >= 0.01:
                    warnings.append("clipping_ratio_ge_0.01_review")
                result.update(
                    {
                        "duration_seconds": f"{stats.num_frames / stats.sample_rate_hz:.6f}",
                        "sample_rate_hz": str(stats.sample_rate_hz),
                        "channels": str(stats.channels),
                        "bits_per_sample": str(stats.bits_per_sample),
                        "audio_format": "IEEE_FLOAT" if stats.audio_format == 3 else str(stats.audio_format),
                        "peak_abs": f"{stats.peak_abs:.9f}",
                        "rms": f"{stats.rms:.9f}",
                        "silence_ratio": f"{stats.silence_ratio:.9f}",
                        "clipping_ratio": f"{stats.clipping_ratio:.9f}",
                        "sha256": stats.sha256,
                    }
                )
            else:
                for key in MANIFEST_COLUMNS:
                    result.setdefault(key, "")
            result["audit_status"] = "rejected" if reasons else ("review" if warnings else "accepted")
            result["audit_reasons"] = "|".join(reasons + warnings) or "none"
            results.append(result)

    if unexpected_members:
        sample = ", ".join(sorted(unexpected_members)[:5])
        raise ValueError(
            f"{archive_path}: {len(unexpected_members)} audio members are absent from TSV; "
            f"first entries: {sample}"
        )

    for missing in sorted(expected_members - seen_members):
        row = by_filename[Path(missing).name]
        result = {key: row.get(key, "") for key in SOURCE_COLUMNS}
        result.update(
            {
                "source_split": split,
                "archive_member": missing,
                "audit_status": "rejected",
                "audit_reasons": "missing_audio_member",
            }
        )
        for key in MANIFEST_COLUMNS:
            result.setdefault(key, "")
        results.append(result)
    return results


def deterministic_subset(
    rows: Iterable[dict[str, str]], target_seconds: float, seed: int, split: str
) -> list[dict[str, str]]:
    """Select whole sentence groups in a stable pseudo-random order under a cap."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["audit_status"] != "rejected":
            groups[row["sentence_id"]].append(row)

    def group_key(sentence_id: str) -> str:
        return hashlib.sha256(f"{seed}:{split}:{sentence_id}".encode()).hexdigest()

    selected: list[dict[str, str]] = []
    total = 0.0
    for sentence_id in sorted(groups, key=group_key):
        group = groups[sentence_id]
        duration = sum(float(row["duration_seconds"]) for row in group)
        if total + duration <= target_seconds + 1e-9:
            selected.extend(group)
            total += duration
    return sorted(selected, key=lambda row: (int(row["sentence_id"]), row["filename"]))


def write_tsv(path: Path, rows: Iterable[dict[str, str]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_selected(
    input_dir: Path, output_root: Path, selected_by_split: dict[str, list[dict[str, str]]]
) -> None:
    """Extract selected raw clips and hard-link/copy them into cleaned audio."""
    for split, rows in selected_by_split.items():
        selected = {row["archive_member"]: row for row in rows}
        raw_dir = output_root / "raw" / "audio" / OUTPUT_SPLIT_NAMES[split]
        clean_dir = output_root / "cleaned" / "audio" / OUTPUT_SPLIT_NAMES[split]
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(input_dir / f"{split}.tar", "r:") as archive:
            for member in archive:
                if member.name not in selected:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot extract {member.name}")
                raw_path = raw_dir / Path(member.name).name
                raw_path.unlink(missing_ok=True)
                with raw_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
                expected = selected[member.name]["sha256"]
                if sha256_file(raw_path) != expected:
                    raise ValueError(f"checksum mismatch after extracting {member.name}")
                clean_path = clean_dir / raw_path.name
                clean_path.unlink(missing_ok=True)
                try:
                    os.link(raw_path, clean_path)
                except OSError:
                    shutil.copy2(raw_path, clean_path)


def build_summary(
    input_dir: Path,
    audits: dict[str, list[dict[str, str]]],
    selected: dict[str, list[dict[str, str]]],
    seed: int,
    targets: dict[str, float],
) -> dict[str, object]:
    source_files = {}
    for split in SPLITS:
        for suffix in ("tsv", "tar"):
            path = input_dir / f"{split}.{suffix}"
            source_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    split_summary = {}
    for split in SPLITS:
        rows = audits[split]
        chosen = selected[split]
        split_summary[OUTPUT_SPLIT_NAMES[split]] = {
            "source_split": split,
            "source_rows": len(rows),
            "accepted_rows": sum(row["audit_status"] != "rejected" for row in rows),
            "review_rows": sum(row["audit_status"] == "review" for row in rows),
            "rejected_rows": sum(row["audit_status"] == "rejected" for row in rows),
            "source_hours": round(sum(float(row["duration_seconds"] or 0) for row in rows) / 3600, 9),
            "selected_rows": len(chosen),
            "selected_sentence_ids": len({row["sentence_id"] for row in chosen}),
            "selected_hours": round(sum(float(row["duration_seconds"]) for row in chosen) / 3600, 9),
            "target_hours": targets[split] / 3600,
            "selected_gender_rows": dict(sorted(Counter(row["gender"] for row in chosen).items())),
        }
    all_rows = [row for rows in audits.values() for row in rows]
    all_selected = [row for rows in selected.values() for row in rows]
    overlap: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            left_rows = audits[left]
            right_rows = audits[right]
            overlap[f"{left}_vs_{right}"] = {
                "filenames": len({r["filename"] for r in left_rows} & {r["filename"] for r in right_rows}),
                "sentence_ids": len(
                    {r["sentence_id"] for r in left_rows} & {r["sentence_id"] for r in right_rows}
                ),
                "normalized_transcripts": len(
                    {r["transcription"] for r in left_rows} & {r["transcription"] for r in right_rows}
                ),
                "audio_sha256": len(
                    {r["sha256"] for r in left_rows if r["sha256"]}
                    & {r["sha256"] for r in right_rows if r["sha256"]}
                ),
            }

    return {
        "dataset": "Google FLEURS",
        "configuration": "sw_ke",
        "license": "CC BY 4.0",
        "selection_seed": seed,
        "selection_method": "official split preservation; deterministic whole-sentence-group sampling",
        "speaker_policy": (
            "Official train/dev/test boundaries are preserved. FLEURS documents these boundaries as "
            "speaker-disjoint; this TSV version does not expose speaker IDs, so speaker counts and an "
            "independent ID-level proof are unavailable."
        ),
        "cleaning_policy": (
            "Reject structural, metadata/WAV-format, sample-count, >30 s, nonfinite, or zero-energy "
            "failures. Keep waveform bytes unchanged; log high-silence/high-clipping clips for review."
        ),
        "source_files": source_files,
        "splits": split_summary,
        "totals": {
            "source_rows": len(all_rows),
            "source_hours": round(sum(float(row["duration_seconds"] or 0) for row in all_rows) / 3600, 9),
            "selected_rows": len(all_selected),
            "selected_hours": round(sum(float(row["duration_seconds"]) for row in all_selected) / 3600, 9),
            "rejected_rows": sum(row["audit_status"] == "rejected" for row in all_rows),
            "review_rows": sum(row["audit_status"] == "review" for row in all_rows),
        },
        "cross_split_overlap": overlap,
        "limitations": [
            "No listening-based intelligibility or language-ID review was performed.",
            "No transcript was corrected and no model-generated transcript was treated as ground truth.",
            "Silence/clipping thresholds are review flags, not automatic rejection rules.",
            "The first TSV field is a sentence ID, not a speaker ID.",
        ],
    }


def prepare(
    input_dir: Path,
    output_root: Path,
    *,
    seed: int = 42,
    target_hours: dict[str, float] | None = None,
    extract_audio: bool = False,
) -> dict[str, object]:
    target_hours = target_hours or {"train": 8.0, "dev": 1.0, "test": 1.0}
    targets = {split: target_hours[split] * 3600 for split in SPLITS}
    audits = {split: audit_split(input_dir, split) for split in SPLITS}
    selected = {
        split: deterministic_subset(audits[split], targets[split], seed, split) for split in SPLITS
    }

    audit_rows = [row for split in SPLITS for row in audits[split]]
    log_rows = [row for row in audit_rows if row["audit_status"] != "accepted"]
    selected_rows = [row for split in SPLITS for row in selected[split]]
    write_tsv(output_root / "audit" / "full_audit.tsv", audit_rows, MANIFEST_COLUMNS)
    write_tsv(output_root / "cleaned" / "cleaning_log.tsv", log_rows, MANIFEST_COLUMNS)
    write_tsv(output_root / "raw" / "manifest.tsv", selected_rows, SOURCE_COLUMNS + ("source_split",))
    write_tsv(output_root / "cleaned" / "manifest.tsv", selected_rows, MANIFEST_COLUMNS)
    for split in SPLITS:
        write_tsv(
            output_root / "splits" / f"{OUTPUT_SPLIT_NAMES[split]}.tsv",
            selected[split],
            MANIFEST_COLUMNS,
        )
    summary = build_summary(input_dir, audits, selected, seed, targets)
    (output_root / "audit").mkdir(parents=True, exist_ok=True)
    (output_root / "audit" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if extract_audio:
        extract_selected(input_dir, output_root, selected)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-hours", type=float, default=8.0)
    parser.add_argument("--validation-hours", type=float, default=1.0)
    parser.add_argument("--test-hours", type=float, default=1.0)
    parser.add_argument("--extract-audio", action="store_true")
    args = parser.parse_args()
    targets = {"train": args.train_hours, "dev": args.validation_hours, "test": args.test_hours}
    if any(value <= 0 for value in targets.values()):
        parser.error("all split hour targets must be positive")
    if sum(targets.values()) > 10.0:
        parser.error("requested split targets exceed the 10-hour project cap")
    summary = prepare(
        args.input_dir,
        args.output_root,
        seed=args.seed,
        target_hours=targets,
        extract_audio=args.extract_audio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
