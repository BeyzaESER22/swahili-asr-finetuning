"""T3 - Vanilla pretrained ASR inference on the frozen test split (no training).

E1  main course baseline      openai/whisper-tiny          (encoder-decoder, autoregressive)
E2  additional comparator     facebook/mms-1b-all + "swh"  (wav2vec2 + CTC head, Swahili adapter)

Both read the same frozen FLEURS `data/splits/test.tsv` (252 clips, audio in
`data/cleaned/audio/test/`, fetched with `git lfs pull`), use `raw_transcription`
as the human reference, feed 16 kHz mono audio to the model and write
`results/predictions/<RUN>.tsv` (id, reference, hypothesis) plus a provenance file
`results/runs/<RUN>.json` (exact model commit, versions, device, timing, input hash).
WER/CER are computed separately by `src/score_predictions.py` so every model is
scored by the same code.

Usage:
python src/run_baseline.py --run E1
python src/run_baseline.py --run E2
# quick smoke test on 8 clips:
python src/run_baseline.py --run E1 --limit 8 --output results/smoke/E1.tsv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_RATE = 16_000
ID_COL = "filename"            # FLEURS clip file name, unique per recording
REF_COL = "raw_transcription"  # human reference text (normalised identically to hypotheses at scoring time)
LFS_PREFIX = b"version https://git-lfs"

RUNS = {
    "E1": {
        "family": "whisper",
        "role": "main_baseline",
        "model_id": "openai/whisper-tiny",
        "language": "sw",          # Whisper language token
    },
    "E2": {
        "family": "ctc",
        "role": "additional_pretrained_comparator",
        "model_id": "facebook/mms-1b-all",
        "target_lang": "swh",      # MMS adapter code for Swahili (ISO 639-3)
    },
}


# ------------------------------------------------------------- data helpers
def read_manifest(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for col in (ID_COL, REF_COL):
        if col not in rows[0]:
            raise ValueError(f"Manifest is missing column {col!r}")
    ids = [r[ID_COL] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip ids in manifest")
    return rows[:limit] if limit else rows


def write_predictions(path: Path, rows: list[dict], hypotheses: list[str]) -> None:
    if len(rows) != len(hypotheses):
        raise ValueError("Prediction count does not match manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "reference", "hypothesis"])
        for row, hyp in zip(rows, hypotheses):
            clean = " ".join(hyp.split())  # no tabs/newlines inside a cell
            writer.writerow([row[ID_COL], row[REF_COL], clean])


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_audio(path: Path):
    """Read a WAV as float32 mono 16 kHz (FLEURS files already are; resample otherwise)."""
    import soundfile

    with path.open("rb") as f:
        if f.read(len(LFS_PREFIX)) == LFS_PREFIX:
            raise RuntimeError(f"{path} is a Git LFS pointer, not audio. Run `git lfs pull` first.")
    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return audio


def resolve_revision(model_id: str, revision: str) -> str:
    """Turn a branch/tag (e.g. 'main') into the immutable commit hash we actually load."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception as exc:  # offline: keep what was requested, but say so
        print(f"WARNING: could not resolve revision ({exc}); using {revision!r} unpinned")
        return revision


# ------------------------------------------------------------- model wrappers
class WhisperASR:
    def __init__(self, model_id, revision, device, language="sw", num_beams=1, max_new_tokens=128):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, revision=revision).to(device).eval()
        self.device, self.language = device, language
        self.gen = {"num_beams": num_beams, "max_new_tokens": max_new_tokens}

    def transcribe(self, audios):
        feats = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(self.device)
        with self.torch.inference_mode():
            ids = self.model.generate(feats, language=self.language, task="transcribe", **self.gen)
        return self.processor.batch_decode(ids, skip_special_tokens=True)


class CTCASR:
    def __init__(self, model_id, revision, device, target_lang=None):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.torch = torch
        kw = {"target_lang": target_lang, "ignore_mismatched_sizes": True} if target_lang else {}
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, **({"target_lang": target_lang} if target_lang else {}))
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id, revision=revision, **kw).to(device).eval()
        self.device = device

    def transcribe(self, audios):
        inputs = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        return self.processor.batch_decode(logits.argmax(dim=-1))  # greedy CTC decoding


def build_model(cfg: dict, model_id: str, revision: str, device: str, num_beams: int):
    if cfg["family"] == "whisper":
        return WhisperASR(model_id, revision, device, cfg.get("language", "sw"), num_beams)
    return CTCASR(model_id, revision, device, cfg.get("target_lang"))


# ------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=sorted(RUNS), required=True)
    p.add_argument("--manifest", type=Path, default=Path("data/splits/test.tsv"))
    p.add_argument("--audio-root", type=Path, default=Path("data/cleaned/audio/test"))
    p.add_argument("--model-id", help="override the default checkpoint for this run")
    p.add_argument("--revision", default="main", help="branch/tag/commit; resolved to a commit hash")
    p.add_argument("--output", type=Path, help="default: results/predictions/<RUN>.tsv")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=1, help="Whisper only; 1 = greedy")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--limit", type=int, help="only the first N clips (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    import transformers

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RUNS[args.run]
    model_id = args.model_id or cfg["model_id"]
    commit = resolve_revision(model_id, args.revision)
    output = args.output or Path("results/predictions") / f"{args.run}.tsv"

    rows = read_manifest(args.manifest, args.limit)
    print(f"{args.run}: {model_id}@{commit[:10]} on {device}, {len(rows)} clips")
    model = build_model(cfg, model_id, commit, device, args.num_beams)

    hyps, audio_seconds, start = [], 0.0, time.perf_counter()
    for i, batch in enumerate(batched(rows, args.batch_size)):
        audios = [load_audio(args.audio_root / r[ID_COL]) for r in batch]
        audio_seconds += sum(len(a) for a in audios) / SAMPLE_RATE
        hyps.extend(model.transcribe(audios))
        print(f"  {min((i + 1) * args.batch_size, len(rows))}/{len(rows)}", end="\r")
    elapsed = time.perf_counter() - start
    write_predictions(output, rows, hyps)

    meta = {
        "run": args.run,
        "role": cfg["role"],
        "group_training": False,
        "model_id": model_id,
        "requested_revision": args.revision,
        "model_commit": commit,
        "decoding": {"whisper": f"generate, language={cfg.get('language')}, task=transcribe, num_beams={args.num_beams}",
                     "ctc": f"greedy argmax CTC, target_lang={cfg.get('target_lang')}"}[cfg["family"]],
        "manifest": str(args.manifest).replace("\\", "/"),
        "manifest_sha256": sha256_file(args.manifest),
        "num_clips": len(rows),
        "limit": args.limit,
        "audio_hours": round(audio_seconds / 3600, 4),
        "inference_seconds": round(elapsed, 2),
        "real_time_factor": round(elapsed / audio_seconds, 4) if audio_seconds else None,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
        "repo_commit": git_commit(),
        "predictions": str(output).replace("\\", "/"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = output.parent.parent / "runs" / f"{output.stem}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {output} and {meta_path} ({elapsed:.1f}s, RTF {meta['real_time_factor']})")


if __name__ == "__main__":
    main()
