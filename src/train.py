"""T4 - Fine-tune Whisper-Tiny on the frozen Swahili FLEURS training split.

E3  trained group contribution   openai/whisper-tiny (fine-tuned)

Protocol rules (enforced by code):
- Train ONLY on `data/splits/train.tsv` (1,902 clips, ~8.0 h).
- Validate ONLY on `data/splits/validation.tsv` (210 clips, ~0.8 h) for model/checkpoint selection.
- NEVER access or evaluate on `data/splits/test.tsv` during training or tuning.
- Base checkpoint revision is pinned to commit `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` (same as E1).

Usage:
  # Quick 2-sample smoke test on CPU or GPU (finishes in seconds):
  python src/train.py --smoke-test

  # Full training on Google Colab T4 GPU:
  python src/train.py --fp16 --epochs 3 --batch-size 8 --grad-accum 2

  # Custom paths and learning rate:
  python src/train.py --learning-rate 1e-4 --output-dir results/checkpoints/E3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
ID_COL = "filename"
REF_COL = "raw_transcription"
LFS_PREFIX = b"version https://git-lfs"

# Pinned base checkpoint and revision from E1
DEFAULT_MODEL_ID = "openai/whisper-tiny"
DEFAULT_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


# ------------------------------------------------------------- manifest & audio
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def read_manifest(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for col in (ID_COL, REF_COL):
        if col not in rows[0]:
            raise ValueError(f"Manifest missing required column {col!r}")
    return rows[:limit] if limit else rows


def resolve_audio_path(audio_root: Path, filename: str) -> Path:
    """Find audio file under audio_root or common subdirectories."""
    direct = audio_root / filename
    if direct.is_file():
        return direct
    # Try finding in split subfolders (e.g. audio_root/train/filename)
    matches = list(audio_root.glob(f"*/{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Audio file not found: {filename} under {audio_root}")


def load_audio(path: Path):
    """Load WAV as float32 mono 16 kHz."""
    with path.open("rb") as f:
        if f.read(len(LFS_PREFIX)) == LFS_PREFIX:
            raise RuntimeError(
                f"{path} is a Git LFS pointer, not real audio. Run `git lfs pull` first."
            )

    try:
        import soundfile
        audio, rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    except ImportError:
        import numpy as np
        from scipy.io import wavfile
        rate, audio = wavfile.read(str(path))
        if audio.dtype != np.float32:
            audio = audio.astype("float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return audio


# ------------------------------------------------------------- dataset & collator
class FleursDataset:
    """Lightweight PyTorch Dataset reading FLEURS manifests directly."""

    def __init__(self, rows: list[dict], audio_root: Path, processor: Any):
        self.rows = rows
        self.audio_root = audio_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        wav_path = resolve_audio_path(self.audio_root, row[ID_COL])
        audio = load_audio(wav_path)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE
        ).input_features[0]

        labels = self.processor.tokenizer(row[REF_COL]).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If BOS token was prepended, truncate if necessary
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ------------------------------------------------------------- metrics
def build_compute_metrics(processor: Any):
    import sys

    # Add src to path to import metrics.py
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from metrics import evaluate_asr

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        res = evaluate_asr(label_str, pred_str)
        return {"wer": res["wer"], "cer": res["cer"]}

    return compute_metrics


def configure_model_for_whisper_finetuning(
    model: Any, language: str = "sw", task: str = "transcribe"
) -> None:
    """Configure Whisper model and generation settings for fine-tuning and evaluation."""
    # Disable forced decoder IDs during training so model can learn tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Explicitly pin generation_config for Swahili transcription during evaluation/generation
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.language = language
        model.generation_config.task = task
        model.generation_config.forced_decoder_ids = None


# ------------------------------------------------------------- main training logic
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Manifests and audio paths (train and validation only, NO test)
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/splits/train.tsv"),
        help="Path to training manifest",
    )
    p.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/splits/validation.tsv"),
        help="Path to validation manifest",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/cleaned/audio"),
        help="Root folder containing train/ and validation/ audio",
    )

    # Base model and revision
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID (default: openai/whisper-tiny)",
    )
    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned Git commit SHA of base model (default: E1 baseline commit)",
    )
    p.add_argument(
        "--language",
        default="sw",
        help="Language code (sw = Swahili)",
    )

    # Training hyperparameters
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/checkpoints/E3"),
        help="Directory to save checkpoints",
    )
    p.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of training epochs (default: 3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train and eval batch size (default: 8)",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2, effective batch size 16)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (default: 1e-4)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps (default: 50)",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 half-precision training (recommended on GPU)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Smoke test option
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick 2-sample trial on train and validation (finishes in seconds)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Explicitly limit number of samples for testing",
    )
    args = p.parse_args()

    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    # Set seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine limits
    limit = 2 if args.smoke_test else args.limit
    if args.smoke_test:
        print("=== SMOKE TEST MODE (2 samples) ===")
        args.output_dir = Path("results/checkpoints/E3_smoke")
        args.epochs = 1.0

    # Ensure train and validation manifests exist
    if not args.train_manifest.is_file():
        raise FileNotFoundError(f"Train manifest not found: {args.train_manifest}")
    if not args.val_manifest.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {args.val_manifest}")

    train_rows = read_manifest(args.train_manifest, limit=limit)
    val_rows = read_manifest(args.val_manifest, limit=limit)
    print(f"Loaded {len(train_rows)} train clips and {len(val_rows)} validation clips")

    # Load processor and model
    print(f"Loading processor and model {args.model_id} (revision: {args.revision[:10]})...")
    processor = WhisperProcessor.from_pretrained(
        args.model_id, revision=args.revision, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.revision
    )

    # Whisper fine-tuning configurations:
    configure_model_for_whisper_finetuning(model, language=args.language, task="transcribe")

    # Datasets and Collator
    train_dataset = FleursDataset(train_rows, args.audio_root, processor)
    val_dataset = FleursDataset(val_rows, args.audio_root, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = build_compute_metrics(processor=processor)

    # Device & precision setup
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = args.fp16 and (device_type == "cuda")
    print(f"Training on device: {device_type} (FP16={use_fp16})")

    # Training arguments
    eval_strategy = "steps" if args.smoke_test else "epoch"
    save_strategy = "steps" if args.smoke_test else "epoch"
    eval_steps = 1 if args.smoke_test else None
    save_steps = 1 if args.smoke_test else None
    logging_steps = 1 if args.smoke_test else 25

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=2 if args.smoke_test else -1,
        gradient_checkpointing=True,
        fp16=use_fp16,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        predict_with_generate=True,
        generation_max_length=128,
        save_total_limit=2,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=["none"],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    start_time = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start_time
    print(f"Training finished in {elapsed:.1f}s")

    # Save best model and processor
    best_dir = args.output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f"Saved best model checkpoint and processor to: {best_dir}")

    # Final evaluation on validation set
    print("Evaluating best model on validation split...")
    eval_metrics = trainer.evaluate()
    print(
        f"Validation WER: {eval_metrics.get('eval_wer', 'N/A')} | CER: {eval_metrics.get('eval_cer', 'N/A')}"
    )

    # Save provenance metadata
    summary = {
        "run": "E3",
        "role": "trained_contribution",
        "base_model_id": args.model_id,
        "base_model_commit": args.revision,
        "language": args.language,
        "train_manifest": str(args.train_manifest).replace("\\", "/"),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "val_manifest": str(args.val_manifest).replace("\\", "/"),
        "val_manifest_sha256": sha256_file(args.val_manifest),
        "train_clips": len(train_rows),
        "val_clips": len(val_rows),
        "limit": limit,
        "smoke_test": args.smoke_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "device": device_type,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fp16": use_fp16,
        "training_seconds": round(elapsed, 2),
        "train_loss": train_result.training_loss,
        "val_loss": eval_metrics.get("eval_loss"),
        "val_wer": eval_metrics.get("eval_wer"),
        "val_cer": eval_metrics.get("eval_cer"),
        "checkpoint_path": str(best_dir).replace("\\", "/"),
        "seed": args.seed,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "repo_commit": git_commit(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    meta_path = Path("results/runs/E3_train.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote training metadata to: {meta_path}")


if __name__ == "__main__":
    main()
