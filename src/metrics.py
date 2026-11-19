"""Reusable WER/CER; use identical normalization on reference and hypothesis."""
from __future__ import annotations

import re
import unicodedata
from jiwer import wer, cer


def normalize_transcript(text: str) -> str:
    """Unicode NFC + lower case + punctuation -> space + whitespace collapse.

    Punctuation (Unicode category P*) is removed because Whisper emits it and the
    FLEURS `raw_transcription` reference contains it; neither is spoken. The
    apostrophe is kept because it is part of Swahili spelling (e.g. ng'ombe).
    Digits are left as written: "2008" vs "elfu mbili na nane" still counts as an error.
    """
    value = unicodedata.normalize("NFC", text).lower()
    value = "".join(" " if unicodedata.category(c).startswith("P") and c not in "'’" else c for c in value)
    value = value.replace("’", "'")
    return re.sub(r"\s+", " ", value).strip()


def evaluate_asr(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("Reference/prediction count mismatch")
    if not references:
        raise ValueError("Cannot score empty ASR predictions")
    refs = [normalize_transcript(s) for s in references]
    hyps = [normalize_transcript(s) for s in hypotheses]
    if any(not ref for ref in refs):
        raise ValueError("Blank normalized reference")
    return {"wer": float(wer(refs, hyps)), "cer": float(cer(refs, hyps))}
