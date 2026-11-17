import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_fleurs import deterministic_subset, inspect_float_wav, read_source_tsv


def float_wav(samples: np.ndarray, rate: int = 16000) -> bytes:
    payload = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


class FleursParserTests(unittest.TestCase):
    def test_literal_tsv_does_not_apply_quote_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.tsv"
            path.write_text('1\ta.wav\tA "quote\ta quote\ta | quote |\t16000\tFEMALE\n', encoding="utf-8")
            rows = read_source_tsv(path, "train")
            self.assertEqual(rows[0]["filename"], "a.wav")
            self.assertEqual(rows[0]["transcription"], "a quote")

    def test_rejects_wrong_field_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.tsv"
            path.write_text("only\ttwo\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 7"):
                read_source_tsv(path, "train")

    def test_float_wav_statistics(self):
        samples = np.array([0.0, 0.5, -1.0, 0.0], dtype=np.float32)
        stats = inspect_float_wav(io.BytesIO(float_wav(samples)))
        self.assertEqual(stats.audio_format, 3)
        self.assertEqual(stats.num_frames, 4)
        self.assertEqual(stats.sample_rate_hz, 16000)
        self.assertAlmostEqual(stats.peak_abs, 1.0)
        self.assertAlmostEqual(stats.silence_ratio, 0.5)
        self.assertAlmostEqual(stats.clipping_ratio, 0.25)

    def test_selection_is_deterministic_and_keeps_sentence_groups(self):
        rows = []
        for sentence_id, durations in {"1": [2.0, 3.0], "2": [4.0], "3": [1.0]}.items():
            for index, duration in enumerate(durations):
                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "filename": f"{sentence_id}-{index}.wav",
                        "duration_seconds": str(duration),
                        "audit_status": "accepted",
                    }
                )
        first = deterministic_subset(rows, 6.0, 42, "train")
        second = deterministic_subset(rows, 6.0, 42, "train")
        self.assertEqual(first, second)
        picked = {row["sentence_id"] for row in first}
        for sentence_id in picked:
            self.assertEqual(
                sum(row["sentence_id"] == sentence_id for row in first),
                sum(row["sentence_id"] == sentence_id for row in rows),
            )
        self.assertLessEqual(sum(float(row["duration_seconds"]) for row in first), 6.0)


if __name__ == "__main__":
    unittest.main()
