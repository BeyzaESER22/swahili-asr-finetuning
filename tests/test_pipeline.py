"""Tests for initial metadata pipeline (not a model training test suite)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_manifest import audit_rows
from split_manifest import split_rows
from metrics import evaluate_asr, normalize_transcript


class ManifestTests(unittest.TestCase):
    def test_audit_logs_missing_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp3").write_bytes(b"dummy")
            rows = [
                {"path": "a.mp3", "sentence": "Habari", "client_id": "one"},
                {"path": "a.mp3", "sentence": "Habari", "client_id": "one"},
                {"path": "b.mp3", "sentence": "Habari", "client_id": "one"},
                {"path": "../escape.mp3", "sentence": "Hi", "client_id": "two"},
                {"path": "z.mp3", "sentence": "", "client_id": ""},
            ]
            good, bad = audit_rows(rows, root)
            self.assertEqual(len(good), 1)
            self.assertEqual(len(bad), 4)
            self.assertIn("duplicate_clip_path", bad[0]["reason"])

    def test_speaker_split(self):
        rows = [
            {"path": f"{i}.mp3", "sentence": "habari", "client_id": f"speaker{i // 4}"}
            for i in range(80)
        ]
        a = split_rows(rows)
        b = split_rows(rows)
        self.assertEqual(a, b)
        groups = [{r["client_id"] for r in v} for v in a.values()]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])

    def test_reject_missing_speaker(self):
        with self.assertRaises(ValueError):
            split_rows([{"client_id": ""}] * 5)


class MetricTests(unittest.TestCase):
    def test_perfect_score(self):
        self.assertEqual(evaluate_asr(["Habari  yako"], ["habari yako"]), {"wer": 0.0, "cer": 0.0})

    def test_error_and_length_check(self):
        self.assertGreater(evaluate_asr(["habari yako"], ["habari"] )["wer"], 0)
        with self.assertRaises(ValueError):
            evaluate_asr(["a"], [])

    def test_normalization(self):
        self.assertEqual(normalize_transcript("  HABARI\nYAKO "), "habari yako")


if __name__ == "__main__":
    unittest.main()
