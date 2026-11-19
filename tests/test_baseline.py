"""T3 tests that need no model download: manifest/prediction I/O, audio guard, scoring."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import evaluate_asr, normalize_transcript
from run_baseline import ID_COL, REF_COL, RUNS, batched, load_audio, read_manifest, write_predictions
from score_predictions import read_tsv, score

REPO = Path(__file__).resolve().parents[1]
TEST = [{"filename": "a.wav", "raw_transcription": "Habari yako?", "transcription": "habari yako"},
        {"filename": "b.wav", "raw_transcription": "Asante sana.", "transcription": "asante sana"}]


def write_manifest(path: Path, rows):
    cols = list(rows[0])
    lines = ["\t".join(cols)] + ["\t".join(r[c] for c in cols) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class BaselineIOTests(unittest.TestCase):
    def test_run_registry(self):
        self.assertEqual(RUNS["E1"]["model_id"], "openai/whisper-tiny")
        self.assertEqual(RUNS["E1"]["role"], "main_baseline")
        self.assertEqual(RUNS["E2"]["family"], "ctc")
        self.assertEqual((ID_COL, REF_COL), ("filename", "raw_transcription"))

    def test_manifest_limit_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = Path(tmp) / "test.tsv"
            write_manifest(m, TEST)
            self.assertEqual(len(read_manifest(m, limit=1)), 1)
            write_manifest(m, TEST + TEST[:1])
            with self.assertRaises(ValueError):
                read_manifest(m)

    def test_real_frozen_test_split_is_readable(self):
        rows = read_manifest(REPO / "data" / "splits" / "test.tsv")
        self.assertEqual(len(rows), 252)
        self.assertTrue(all(r[REF_COL].strip() for r in rows))

    def test_lfs_pointer_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ptr = Path(tmp) / "x.wav"
            ptr.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
            with self.assertRaisesRegex(RuntimeError, "git lfs pull"):
                load_audio(ptr)

    def test_batched(self):
        self.assertEqual(list(batched([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_predictions_roundtrip_and_perfect_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predictions" / "E1.tsv"
            write_predictions(pred, TEST, ["habari\tyako", "Asante  sana!\n"])
            rows = read_tsv(pred)
            self.assertEqual(rows[0]["hypothesis"], "habari yako")
            result = score(TEST, rows, "E1")
            self.assertEqual((result["wer"], result["cer"], result["n_utterances"]), (0.0, 0.0, 2))

    def test_score_refuses_mismatched_test_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "E2.tsv"
            write_predictions(pred, TEST[:1], ["habari"])
            with self.assertRaises(ValueError):  # missing a test id
                score(TEST, read_tsv(pred), "E2")
            write_predictions(pred, [{**TEST[0], REF_COL: "edited"}, TEST[1]], ["a", "b"])
            with self.assertRaises(ValueError):  # reference text was altered
                score(TEST, read_tsv(pred), "E2")


class NormalizationTests(unittest.TestCase):
    def test_punctuation_removed_apostrophe_kept(self):
        self.assertEqual(normalize_transcript('"Ng’ombe, wa Bw. Costello!"'), "ng'ombe wa bw costello")

    def test_punctuation_only_difference_is_not_an_error(self):
        self.assertEqual(evaluate_asr(["Uroma ulikuwa na kipengele."], ["uroma ulikuwa na kipengele"])["wer"], 0.0)

    def test_hand_checked_wer(self):
        # ref 4 words; hyp: 1 substitution + 1 deletion -> WER 2/4
        self.assertEqual(evaluate_asr(["habari ya leo rafiki"], ["habari za leo"])["wer"], 0.5)


if __name__ == "__main__":
    unittest.main()
