"""T5: tests for src/error_analysis.py (per-utterance errors, loops, subgroups, substitutions)."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from error_analysis import (
    analyse,
    duration_bucket,
    longest_repeat_run,
    pooled_wer,
    top_substitutions,
    utterance_errors,
)

REPO = Path(__file__).resolve().parents[1]


class UtteranceErrorTests(unittest.TestCase):
    def test_counts_substitution_deletion_insertion(self):
        e = utterance_errors("habari ya leo rafiki", "habari za leo")
        self.assertEqual((e["n_ref"], e["sub"], e["del"], e["ins"]), (4, 1, 1, 0))
        self.assertEqual(e["wer"], 0.5)

    def test_normalises_before_counting(self):
        e = utterance_errors("Habari, yako?", "habari yako")
        self.assertEqual(e["sub"] + e["del"] + e["ins"], 0)

    def test_blank_hypothesis_is_all_deletions(self):
        e = utterance_errors("asante sana", "")
        self.assertEqual((e["del"], e["hyp_words"]), (2, 0))

    def test_substitution_pairs_are_recorded(self):
        e = utterance_errors("habari ya leo", "habari za leo")
        self.assertEqual(e["pairs"], [("ya", "za")])


class LoopDetectionTests(unittest.TestCase):
    def test_single_word_loop(self):
        self.assertEqual(longest_repeat_run("watababu na na na na na".split()), 5)

    def test_bigram_loop(self):
        self.assertEqual(longest_repeat_run("na 400 ya 400 ya 400 ya 400 ya".split()), 4)

    def test_normal_sentence_has_no_repeat(self):
        self.assertEqual(longest_repeat_run("kwa ujumla tabia mbili zinaweza kuibuka".split()), 1)

    def test_empty(self):
        self.assertEqual(longest_repeat_run([]), 0)


class GroupingTests(unittest.TestCase):
    def test_duration_buckets(self):
        self.assertEqual([duration_bucket(x) for x in (4.5, 10.0, 14.9, 15.0, 28.4)],
                         ["<10s", "10-15s", "10-15s", "15-20s", ">=20s"])

    def test_pooled_wer_sums_errors_over_words(self):
        rows = [{"n_ref": 1, "sub": 0, "del": 0, "ins": 0}, {"n_ref": 4, "sub": 1, "del": 1, "ins": 0}]
        self.assertAlmostEqual(pooled_wer(rows), 0.4)

    def test_top_substitutions_counts_across_utterances(self):
        rows = [{"pairs": [("na", "la"), ("ya", "za")]}, {"pairs": [("na", "la")]}]
        self.assertEqual(top_substitutions(rows, 1), [(("na", "la"), 2)])


class AnalyseOnRealPredictionsTests(unittest.TestCase):
    """Uses the committed E1/E2 predictions and frozen test manifest."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.preds = [REPO / f"results/predictions/{run}.tsv" for run in ("E1", "E2", "E3")]
        analyse(cls.preds, REPO / "data/splits/test.tsv", cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, name):
        with (self.out / name).open(encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t" if name.endswith(".tsv") else ","))

    def test_pooled_wer_matches_committed_metrics(self):
        with (REPO / "results/metrics.csv").open(encoding="utf-8") as f:
            committed = {r["run"]: float(r["wer"]) for r in csv.DictReader(f)}
        summary = {r["run"]: float(r["wer"]) for r in self.read("summary.csv")}
        for run in ("E1", "E2", "E3"):
            self.assertAlmostEqual(summary[run], committed[run], places=4)

    def test_one_row_per_run_and_utterance(self):
        rows = self.read("utterances.tsv")
        self.assertEqual(len(rows), 3 * 252)

    def test_subgroups_cover_every_utterance(self):
        groups = [r for r in self.read("subgroups.csv") if r["run"] == "E1" and r["factor"] == "gender"]
        self.assertEqual(sum(int(r["n_utts"]) for r in groups), 252)

    def test_loops_found_in_e1_not_e2(self):
        summary = {r["run"]: r for r in self.read("summary.csv")}
        self.assertGreater(int(summary["E1"]["loop_utts"]), 0)
        self.assertEqual(int(summary["E2"]["loop_utts"]), 0)

    def test_writes_examples_and_substitutions(self):
        self.assertTrue((self.out / "examples.md").read_text(encoding="utf-8").startswith("#"))
        self.assertGreater(len(self.read("substitutions.tsv")), 0)

    def test_rejects_prediction_file_with_wrong_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "E9.tsv"
            bad.write_text("id\treference\thypothesis\nx.wav\ta\tb\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyse([bad], REPO / "data/splits/test.tsv", Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
