"""T5 extended analyses: sentence-clustered bootstrap, train-vocabulary (OOV) errors,
word-boundary (space) errors and Swahili-oriented substitution categories."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from error_analysis import (
    analyse,
    bootstrap_ci,
    classify_substitution,
    oov_error_rates,
    paired_bootstrap,
    space_edits,
    utterance_errors,
)

REPO = Path(__file__).resolve().parents[1]


def row(sentence_id, n_ref, errors):
    return {"sentence_id": sentence_id, "n_ref": n_ref, "sub": errors, "del": 0, "ins": 0}


class BootstrapTests(unittest.TestCase):
    ROWS = [row("s1", 10, 1), row("s1", 10, 3), row("s2", 5, 0), row("s3", 8, 4), row("s4", 12, 2)]

    def test_ci_brackets_point_estimate_and_is_reproducible(self):
        low, high = bootstrap_ci(self.ROWS, n_boot=500, seed=42)
        point = 10 / 45
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)
        self.assertEqual((low, high), bootstrap_ci(self.ROWS, n_boot=500, seed=42))

    def test_constant_error_rate_has_zero_width_ci(self):
        rows = [row(f"s{i}", 10, 2) for i in range(5)]
        self.assertEqual(bootstrap_ci(rows, n_boot=200, seed=1), (0.2, 0.2))

    def test_paired_identical_runs_have_no_difference(self):
        res = paired_bootstrap(self.ROWS, self.ROWS, n_boot=200, seed=1)
        self.assertEqual((res["delta"], res["ci_low"], res["ci_high"]), (0.0, 0.0, 0.0))
        self.assertEqual(res["p_value"], 1.0)

    def test_paired_uniformly_better_run_is_significant(self):
        better = [{**r, "sub": 0} for r in self.ROWS]
        res = paired_bootstrap(self.ROWS, better, n_boot=500, seed=1)
        self.assertLess(res["delta"], 0)
        self.assertLess(res["ci_high"], 0)
        self.assertLess(res["p_value"], 0.01)

    def test_paired_requires_same_utterance_order(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.ROWS, self.ROWS[::-1], n_boot=10, seed=1)


class OovTests(unittest.TestCase):
    def test_reference_word_outcomes_are_recorded(self):
        e = utterance_errors("habari ya leo rafiki", "habari za leo")
        self.assertEqual(e["ref_ops"], [("habari", "equal"), ("ya", "substitute"), ("leo", "equal"), ("rafiki", "delete")])

    def test_error_rate_split_by_train_vocabulary(self):
        rows = [{"ref_ops": [("habari", "equal"), ("ya", "substitute"), ("goethe", "delete"), ("leo", "equal")]}]
        rates = oov_error_rates(rows, train_vocab={"habari", "ya", "leo"})
        self.assertEqual(rates["in_vocab"], {"ref_words": 3, "errors": 1})
        self.assertEqual(rates["oov"], {"ref_words": 1, "errors": 1})


class WordBoundaryTests(unittest.TestCase):
    def test_merged_words_is_one_space_edit(self):
        self.assertEqual(space_edits("kwa ujumla", "kwaujumla"), (1, 1))

    def test_split_word_is_space_edit(self):
        self.assertEqual(space_edits("kipengele", "kipe ngele"), (1, 1))

    def test_letter_error_is_not_space_edit(self):
        self.assertEqual(space_edits("paka", "baka"), (0, 1))

    def test_normalises_first(self):
        self.assertEqual(space_edits("Kwa, ujumla!", "kwa ujumla"), (0, 0))


class SubstitutionCategoryTests(unittest.TestCase):
    def test_short_function_words(self):
        self.assertEqual(classify_substitution("na", "la"), "function_word")

    def test_change_at_word_start(self):
        # noun-class / subject-prefix position
        self.assertEqual(classify_substitution("hukubali", "ukubali"), "start")

    def test_change_at_word_end(self):
        self.assertEqual(classify_substitution("kitamaduni", "kitamadumu"), "end")

    def test_change_inside_word(self):
        self.assertEqual(classify_substitution("dhoruba", "dharuba"), "middle")

    def test_unrelated_words_are_other_not_prefix_errors(self):
        # nothing shared at either edge -> not a prefix error
        self.assertEqual(classify_substitution("watoto", "kitabu"), "other")


class ExtendedOutputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        preds = [REPO / f"results/predictions/{run}.tsv" for run in ("E1", "E2", "E3")]
        analyse(preds, REPO / "data/splits/test.tsv", cls.out,
                train_manifest=REPO / "data/splits/train.tsv", n_boot=200)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, name):
        with (self.out / name).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_bootstrap_table_has_ci_per_run_and_pairs(self):
        rows = self.read("bootstrap.csv")
        runs = {r["run"]: r for r in rows if r["kind"] == "run"}
        self.assertEqual(set(runs), {"E1", "E2", "E3"})
        for r in runs.values():
            self.assertLessEqual(float(r["ci_low"]), float(r["wer"]))
            self.assertGreaterEqual(float(r["ci_high"]), float(r["wer"]))
        pairs = {(r["baseline_run"], r["run"]) for r in rows if r["kind"] == "pair"}
        self.assertEqual(pairs, {("E1", "E2"), ("E1", "E3"), ("E2", "E3")})

    def test_oov_table_has_both_statuses_per_run(self):
        rows = self.read("oov.csv")
        self.assertEqual({(r["run"], r["vocab_status"]) for r in rows},
                         {(run, s) for run in ("E1", "E2", "E3") for s in ("in_vocab", "oov")})
        totals = {r["run"]: 0 for r in rows}
        for r in rows:
            totals[r["run"]] += int(r["ref_words"])
        self.assertEqual(set(totals.values()), {5152})  # every reference word counted once

    def test_word_boundary_and_substitution_tables(self):
        wb = {r["run"]: r for r in self.read("word_boundary.csv")}
        for r in wb.values():
            self.assertLessEqual(float(r["cer_no_space"]), 1.0)
            self.assertLessEqual(float(r["space_edit_share"]), 1.0)
        cats = {(r["run"], r["category"]) for r in self.read("substitution_categories.csv")}
        self.assertIn(("E3", "function_word"), cats)


if __name__ == "__main__":
    unittest.main()
