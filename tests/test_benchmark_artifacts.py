"""T5: audit of the committed benchmark artefacts (E1/E2/E3 predictions, run logs, config).

These tests read only committed files (no model, no audio) and check that the E3 fine-tuning
evidence is consistent with the frozen splits, the recorded configuration and the scorer.
"""
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_baseline import RUNS as RUN_REGISTRY, group_training
from score_predictions import read_tsv, score

REPO = Path(__file__).resolve().parents[1]
RUNS = ("E1", "E2", "E3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(rel: str) -> dict:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def filenames(rel: str) -> set[str]:
    return {r["filename"] for r in read_tsv(REPO / rel)}


class E3TrainingEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_json("results/runs/E3_train.json")
        config = load_json("configs/experiment.json")
        cls.e3_config = next(r for r in config["runs"] if r["id"] == "E3")

    def test_trained_on_frozen_train_and_validation_manifests(self):
        self.assertEqual(self.train["train_manifest_sha256"], sha256(REPO / "data/splits/train.tsv"))
        self.assertEqual(self.train["val_manifest_sha256"], sha256(REPO / "data/splits/validation.tsv"))

    def test_full_run_not_smoke_test(self):
        self.assertFalse(self.train["smoke_test"])
        self.assertIsNone(self.train["limit"])
        self.assertEqual((self.train["train_clips"], self.train["val_clips"]), (1902, 210))

    def test_base_checkpoint_is_the_e1_baseline_commit(self):
        e1 = load_json("results/runs/E1.json")
        self.assertEqual(self.train["base_model_id"], e1["model_id"])
        self.assertEqual(self.train["base_model_commit"], e1["model_commit"])

    def test_logged_hyperparameters_match_config(self):
        cfg = self.e3_config["training_config"]
        for key in ("learning_rate", "batch_size", "gradient_accumulation_steps", "epochs", "warmup_steps", "fp16", "seed"):
            with self.subTest(key=key):
                self.assertEqual(self.train[key], cfg[key])

    def test_test_split_disjoint_from_training_data(self):
        test = filenames("data/splits/test.tsv")
        self.assertFalse(test & filenames("data/splits/train.tsv"))
        self.assertFalse(test & filenames("data/splits/validation.tsv"))


class PredictionArtefactTests(unittest.TestCase):
    def test_every_run_scores_on_frozen_test_set(self):
        test_rows = read_tsv(REPO / "data/splits/test.tsv")
        for run in RUNS:
            with self.subTest(run=run):
                result = score(test_rows, read_tsv(REPO / f"results/predictions/{run}.tsv"), run)
                self.assertEqual(result["n_utterances"], 252)

    def test_run_logs_point_at_current_test_manifest(self):
        # E1 was run before a line-ending-only change of test.tsv; its references are still checked above.
        current = sha256(REPO / "data/splits/test.tsv")
        for run in ("E2", "E3"):
            with self.subTest(run=run):
                self.assertEqual(load_json(f"results/runs/{run}.json")["manifest_sha256"], current)

    def test_run_registry_derives_group_training_from_role(self):
        # run_baseline.py writes this flag into results/runs/<RUN>.json; it must not be hard-coded
        self.assertEqual({run: group_training(cfg) for run, cfg in RUN_REGISTRY.items()}, {"E1": False, "E2": False, "E3": True})

    def test_only_e3_is_marked_as_group_training(self):
        flags = {run: load_json(f"results/runs/{run}.json")["group_training"] for run in RUNS}
        self.assertEqual(flags, {"E1": False, "E2": False, "E3": True})

    def test_committed_metrics_match_rescoring(self):
        test_rows = read_tsv(REPO / "data/splits/test.tsv")
        with (REPO / "results/metrics.csv").open(encoding="utf-8") as f:
            committed = {r["run"]: r for r in csv.DictReader(f)}
        for run in committed:
            fresh = score(test_rows, read_tsv(REPO / f"results/predictions/{run}.tsv"), run)
            with self.subTest(run=run):
                self.assertAlmostEqual(float(committed[run]["wer"]), fresh["wer"], places=4)
                self.assertAlmostEqual(float(committed[run]["cer"]), fresh["cer"], places=4)

    def test_metrics_table_contains_all_three_runs(self):
        with (REPO / "results/metrics.csv").open(encoding="utf-8") as f:
            self.assertEqual([r["run"] for r in csv.DictReader(f)], list(RUNS))


if __name__ == "__main__":
    unittest.main()
