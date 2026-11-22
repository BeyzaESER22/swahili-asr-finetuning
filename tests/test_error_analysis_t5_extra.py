"""T5: Swahili reduplication vs hallucination loops, and pairwise.csv on hand-made predictions."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from error_analysis import LOOP_MIN_REPEATS, analyse, longest_repeat_run

MANIFEST_COLS = ["sentence_id", "filename", "raw_transcription", "gender", "duration_seconds", "rms"]
REFERENCES = ["habari ya leo", "asante sana rafiki", "pole pole ndio mwendo"]


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


class ReduplicationIsNotALoopTests(unittest.TestCase):
    """Swahili doubles words on purpose (pole pole = slowly); that must not count as a hallucination."""

    def test_reduplicated_words_stay_below_loop_threshold(self):
        for sentence in ("pole pole ndio mwendo", "sawa sawa basi", "haraka haraka haina baraka",
                         "polepole tutafika"):
            self.assertLess(longest_repeat_run(sentence.split()), LOOP_MIN_REPEATS, sentence)

    def test_real_loop_still_detected(self):
        self.assertGreaterEqual(longest_repeat_run("gari la yv yv yv yv yv".split()), LOOP_MIN_REPEATS)


class PairwiseOnHandMadePredictionsTests(unittest.TestCase):
    """Three clips, three runs, outcomes chosen by hand so the expected pairwise.csv is known."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.manifest = root / "test.tsv"
        write_tsv(self.manifest, MANIFEST_COLS, [
            {"sentence_id": str(i), "filename": f"c{i}.wav", "raw_transcription": ref,
             "gender": "MALE", "duration_seconds": "5.0", "rms": str(0.1 * (i + 1))}
            for i, ref in enumerate(REFERENCES)])
        hyps = {
            # A: all clips have errors
            "A": ["habari za leo", "asante rafiki", "pole ndio"],
            # B: better than A on clip 0 and 1, identical on clip 2
            "B": ["habari ya leo", "asante sana rafiki", "pole ndio"],
            # C: worse than A on clip 0, better on clip 1, better on clip 2
            "C": ["xx yy zz ww", "asante sana rafiki", "pole pole ndio mwendo"],
        }
        self.paths = []
        for run, hs in hyps.items():
            path = root / f"{run}.tsv"
            write_tsv(path, ["id", "reference", "hypothesis"],
                      [{"id": f"c{i}.wav", "reference": REFERENCES[i], "hypothesis": h} for i, h in enumerate(hs)])
            self.paths.append(path)
        self.out = root / "out"
        analyse(self.paths, self.manifest, self.out, n_boot=50)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairwise_counts_match_hand_count(self):
        with (self.out / "pairwise.csv").open(encoding="utf-8") as f:
            rows = {r["run"]: r for r in csv.DictReader(f)}
        self.assertEqual(set(rows), {"B", "C"})
        for run, (better, worse, equal) in {"B": (2, 0, 1), "C": (2, 1, 0)}.items():
            r = rows[run]
            self.assertEqual(r["baseline_run"], "A")
            self.assertEqual((int(r["run_better"]), int(r["run_worse"]), int(r["equal"])), (better, worse, equal))

    def test_run_with_every_clip_above_wer_one_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            looping = Path(tmp) / "L.tsv"
            write_tsv(looping, ["id", "reference", "hypothesis"],
                      [{"id": f"c{i}.wav", "reference": r, "hypothesis": "yv " * 20} for i, r in enumerate(REFERENCES)])
            analyse([looping], self.manifest, Path(tmp) / "out", n_boot=50)
            with (Path(tmp) / "out" / "summary.csv").open(encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual((int(row["utts_wer_gt1"]), row["wer_excl_wer_gt1"]), (3, ""))

    def test_perfect_run_has_zero_insertion_share_and_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            perfect = Path(tmp) / "P.tsv"
            write_tsv(perfect, ["id", "reference", "hypothesis"],
                      [{"id": f"c{i}.wav", "reference": r, "hypothesis": r} for i, r in enumerate(REFERENCES)])
            analyse([perfect], self.manifest, Path(tmp) / "out", n_boot=50)
            with (Path(tmp) / "out" / "summary.csv").open(encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual((float(row["wer"]), float(row["ins_share"]), int(row["perfect_utts"])), (0.0, 0.0, 3))


if __name__ == "__main__":
    unittest.main()
