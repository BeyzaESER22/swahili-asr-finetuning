"""T5: hand-checked WER/CER cases for the shared scorer in src/metrics.py.

WER = (S + D + I) / N_reference_words, CER = character edits / N_reference_chars,
both computed on normalised text (see metrics.normalize_transcript).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import evaluate_asr, normalize_transcript


class WerCerHandCheckedTests(unittest.TestCase):
    def test_perfect_match_is_zero(self):
        self.assertEqual(evaluate_asr(["habari ya leo"], ["habari ya leo"]), {"wer": 0.0, "cer": 0.0})

    def test_single_substitution(self):
        # 1 substituted word of 3
        self.assertAlmostEqual(evaluate_asr(["habari ya leo"], ["habari za leo"])["wer"], 1 / 3)

    def test_pure_deletion(self):
        # 2 of 4 reference words missing
        self.assertEqual(evaluate_asr(["habari ya leo rafiki"], ["habari leo"])["wer"], 0.5)

    def test_pure_insertion_can_exceed_one(self):
        # 2 ref words, 3 inserted words -> WER 3/2; this is how hallucination loops push WER > 1
        self.assertEqual(evaluate_asr(["asante sana"], ["asante sana na na na"])["wer"], 1.5)

    def test_blank_hypothesis_is_all_deletions(self):
        # an empty model output must score as every reference word deleted, not crash
        self.assertEqual(evaluate_asr(["asante sana"], [""]), {"wer": 1.0, "cer": 1.0})

    def test_punctuation_only_hypothesis_is_blank(self):
        self.assertEqual(evaluate_asr(["asante sana"], ["..."])["wer"], 1.0)

    def test_corpus_wer_pools_words_not_utterance_average(self):
        # utt1: 0/1 errors, utt2: 2/4 errors -> pooled 2/5 = 0.4 (utterance mean would be 0.25)
        result = evaluate_asr(["ndiyo", "habari ya leo rafiki"], ["ndiyo", "habari leo"])
        self.assertAlmostEqual(result["wer"], 0.4)

    def test_cer_counts_characters(self):
        # "paka" -> "baka": 1 char edit of 4
        self.assertEqual(evaluate_asr(["paka"], ["baka"])["cer"], 0.25)

    def test_blank_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_asr(["  ,. "], ["habari"])

    def test_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_asr(["habari", "asante"], ["habari"])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_asr([], [])


class NormalisationPolicyTests(unittest.TestCase):
    def test_case_and_whitespace(self):
        self.assertEqual(normalize_transcript("  Habari   YAKO \n"), "habari yako")

    def test_nfc_composed_and_decomposed_are_equal(self):
        composed, decomposed = "Goethé", "Goethé"  # é as one code point vs e + combining acute
        self.assertNotEqual(composed, decomposed)
        self.assertEqual(normalize_transcript(composed), normalize_transcript(decomposed))

    def test_digits_are_not_verbalised(self):
        # documented policy: "1967" vs spoken form is still an error
        self.assertEqual(normalize_transcript("Vita vya 1967."), "vita vya 1967")
        self.assertGreater(evaluate_asr(["vita vya 1967"], ["vita vya elfu moja"])["wer"], 0)


if __name__ == "__main__":
    unittest.main()
