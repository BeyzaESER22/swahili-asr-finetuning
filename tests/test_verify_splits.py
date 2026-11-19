"""T2 tests: independent QA of the frozen FLEURS splits (real data + injected faults)."""
import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verify_splits import SPLITS, norm_text, read_tsv, verify, wav_header

DATA = Path(__file__).resolve().parents[1] / "data"


def load_real():
    splits = {n: read_tsv(DATA / "splits" / f"{n}.tsv") for n in SPLITS}
    cleaned = read_tsv(DATA / "cleaned" / "manifest.tsv")
    log = read_tsv(DATA / "cleaned" / "cleaning_log.tsv")
    summary = json.loads((DATA / "audit" / "summary.json").read_text(encoding="utf-8"))
    return splits, cleaned, log, summary


def float_wav(n_samples: int, rate: int = 16000) -> bytes:
    payload = b"\x00\x00\x00\x00" * n_samples
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = load_real()

    def run_verify(self, splits=None, cleaned=None, log=None, summary=None, **kw):
        s, c, l, m = copy.deepcopy(self.real)
        return verify(splits or s, cleaned or c, log or l, summary or m, **kw)

    def test_frozen_splits_pass(self):
        report = self.run_verify()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual([report["splits"][n]["clips"] for n in SPLITS], [1902, 210, 252])
        self.assertLessEqual(report["total_hours"], 10)

    def test_detects_test_clip_copied_into_train(self):
        s, c, l, m = copy.deepcopy(self.real)
        s["train"].append(s["test"][0])
        report = verify(s, c, l, m)
        self.assertFalse(report["passed"])
        self.assertEqual(report["checks"]["H5_no_cross_split_overlap"], "FAIL")
        self.assertEqual(report["checks"]["H2_official_boundaries"], "FAIL")

    def test_detects_rejected_clip_in_split(self):
        s, c, l, m = copy.deepcopy(self.real)
        s["train"][0]["filename"] = l[0]["filename"]
        self.assertEqual(verify(s, c, l, m)["checks"]["H3_only_accepted"], "FAIL")

    def test_detects_edited_reference(self):
        s, c, l, m = copy.deepcopy(self.real)
        s["test"][0]["raw_transcription"] += " x"
        self.assertEqual(verify(s, c, l, m)["checks"]["H4_matches_cleaned_manifest"], "FAIL")

    def test_detects_hours_mismatch_with_summary(self):
        s, c, l, m = copy.deepcopy(self.real)
        m["splits"]["test"]["selected_hours"] = 0.5
        self.assertEqual(verify(s, c, l, m)["checks"]["H7_counts_and_hours"], "FAIL")


class AudioCheckTests(unittest.TestCase):
    def make(self, tmp, data: bytes, n_samples: int):
        root = Path(tmp)
        rows = {}
        for i, name in enumerate(SPLITS):
            (root / name).mkdir()
            content = data if name == "test" else float_wav(10 + i)
            (root / name / f"{name}.wav").write_bytes(content)
            rows[name] = [{
                "sentence_id": str(i), "filename": f"{name}.wav", "raw_transcription": f"sentensi {i}",
                "num_samples": str(n_samples if name == "test" else 10 + i), "gender": "MALE",
                "source_split": {"validation": "dev"}.get(name, name), "duration_seconds": "1",
                "sample_rate_hz": "16000", "channels": "1", "audit_status": "accepted",
                "sha256": hashlib.sha256(content).hexdigest()}]
            rows[name][0]["duration_seconds"] = str(int(rows[name][0]["num_samples"]) / 16000)
        cleaned = [r for v in rows.values() for r in v]
        return rows, cleaned, root

    def test_good_audio_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cleaned, root = self.make(tmp, float_wav(16), 16)
            report = verify(rows, cleaned, [], None, audio_dir=root)
            self.assertTrue(report["passed"], report["failures"])
            self.assertEqual(report["audio"]["checked"], 3)

    def test_lfs_pointer_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cleaned, root = self.make(tmp, b"version https://git-lfs.github.com/spec/v1\n", 16)
            report = verify(rows, cleaned, [], None, audio_dir=root)
            self.assertEqual(report["audio"]["lfs_pointer_not_downloaded"], 1)
            self.assertFalse(report["passed"])

    def test_wav_header_parser(self):
        self.assertEqual(wav_header(float_wav(5)),
                         {"format_tag": 3, "channels": 1, "rate": 16000, "bits": 32, "frames": 5})

    def test_norm_text(self):
        self.assertEqual(norm_text('"Kama, itakuwa." Bw.'), "kama itakuwa bw")


if __name__ == "__main__":
    unittest.main()
