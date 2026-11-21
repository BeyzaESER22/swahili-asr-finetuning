"""Tests for T4 training pipeline utilities (manifest I/O, audio path resolution, data preparation)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    ID_COL,
    REF_COL,
    configure_model_for_whisper_finetuning,
    load_audio,
    read_manifest,
    resolve_audio_path,
)

REPO = Path(__file__).resolve().parents[1]


class TrainPipelineTests(unittest.TestCase):
    def test_pinned_defaults(self):
        self.assertEqual(DEFAULT_MODEL_ID, "openai/whisper-tiny")
        self.assertEqual(DEFAULT_REVISION, "169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
        self.assertEqual((ID_COL, REF_COL), ("filename", "raw_transcription"))

    def test_train_and_val_manifests_readable(self):
        train_rows = read_manifest(REPO / "data" / "splits" / "train.tsv")
        val_rows = read_manifest(REPO / "data" / "splits" / "validation.tsv")
        self.assertEqual(len(train_rows), 1902)
        self.assertEqual(len(val_rows), 210)
        # Verify limit option works
        train_limit = read_manifest(REPO / "data" / "splits" / "train.tsv", limit=2)
        self.assertEqual(len(train_limit), 2)

    def test_resolve_audio_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            sample_file = root / "train" / "sample.wav"
            sample_file.write_bytes(b"RIFF dummy")

            found = resolve_audio_path(root, "sample.wav")
            self.assertEqual(found, sample_file)

            with self.assertRaises(FileNotFoundError):
                resolve_audio_path(root, "missing.wav")

    def test_lfs_pointer_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ptr = Path(tmp) / "ptr.wav"
            ptr.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:123\n")
            with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
                load_audio(ptr)


    def test_whisper_generation_config_pinning(self):
        class DummyConfig:
            forced_decoder_ids = [1, 2]
            suppress_tokens = [3, 4]
            use_cache = True

        class DummyGenerationConfig:
            language = "en"
            task = "translate"
            forced_decoder_ids = [1, 2]

        class DummyModel:
            config = DummyConfig()
            generation_config = DummyGenerationConfig()

        model = DummyModel()
        configure_model_for_whisper_finetuning(model, language="sw", task="transcribe")
        self.assertIsNone(model.config.forced_decoder_ids)
        self.assertEqual(model.config.suppress_tokens, [])
        self.assertFalse(model.config.use_cache)
        self.assertEqual(model.generation_config.language, "sw")
        self.assertEqual(model.generation_config.task, "transcribe")
        self.assertIsNone(model.generation_config.forced_decoder_ids)


if __name__ == "__main__":
    unittest.main()

