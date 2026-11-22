# Individual contribution — Asya Oral

- Student name: Asya Oral
- Student ID: 2309011045
- GitHub account: `asyaoral`
- Project role: **Whisper-Tiny Fine-Tuning Pipeline, Model Training Contribution & Artifact Management (T4)**

## Scope of my work (2026-11-22)
- Took over the T4 handoff after the FLEURS `sw_ke` split was frozen in T1, independently audited in T2, and baseline inferences (E1 Whisper-Tiny, E2 MMS-1B) were completed in T3.
- Owned the single required trainable group contribution: supervised fine-tuning of `openai/whisper-tiny` (pinned base revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`) on the Swahili training corpus.
- Maintained a strict partition between training/validation and the frozen test set: the test split (`data/splits/test.tsv`, 252 clips) was held out completely during training and used exclusively for final offline model evaluation.

## Completed Whisper-Tiny fine-tuning contribution (2026-11-22)

I implemented the end-to-end fine-tuning pipeline in `src/train.py`. The script configures the Whisper processor and tokenizer for Swahili (`language="sw"`, `task="transcribe"`), suppresses decoder cache and timestamp generation for CTC-free autoregressive training, and loads audio at 16 kHz mono float32 with Git LFS pointer validation. It handles dynamic sequence padding via `DataCollatorSpeechSeq2SeqWithPadding`, cross-entropy sequence-to-sequence loss with label masking (`-100` for pad tokens), linear learning rate warmup and decay, and validation-loss/WER tracking across epochs.

To support cloud-based GPU execution, I authored `notebooks/train_whisper_e3.ipynb` and `docs/COLAB_TRAINING_GUIDE.md`, detailing step-by-step reproduction on Google Colab:
- **Hardware:** Google Colab Tesla T4 GPU (16 GB VRAM), CUDA 12.4, PyTorch 2.5.1, Transformers 4.46.3.
- **Dataset protocol:** Trained exclusively on `data/splits/train.tsv` (1,902 clips, 7.9993 hours); validated exclusively on `data/splits/validation.tsv` (210 clips, 0.7899 hours).
- **Hyperparameters:** 3.0 epochs, per-device batch size 8, gradient accumulation 2 (effective batch size 16), learning rate 1e-4 with AdamW, 50 warmup steps, mixed precision (FP16), seed 42. Total training wall-clock time: 393.43 seconds (~6.5 minutes).
- **Checkpoint selection:** Validation WER and loss were evaluated at each epoch. The best checkpoint was achieved and selected at **Epoch 2** with a validation WER of **0.4096** (val loss: 0.8235, train loss: 0.5713), automatically saved to `results/checkpoints/E3/best/`.

Following training, I evaluated the best checkpoint on the frozen test split (`data/splits/test.tsv`, 252 clips, 0.9977 hours) using `src/run_baseline.py --run E3`, recording raw hypotheses in `results/predictions/E3.tsv` and full execution provenance in `results/runs/E3.json` (runtime: 30.42 s, real-time factor: 0.0085 on Tesla T4).

### Test results summary:

| Run | Role | Model / Checkpoint | Test clips | WER | CER |
| --- | --- | --- | ---: | ---: | ---: |
| **E1** | Main course baseline | `openai/whisper-tiny` (vanilla, `169d4a4341`) | 252 | 1.1176 | 0.4891 |
| **E2** | Pretrained comparator | `facebook/mms-1b-all` + `swh` (`3d33597edb`) | 252 | 0.1483 | 0.0367 |
| **E3** | Trained contribution | Fine-tuned Whisper-Tiny (`results/checkpoints/E3/best`) | 252 | **0.4804** | **0.1627** |

- **Comparison vs Baseline (E1):** Supervised fine-tuning yielded a dramatic improvement over the vanilla baseline. E3 reduced WER from **1.1176 to 0.4804** (an absolute drop of 63.72 percentage points; relative error reduction of 57.0%) and CER from **0.4891 to 0.1627** (relative error reduction of 66.7%), effectively suppressing the runaway token hallucination loops seen in E1.
- **Comparison vs Comparator (E2):** E3 performs lower than the external comparator E2 (MMS-1B, WER 0.1483). This is expected and technically justified: MMS-1B contains 1 billion parameters (~25× larger than Whisper-Tiny's 39M parameters) and was directly pre-trained on FLEURS Swahili speech, whereas our model was fine-tuned under strict parameter and compute constraints from a generic baseline.

### Artifact management & GitHub file limit:
The fine-tuned model checkpoint (`results/checkpoints/E3/best/model.safetensors`) is **151 MB**, exceeding GitHub's standard 100 MB hard limit. In compliance with repository policy and without modifying `.gitattributes`, model weights were not committed to Git; the trained weights are preserved locally and archived in `E3_backup.zip`. Provenance and reproducibility are guaranteed by committing the full training log (`results/runs/E3_train.json`), evaluation log (`results/runs/E3.json`), and raw per-utterance predictions (`results/predictions/E3.tsv`).

I also implemented unit tests in `tests/test_train.py` (5 tests covering pinned defaults, manifest I/O, audio path resolution, LFS pointer guards, and generation config pinning; full test suite passes). I updated `configs/experiment.json`, README T4 documentation, and `docs/TEAM_TASKS.md`.

Final shared benchmark evaluation and linguistic error analysis (T5), README review (T6), and the technical presentation (T7) remain shared or assigned to other team members.

## Attribution limitations
Code and documentation were prepared with AI assistance, under my direction and review. I authored the training pipeline, executed the training on Google Colab T4 GPU, checked training curves and test outputs, and committed the work from my own GitHub account (`asyaoral`). The dataset preparation (T1, Beyza Eser), independent split QA (T2, Meltem Çelik), and baseline implementations (T3, Berk Ötün) are acknowledged and **not** claimed here.

## Evidence
- Code commit: [`2740062`](https://github.com/BeyzaESER22/swahili-asr-finetuning/commit/274006278833ed215b8303a3a1092cd25928a707) — `feat(T4): add Whisper-Tiny fine-tuning pipeline, config, tests and Colab guide`
- Results commit: [`bfb3af5`](https://github.com/BeyzaESER22/swahili-asr-finetuning/commit/bfb3af52614155c0b825a34cd6eb779cd6cbad19) — `feat(T4): add fine-tuned Whisper-Tiny E3 training logs, predictions and status updates`
- Tracked artifacts: `src/train.py`, `notebooks/train_whisper_e3.ipynb`, `docs/COLAB_TRAINING_GUIDE.md`, `tests/test_train.py`, `results/runs/E3_train.json`, `results/runs/E3.json`, `results/predictions/E3.tsv`, `configs/experiment.json`, and this file's Git history.
- Reproduce training: `python src/train.py --fp16 --epochs 3 --batch-size 8 --grad-accum 2 --learning-rate 1e-4 --output-dir results/checkpoints/E3`
- Reproduce inference: `python src/run_baseline.py --run E3`
