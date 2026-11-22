# Swahili ASR: Fine-Tuning and Reproducible Benchmark

**Status (22 November 2026): FLEURS `sw_ke` data acquisition and preparation (T1/T2) and baseline runs (T3: E1, E2) are complete. The fine-tuning contribution (T4 / E3) has been trained on Google Colab T4 GPU (3 epochs, validation WER 0.4096), producing test predictions in `results/predictions/E3.tsv` (the checkpoint itself, ~151 MB, is not committed). The unified benchmark and error analysis (T5) are complete: E3 reaches WER 0.4804 / CER 0.1627 versus 1.1176 / 0.4891 for E1 (see T5 below).**

## Project objective
Evaluate **automatic speech recognition (ASR)** for Swahili (`sw`) using a common, held-out, trustworthy transcription test set. Compare instructor-required **vanilla baselines** against separately trained/fine-tuned models. The proposed contribution is supervised fine-tuning of a specific pretrained multilingual ASR checkpoint, contingent on the Week 9 baseline list and available compute.

## Role allocation and implementation handoff

**Beyza Eser — assigned work:** repository architecture, project workflow, dataset research and actual FLEURS data acquisition/preparation. The local deliverable now includes source hashes, a 3,768-file waveform audit, a real cleaning log, frozen manifests, raw/cleaned selected audio, reproducible code and license attribution. She does not own ASR model training or benchmark experiments. Independent split QA was done separately in T2.

**Whole-team final deliverables:** a technical PPTX presentation and a **finalized, verified README.md** with genuine results and runnable commands. These are mandatory work items, not automatic outputs of the initial scaffold.

**Other members:** independent split/leak QA (T2, Meltem Çelik), main baseline and additional comparator (T3, Berk Ötün), fine-tuning (T4, Asya Oral), WER/CER benchmark and error analysis (T5, Nilasu Yıldız). Each member's work is described in their own file in `contributions/`. Final README review and slide integration are separate shared deliverables.

**Start from [docs/TEAM_TASKS.md](docs/TEAM_TASKS.md)** for explicit inputs, steps, commands/acceptance tests, evidence and grading traps. 

## Agreed experiment scope (team decision, 2026-11-17)

**Two ASR architectures, three evaluation runs, only one fine-tuning training job:**

| Run | Model | Role in our report | Group training |
| --- | --- | --- | --- |
| E1 | `openai/whisper-tiny` (commit `169d4a4341`) | **Main course baseline** (vanilla pretrained inference) | None |
| E2 | `facebook/mms-1b-all` (commit `3d33597edb`), wav2vec2 **CTC ASR checkpoint with a transcription head** and Swahili adapter `swh` | **Additional pretrained comparator** (also technically a baseline; not a new model contribution) | None |
| E3 | E1 Whisper-Tiny checkpoint fine-tuned on the selected Swahili training split | **Our trained contribution** | Fine-tuning required |

All three runs must use the **same frozen test utterances**, transcription normalization, WER/CER code and test IDs. E1 and E2 require inference; E3 requires training. The lecturer requires **one main baseline**, as clarified by the student; E2 is explicitly an *additional baseline-type comparison*, not a second required course baseline. If written Project 3 instructions later differ, reconcile them before submission.

**Resource cap:** maximum **10 hours of usable audio in the selected corpus overall**, target **8h training + 1h validation + 1h test**, measured by actual clip durations rather than count. Speaker-disjoint splitting means the exact 8/1/1 ratio may be approximate. The 10-hour cap is a **data budget**, not a guaranteed 6–13 hour runtime; E3 training took 6.5 minutes on a Colab T4 GPU.

**Dataset overlap:** the student confirmed the chosen dataset does not overlap other groups; retain that confirmation in the coordination notes. Dataset minimum-hours threshold from the instructor has not been provided.

## Dataset decisions and remaining gates
1. Other-group overlap was checked and reported as absent by the student; retain documentary confirmation before submission.
2. Current student clarification: the instructor requests one main baseline; our main baseline is vanilla Whisper-Tiny. Validate any contradictory written Week 9 project-specific instructions before submission.
3. Selected dataset: Google FLEURS `sw_ke`, CC BY 4.0. The verified frozen subset is 7.999300h train, 0.789900h validation and 0.997716667h test (9.786916667h total).
4. Audio is stored with Git LFS (about 2.2 GB for the 2,364 cleaned WAVs; GitHub Free/Pro includes 10 GiB LFS storage and 10 GiB monthly bandwidth). A fresh clone on Colab pulled all 2,364 cleaned WAVs with `git lfs pull`.
5. Model IDs and resolved commits, training configuration, split manifests and metric normalization are recorded in `results/runs/*.json`, `configs/experiment.json` and `src/metrics.py`.

## Documentation and folders
| Path | Purpose |
| --- | --- |
| `docs/PROJECT_PLAN.md` | Research questions, stages, acceptance gates, schedule and ownership |
| `docs/DATASET_RESEARCH.md` | Source evidence, license, access, selection criteria, feasibility |
| `docs/TEAM_WORKFLOW.md` | Git and contribution requirements, definition of done |
| `docs/TEAM_TASKS.md` | Detailed member-facing implementation tasks and acceptance gates |
| `docs/GRADING_AUDIT.md` | Instructor announcement mapped to submission evidence |
| `docs/BASELINE_REGISTRY.md` | Instructor baseline checklist and proposed experiments |
| `docs/COLAB_TRAINING_GUIDE.md` | T4: step-by-step guide for E3 training and inference on a Colab T4 GPU (in Turkish) |
| `contributions/` | One correctly named file per member (Beyza T1, Meltem T2, Berk T3, Asya T4, Nilasu T5) |
| `data/raw/` | Selected original FLEURS audio (Git LFS) and its manifest |
| `data/cleaned/` | Cleaned audio (Git LFS), manifest and cleaning log generated from raw |
| `data/audit/` | Full waveform audit of all 3,768 source clips and its summary |
| `data/splits/` | Frozen train/validation/test manifests (official FLEURS boundaries) and the T2 QA report |
| `src/prepare_fleurs.py` | Audits every FLEURS TSV/WAV, measures duration/quality, selects <=10h and extracts raw/cleaned audio |
| `src/verify_splits.py` | T2: independent QA of the frozen FLEURS splits (boundaries, overlap, counts/hours, audio hashes) -> `data/splits/qa_report.json` |
| `src/audit_manifest.py` | Earlier generic Common Voice-like metadata-only utility; not the FLEURS evidence source |
| `src/split_manifest.py` | Speaker-group-disjoint train/validation/test split utility |
| `src/run_baseline.py` | T3/T4: test-set inference for E1 Whisper-Tiny, E2 MMS wav2vec2-CTC and the fine-tuned E3 checkpoint -> `results/predictions/`, `results/runs/` |
| `src/train.py` | T4: Whisper-Tiny fine-tuning on the train split with checkpoint selection on validation -> `results/runs/E3_train.json` |
| `src/score_predictions.py` | Scores prediction files on the frozen test IDs with `metrics.py` -> `results/metrics.csv` |
| `src/metrics.py` | Explicit WER/CER scoring and reference normalization (NFC, lowercase, punctuation removed except apostrophe) |
| `src/error_analysis.py` | T5: bootstrap CIs, per-clip errors, subgroups, OOV, word-boundary and substitution analysis -> `results/error_analysis/` |
| `tests/` | Standard-library tests for cleaning, splitting, baselines, training artefacts, WER/CER and error analysis |
| `configs/` | Experiment configuration: model IDs/revisions, split sizes and E3 training settings |
| `notebooks/` | T4: Colab notebook for E3 training and evaluation |
| `results/` | Machine-produced predictions, run logs, `metrics.csv` and `error_analysis/` tables |
| `requirements.txt` | Exact versions of first-stage preparation/evaluation dependencies |
| `requirements-models.txt` | Model inference and training dependencies for T3/T4 (torch, transformers, accelerate, soundfile) |
| `LICENSE` | Project-owned source code license; third-party datasets retain own terms |

## Setup
Python 3.11 is the recommended starting environment. Create a virtual environment and run:
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
The tests need no GPU or audio download. They cover data integrity and leakage, WER/CER scoring, baseline and training utilities, and the committed benchmark artefacts; they do not re-run model training.

## Dataset pipeline
Place the six original FLEURS `sw_ke` files (`train/dev/test.tsv` and `.tar`) in one input directory, then run:
```bash
python src/prepare_fleurs.py --input-dir /path/to/FLEURS-sw_ke-source-files --output-root data --seed 42 --extract-audio
```
The pipeline checks literal seven-field TSV structure; tar membership; RIFF/IEEE-Float32, mono and 16 kHz format; frame-count agreement; nonfinite/zero-energy audio; >30-second clips; and clipping/silence review thresholds. It hashes every source archive and WAV. It does **not** correct reference transcripts, denoise waveforms, or replace listening-based intelligibility/language review. The source TSV has no speaker IDs, so official speaker-disjoint split boundaries are preserved instead of claiming a newly generated speaker split.

## T2 — Independent split QA (Meltem)
```bash
python src/verify_splits.py                                  # metadata checks H1-H7
git lfs pull                                                 # download the real WAVs (~2.3 GB)
python src/verify_splits.py --audio-dir data/cleaned/audio    # + H8: SHA-256 and WAV header of every clip
```
The FLEURS splits are **not re-generated**: T1 kept the official boundaries. `verify_splits.py` re-checks T1's claims with separately written code and writes `data/splits/qa_report.json`. Result on the committed manifests and LFS audio (commit `d8ebe3e`): **all H1–H8 PASS**, 2,364/2,364 WAVs verified (SHA-256 matches manifest; IEEE-float32 mono 16 kHz header; sample count matches) — 1,902 / 210 / 252 clips, 7.9993 / 0.7899 / 0.9977 h (9.7869 h total), zero shared filename, sentence ID, normalised transcript or audio SHA-256 across splits, no rejected clip in any split, 0 near-duplicate transcript pairs (trigram Jaccard ≥ 0.8). Limitation: FLEURS TSVs have no speaker ID, so speaker disjointness relies on the official FLEURS split design and cannot be proven by ID.

## T3 — Baselines E1 / E2 (Berk)
```bash
git lfs pull --include="data/cleaned/audio/test/*"            # only the 252 test WAVs
python -m pip install -r requirements-models.txt
python src/run_baseline.py --run E1 --limit 8 --output results/smoke/E1.tsv   # smoke test
python src/run_baseline.py --run E1      # openai/whisper-tiny, language=sw, greedy
python src/run_baseline.py --run E2      # facebook/mms-1b-all + Swahili adapter "swh", greedy CTC
python src/score_predictions.py --predictions results/predictions/E1.tsv results/predictions/E2.tsv
```
Defaults: manifest `data/splits/test.tsv`, audio `data/cleaned/audio/test/`, reference column `raw_transcription`. Each run writes `results/predictions/<RUN>.tsv` (id, reference, hypothesis) and `results/runs/<RUN>.json` (resolved model commit, versions, device, runtime, manifest SHA-256). `score_predictions.py` refuses to score unless predictions cover exactly the 252 frozen test IDs with unchanged references. E2 (MMS-1B, ~4 GB) needs a GPU (e.g. Colab T4). **Caveat:** MMS-1b-all was fine-tuned on FLEURS training data among other corpora, so E2 is in-domain; its test split is still held out.

## T4 — Whisper-Tiny fine-tuning contribution (Asya Oral)

The fine-tuning contribution (E3) trains `openai/whisper-tiny` (pinned base revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`) on `data/splits/train.tsv` (1,902 clips, ~8h) with validation on `data/splits/validation.tsv` (210 clips, ~0.8h) using Google Colab T4 GPU:
- Training protocol: 3 epochs, batch size 8, gradient accumulation 2 (effective batch 16), lr 1e-4, warmup 50, fp16.
- The best checkpoint was selected by validation WER (0.4096, loss 0.8235) and saved to `results/checkpoints/E3/best` on Colab (backed up in `E3_backup.zip`; not committed to the repository).
- Full training logs and hardware metrics are saved in `results/runs/E3_train.json`.
- Inference on the frozen test split (`data/splits/test.tsv`, 252 clips) generates `results/predictions/E3.tsv` and `results/runs/E3.json`.

```bash
# E3 fine-tuning training command:
python src/train.py --fp16 --epochs 3 --batch-size 8 --grad-accum 2 --learning-rate 1e-4 --output-dir results/checkpoints/E3

# E3 test-set inference:
python src/run_baseline.py --run E3
```

## T5 — Benchmark, statistics and error analysis (Nilasu Yıldız)

All three runs are scored on the same 252 frozen test clips (`data/splits/test.tsv`) with one shared scorer. Both scripts refuse to run unless a prediction file covers exactly the frozen test IDs with unchanged reference text.

```bash
python -m pip install -r requirements.txt          # no GPU, audio or model download needed
python src/score_predictions.py --predictions results/predictions/E1.tsv results/predictions/E2.tsv results/predictions/E3.tsv
python src/error_analysis.py   --predictions results/predictions/E1.tsv results/predictions/E2.tsv results/predictions/E3.tsv
python -m unittest discover -s tests -v
```

The first command writes `results/metrics.csv`; the second writes every table in `results/error_analysis/` (the first prediction file is the reference run for paired comparisons). Bootstrap resampling uses a fixed seed (42, 2,000 resamples), so reruns reproduce the committed files exactly. Neither file is edited by hand.

### Metrics
- **WER** (primary) = (S + D + I) / N: substituted, deleted and inserted words over the number of reference words, pooled over all clips (not an average of per-clip WER). Insertions can push WER above 1, which is what happens when E1 loops.
- **CER** (secondary) = character edits / reference characters. Swahili is agglutinative, so a one-letter mistake makes a whole word wrong under WER; CER shows how close the transcript still is.

### Normalisation policy (`src/metrics.py`, identical for references and hypotheses)
Unicode NFC, lower case, every Unicode punctuation character replaced by a space **except the apostrophe**, curly apostrophe mapped to `'`, whitespace collapsed. Punctuation is removed because Whisper emits it and the FLEURS `raw_transcription` contains it, but it is not spoken. The apostrophe is kept because it is part of Swahili spelling (`ng'ombe`). Digits are **not** verbalised: `1967` vs `elfu moja...` counts as an error. The policy was fixed before E3 was scored and is covered by `tests/test_metrics.py`.

### Results (`results/metrics.csv`)

| Run | Role | WER | CER | WER 95% CI |
| --- | --- | ---: | ---: | --- |
| E1 | Whisper-Tiny, vanilla (main baseline) | 1.1176 | 0.4891 | 1.051 – 1.197 |
| E2 | MMS-1B-all + `swh` (comparator, saw FLEURS train) | 0.1483 | 0.0367 | 0.132 – 0.165 |
| E3 | Whisper-Tiny fine-tuned on our 8 h (contribution) | 0.4804 | 0.1627 | 0.416 – 0.562 |

Paired bootstrap, E3 − E1: **−0.637 WER** (95% CI −0.732 to −0.553, p ≤ 0.001). E3 is better than E1 on 246 of 252 clips. Confidence intervals resample whole `sentence_id` clusters, because FLEURS records the same sentence with several speakers (252 clips, 162 sentences). Values from `results/error_analysis/bootstrap.csv` and `pairwise.csv`.

### Error analysis (`results/error_analysis/`)

| File | Content |
| --- | --- |
| `summary.csv` | Per run: WER/CER, S/D/I counts, insertion share, median clip WER, clips with WER > 1, repetition loops, perfect clips |
| `utterances.tsv` | Every run × clip with its error counts and test-set metadata |
| `subgroups.csv` | WER/CER by gender, clip duration, digits in the reference and loudness (RMS tercile) |
| `pairwise.csv` | Clips where each run is better / worse / equal compared with E1 |
| `bootstrap.csv` | 95% CI per run and paired differences with p-values |
| `oov.csv` | Error rate on reference words seen vs never seen in the training transcripts |
| `word_boundary.csv` | CER with and without spaces; share of character edits that are merged/split words |
| `substitution_categories.csv` | Where substituted words differ: short function word, word start, middle, end, unrelated |
| `substitutions.tsv` | Most frequent reference → hypothesis word substitutions |
| `examples.md` | Best, median and worst clips per run for qualitative review |

Main findings:
- **E1's problem is mostly wrong words, not only loops.** 24 clips contain a repetition loop (an n-gram of up to 3 words repeated back-to-back at least 4 times; natural reduplication such as `pole pole` stays below this threshold), but substitutions are 3,849 of 5,758 errors and WER stays 0.92 even without the clips above WER 1. In E3 loops drop to 5 clips, although insertions remain 26% of its errors.
- **Fine-tuning helps most on words seen in training.** 1,064 of 5,152 test words (21%) never occur in the training transcripts. E3's error rate is 27% on seen words but 70% on unseen ones (E1: 90% / 92%). Unseen words are rarer and often names, so even E2 triples its error on them (10% / 31%), but this shows the limit of 8 hours of data.
- **E3's errors become near misses.** Substitutions with nothing in common with the reference fall from 69% (E1) to 31% (E3); the rest differ only at the start, middle or end of the word. Merged/split words account for 24% (E1) and 19% (E3) of character edits.
- **Quiet recordings are the weakest subgroup** for every model: E3 WER 0.62 on the quietest RMS tercile vs 0.37 on the loudest. Clips containing digits are also worse (0.58 vs 0.46).

Limitations: FLEURS has no speaker IDs, so no per-speaker analysis is possible; some subgroups are small (30–39 clips), so their differences carry wide uncertainty; the substitution categories are a character-overlap heuristic, not a morphological parser; E2's low error is in-domain because MMS was trained on FLEURS; E3 was trained once with one seed, so the confidence intervals cover test-set sampling but not training randomness.

## Reproducibility and grading
- Every member commits their own work from their own GitHub account.
- Use `contributions/<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` for **each actual member**. Do not use placeholders as submissions.
- Pin versions; document model IDs vs architecture names. Record Git commit SHA, random seeds, hardware and dataset fingerprints with every completed run.
- Do not rely on a Colab-only upload, unpublished numbers, unchecked notebook, or manual result-table edits.
- Keep all tracked files relevant and described. Commit the actual datasets when selected and within GitHub/LFS/license constraints.
- The lecturer grades the final commit **at the deadline**; check the public link in a private browser.

## Data and code licensing
Our original code uses the root MIT license. FLEURS data remains under **CC BY 4.0** with attribution and modification notice in `data/LICENSE.md`; the MIT license does not relicense it. These rights are not implied to transfer to model weights or third-party code.

## Sources
- Google FLEURS dataset: https://huggingface.co/datasets/google/fleurs
- FLEURS paper: https://arxiv.org/abs/2205.12446
- GitHub LFS billing: https://docs.github.com/en/billing/concepts/product-billing/git-lfs
- Hugging Face ASR evaluation guide: https://huggingface.co/learn/audio-course/en/chapter5/evaluation
