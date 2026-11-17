# Swahili ASR: Fine-Tuning and Reproducible Benchmark

**Status (17 November 2026): FLEURS `sw_ke` data acquisition, full waveform audit, deterministic 9.786916667-hour selection, preparation code, manifests and Git LFS staging are complete locally. No model has been trained and no benchmark scores are claimed. Remote data upload still requires repository authentication and LFS quota confirmation.**

## Project objective
Evaluate **automatic speech recognition (ASR)** for Swahili (`sw`) using a common, held-out, trustworthy transcription test set. Compare instructor-required **vanilla baselines** against separately trained/fine-tuned models. The proposed contribution is supervised fine-tuning of a specific pretrained multilingual ASR checkpoint, contingent on the Week 9 baseline list and available compute.

## Role allocation and implementation handoff

**Beyza Eser — assigned work:** repository architecture, project workflow, dataset research and actual FLEURS data acquisition/preparation. The local deliverable now includes source hashes, a 3,768-file waveform audit, a real cleaning log, frozen manifests, raw/cleaned selected audio, reproducible code and license attribution. She does not own ASR model training or benchmark experiments. Independent QA remains a T2 handoff.

**Whole-team final deliverables:** a technical PPTX presentation and a **finalized, verified README.md** with genuine results and runnable commands. These are mandatory work items, not automatic outputs of the initial scaffold.

**To assign to other members:** independent split/leak QA, main baseline and additional comparator, fine-tuning, WER/CER benchmark, error analysis, and individual presentation sections. Final README review and slide integration are separate shared deliverables; exact owners remain to be assigned.

**Start from [docs/TEAM_TASKS.md](docs/TEAM_TASKS.md)** for explicit inputs, steps, commands/acceptance tests, evidence and grading traps. Do not start fine-tuning until dataset and instructor-baseline gates pass.

## Agreed experiment scope (team decision, 2026-11-17)

**Two ASR architectures, three evaluation runs, only one fine-tuning training job:**

| Run | Concrete planned model | Role in our report | Group training |
| --- | --- | --- | --- |
| E1 | `openai/whisper-tiny` (pin exact revision before use) | **Main course baseline** (vanilla pretrained inference) | None |
| E2 | Swahili-compatible wav2vec2 **CTC ASR checkpoint with a transcription head**, ID/revision to be validated | **Additional pretrained comparator** (also technically a baseline; not a new model contribution) | None |
| E3 | E1 Whisper-Tiny checkpoint fine-tuned on the selected Swahili training split | **Our trained contribution** | Fine-tuning required |

All three runs must use the **same frozen test utterances**, transcription normalization, WER/CER code and test IDs. E1 and E2 require inference; E3 requires training. The lecturer requires **one main baseline**, as clarified by the student; E2 is explicitly an *additional baseline-type comparison*, not a second required course baseline. If written Project 3 instructions later differ, reconcile them before submission.

**Resource cap:** maximum **10 hours of usable audio in the selected corpus overall**, target **8h training + 1h validation + 1h test**, measured by actual clip durations rather than count. Speaker-disjoint splitting means the exact 8/1/1 ratio may be approximate. The 10-hour cap is a **data budget**, not a guaranteed 6–13 hour runtime; run a short timing pilot and adjust training steps, hardware or schedule. No real data has been acquired yet.

**Dataset overlap:** the student confirmed the chosen dataset does not overlap other groups; retain that confirmation in the coordination notes. Dataset minimum-hours threshold from the instructor has not been provided.

## Dataset decisions and remaining gates
1. Other-group overlap was checked and reported as absent by the student; retain documentary confirmation before submission.
2. Current student clarification: the instructor requests one main baseline; our main baseline is vanilla Whisper-Tiny. Validate any contradictory written Week 9 project-specific instructions before submission.
3. Selected dataset: Google FLEURS `sw_ke`, CC BY 4.0. The verified frozen subset is 7.999300h train, 0.789900h validation and 0.997716667h test (9.786916667h total).
4. Audio is prepared for Git LFS. GitHub Free/Pro currently includes 10 GiB LFS storage and 10 GiB monthly bandwidth, while the selected unique audio is about 2.1 GiB locally. Confirm the account's remaining quota and authenticate before remote upload.
5. Fix exact model IDs/revisions, training configuration, splits and metric normalization *before* reporting scores.

## Documentation and folders
| Path | Purpose |
| --- | --- |
| `docs/PROJECT_PLAN.md` | Research questions, stages, acceptance gates, schedule and ownership |
| `docs/DATASET_RESEARCH.md` | Source evidence, license, access, selection criteria, feasibility |
| `docs/TEAM_WORKFLOW.md` | Git and contribution requirements, definition of done |
| `docs/TEAM_TASKS.md` | Detailed member-facing implementation tasks and acceptance gates |
| `docs/GRADING_AUDIT.md` | Instructor announcement mapped to submission evidence |
| `docs/BASELINE_REGISTRY.md` | Instructor baseline checklist and proposed experiments |
| `contributions/` | One correctly named file per member; currently only Beyza's verified work |
| `data/raw/` | Original audio/TSV subset once acquired (currently only README) |
| `data/cleaned/` | Actual cleaned audio/metadata generated from raw, not claimed yet |
| `data/splits/` | Speaker-disjoint train/validation/test manifests when produced |
| `src/prepare_fleurs.py` | Audits every FLEURS TSV/WAV, measures duration/quality, selects <=10h and extracts raw/cleaned audio |
| `src/audit_manifest.py` | Earlier generic Common Voice-like metadata-only utility; not the FLEURS evidence source |
| `src/split_manifest.py` | Speaker-group-disjoint train/validation/test split utility |
| `src/metrics.py` | Explicit WER/CER scoring and reference normalization |
| `tests/` | Standard-library tests for cleaning, splitting, WER/CER |
| `configs/` | Explicit experimental choices and unresolved placeholders |
| `results/` | Machine-produced benchmark files and predictions (empty until execution) |
| `requirements.txt` | Exact versions of first-stage preparation/evaluation dependencies |
| `LICENSE` | Project-owned source code license; third-party datasets retain own terms |

## Setup
Python 3.11 is the recommended starting environment. Create a virtual environment and run:
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
This validates only **utilities**, not end-to-end model training.

## Dataset pipeline
Place the six original FLEURS `sw_ke` files (`train/dev/test.tsv` and `.tar`) in one input directory, then run:
```bash
python src/prepare_fleurs.py --input-dir /path/to/FLEURS-sw_ke-source-files --output-root data --seed 42 --extract-audio
```
The pipeline checks literal seven-field TSV structure; tar membership; RIFF/IEEE-Float32, mono and 16 kHz format; frame-count agreement; nonfinite/zero-energy audio; >30-second clips; and clipping/silence review thresholds. It hashes every source archive and WAV. It does **not** correct reference transcripts, denoise waveforms, or replace listening-based intelligibility/language review. The source TSV has no speaker IDs, so official speaker-disjoint split boundaries are preserved instead of claiming a newly generated speaker split.

## Benchmark protocol (planned, not completed)
- Use the frozen official-boundary test subset only for final evaluation; use validation for tuning and checkpoint selection.
- Main baseline: vanilla Whisper-Tiny inference; additional comparator: Swahili ASR-head-equipped wav2vec2-CTC pretrained inference; fine-tune Whisper-Tiny as the group's contribution. Re-check additional written course obligations if supplied.
- Contribution: separately fine-tune an explicitly named pretrained ASR checkpoint; save the exact revision, seed, checkpoints, preprocessing version, training logs and predictions.
- Primary: **WER** (word insertions + deletions + substitutions / reference word count). Secondary: **CER**, to complement WER on Swahili morphology. Record text-normalization rules identically for gold/reference and predictions, and save the per-utterance predictions.
- Optional: inference latency and train-set-size ablation, properly standardized and labeled.
- Never invent ground-truth transcripts. Never use a different model's output as verified reference.

**No results are reported until scripts, data, checkpoints / independently reproducible training route, predictions, configuration and outputs are committed and tested.**

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
