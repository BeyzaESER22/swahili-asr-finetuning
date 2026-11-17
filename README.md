# Swahili ASR: Fine-Tuning and Reproducible Benchmark

**Status: project scaffolding and dataset research only (17 November 2026). No dataset has been downloaded, no model has been trained, and no benchmark scores are claimed.**

## Project objective
Evaluate **automatic speech recognition (ASR)** for Swahili (`sw`) using a common, held-out, trustworthy transcription test set. Compare instructor-required **vanilla baselines** against separately trained/fine-tuned models. The proposed contribution is supervised fine-tuning of a specific pretrained multilingual ASR checkpoint, contingent on the Week 9 baseline list and available compute.

## Role allocation and implementation handoff

**Beyza Eser (completed initial responsibility):** GitHub repository architecture, overall project workflow, dataset source discovery and selection research, source/license feasibility documentation, and the initial acceptance criteria. **Beyza is NOT assigned to download, clean, upload, split or train on the dataset.** Starter scripts are scaffolding only and must be implemented/validated by the team members who take those tasks.

**To assign to other members:** acquisition and Git/LFS raw-audio upload; actual audio+transcript cleaning and cleaned-audio upload; speaker-disjoint splitting and leakage checks; required vanilla baselines; model fine-tuning; unified WER/CER benchmark; reporting and presentation. Owners are `UNASSIGNED` until agreed, not fictitious.

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

## Critical decisions (not yet approved)
1. Other-group overlap was checked and reported as absent by the student; retain documentary confirmation before submission.
2. Current student clarification: the instructor requests one main baseline; our main baseline is vanilla Whisper-Tiny. Validate any contradictory written Week 9 project-specific instructions before submission.
3. Select a feasible **at-most-10-hour** licensed subset of Mozilla Common Voice Scripted Speech v27.0 (Swahili); target 8h train, 1h validation, 1h test, subject to speaker grouping. FLEURS remains a possible external test only.
4. Confirm audio storage, GitHub size/LFS and lecturer's rule that **both raw and cleaned data must actually be inside the repository**. A download link or script alone does not satisfy that rule.
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
| `src/audit_manifest.py` | Audits a Common Voice-like TSV, rejects invalid metadata, logs reasons |
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

## Dataset pipeline (assigned to data engineering member, not Beyza; once legitimately obtained)
Place an allowed Common Voice `validated.tsv` subset in `data/raw/` and the referenced audio files in a matching clips directory. For a reproducible small demonstration:
```bash
python src/audit_manifest.py --input data/raw/validated.tsv --audio-root data/raw/clips --output data/cleaned/validated_cleaned.tsv --log data/cleaned/cleaning_log.csv
python src/split_manifest.py --input data/cleaned/validated_cleaned.tsv --output-dir data/splits --seed 42
```
The manifest auditor checks empty transcripts, missing speaker IDs, missing audio, exact repeated clip paths, and exact repeated text/audio pairs; it logs every removal. **It does not correct incorrect reference transcripts, detect noisy recordings, or alter audio.** Those additional quality checks must be documented and implemented if promised. The splitter rejects missing speaker IDs rather than silently allowing speaker leakage.

## Benchmark protocol (planned, not completed)
- Reserve one **speaker-disjoint held-out test set**. Use validation only for tuning and checkpoint selection.
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
Our original code: MIT license (root `LICENSE`). Mozilla Common Voice Swahili is listed under **CC0-1.0** by the source, subject to its distribution terms and ethical restrictions. Google FLEURS has separate applicable terms: review the official dataset card before use. These rights are **not** implied to transfer to weights, other datasets, or third-party code.

## Sources
- Mozilla Common Voice catalog: https://commonvoice.mozilla.org/en/datasets
- Mozilla release announcement: https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490
- Google FLEURS dataset: https://huggingface.co/datasets/google/fleurs
- Hugging Face ASR evaluation guide: https://huggingface.co/learn/audio-course/en/chapter5/evaluation
