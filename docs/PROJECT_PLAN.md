# Project plan — Swahili automatic speech recognition

**Owner for architecture, workflow, dataset research and actual data preparation:** Beyza Eser. Initial architecture/research documented; real dataset processing pending.
**Role boundary (updated):** Beyza owns dataset discovery, **download/selection, cleaning/preprocessing code and log, raw and processed audio+transcript artifact upload**, as well as repository architecture/project organization. These datasets/scripts are not yet completed. A separate team member should own speaker-disjoint splits/leakage QA (T2) and other members should own ASR baseline, training and evaluation. A final README audit and presentation are mandatory shared deliverables.
**Detailed execution handoff:** `docs/TEAM_TASKS.md`; grade-based exit checks: `docs/GRADING_AUDIT.md`.
**Current phase:** proposal / repository initialization. No training or collection completed.

## Scientific question
Can supervised fine-tuning of a specified multilingual pretrained ASR model reduce WER and CER on speaker-disjoint Swahili speech compared with the same checkpoint run without additional group training?

## Evaluation hypothesis
Fine-tuning may improve transcription accuracy, but this is a testable hypothesis, **not a promised result**. Compare baseline and fine-tuned versions under identical normalization and test data.

## Gates and ordered workflow
| Gate | Deliverable | Exit condition |
| --- | --- | --- |
| 0. Coordination | Assigned language, dataset, member tasks | Confirm no other group uses Swahili or same training dataset |
| 1. Dataset feasibility | Source/license sheet, sample audit, duration/speaker counts | Can obtain trusted transcripts and keep raw/cleaned data in repo |
| 2. Protocol | Fixed split IDs, normalization, metric unit tests | No speaker or exact audio duplication across splits |
| 3. Baseline and comparator | One mandatory Whisper-Tiny main baseline + one optional wav2vec2-CTC pretrained comparator | Both inference runs evaluate on same held-out test split |
| 4. Contribution | Reproducible training/fine-tuning script and checkpoint provenance | Separately identify original training contribution |
| 5. Comparison | WER/CER CSV, per-utterance outputs, error analysis | Every number is regenerated from committed code |
| 6. Final README | Updated commands, provenance, results, role attribution and complete file inventory | Each claim matches committed scripts/data; another member verifies reproduction |
| 7. Presentation | Complete PPTX, technical model explanation, speaker contributions and rehearsal | Submitted deck, under instructor's time limit; no reading |
| 8. Submission | Per-person contribution histories, final audit and PPTX | Public clean repo, full link and PPTX submitted on time |

## Agreed three-run experiment matrix

| Experiment | Model | Classification | Training by group |
| --- | --- | --- | --- |
| E1 | Whisper-Tiny (`openai/whisper-tiny`; revision to pin) | **ONE main vanilla baseline** | No |
| E2 | Swahili-capable wav2vec2-CTC checkpoint with ASR head (ID to verify) | Additional pretrained comparator; *technically another baseline*, not our contribution | No |
| E3 | Fine-tuned Whisper-Tiny, initialized from E1 | **Trained model contribution** | Yes, one fine-tuning job |

**Dataset ceiling:** maximum 10 hours of **total audio**; suggested allocations 8h train, 1h validation, 1h test; measure actual clip durations and preserve disjoint speakers. Training-runtime ceiling is NOT guaranteed; use a brief training throughput pilot, adjust batch size/steps/epoch if needed. Only training corpus should be drawn from the selected permitted dataset and no test utterance should leak into training.

Student confirmation: no other group uses the selected dataset. Instructor's **one main baseline** requirement is treated as known from student clarification, with written Week 9 rule recheck only if additional material conflicts.

Wav2vec2 must be a full CTC-ASR checkpoint with a trained transcription head; an encoder-only checkpoint is not a usable ASR model. All three model runs use identical held-out test IDs, reference text, normalization, WER/CER.

## Metrics / statistical interpretation
WER is the primary ASR task metric, not text-classification accuracy. CER helps with agglutinative/morphological spelling errors. Compute both with one frozen normalization policy, identical reference/hypothesis handling, and report sample counts. Inspect specific substitutions/deletions/insertions. Use latency only under consistent hardware, batch size and decoding parameters.

## Threats to validity
Crowdsourced transcript errors, text normalization bias, cross-split speaker leakage, dialect imbalance, near-duplicate utterances, dataset license restrictions, compute limitations, pretrained model exposure to data, and mismatch between Common Voice and external FLEURS domains.

## Ownership boundary
Beyza: repository structure, methodology/project flow, source evaluation and dataset selection; **real subset acquisition, cleaning and preprocessing scripts, cleaning logs, raw and cleaned corpus in GitHub** (assigned, not yet completed).
Other team members: split generation/leakage checks, model baselines, fine-tuning and evaluations; final README and PPTX presentation are explicit shared tasks. Assign actual names only after agreement. Each member will create their **own** contribution file and commits. No contribution is attributed to another person at this stage.

## Before any training
- Instructor requirement reported by student: one main baseline. Cross-check any contradictory Week 9 written instruction and confirm minimum usable dataset size.
- Confirm no other group selected Swahili / identical training dataset.
- Record official Common Voice release, variant, actual download, licensing and exact file hashes.
- Determine audio subset size compatible with lecturer's data-in-repo rule, including potential Git LFS.
- Choose model ID/revision and training hardware; pin separate model-training dependencies after tested installation.
