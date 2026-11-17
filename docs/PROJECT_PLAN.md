# Project plan — Swahili automatic speech recognition

**Owner for initial architecture, workflow and dataset research:** Beyza Eser.
**Role boundary:** Beyza's task is to identify/document/select a dataset candidate and organize the project. Downloading, actual cleaning, raw/cleaned repository upload, audio quality auditing and real-data splits belong to **other members (unassigned until agreed)**. The presence of starter utility scripts does not imply Beyza owns their full implementation.
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
| 3. Required baselines | All instructor-listed vanilla implementations and run scripts | Baselines completed on one held-out test split |
| 4. Contribution | Reproducible training/fine-tuning script and checkpoint provenance | Separately identify original training contribution |
| 5. Comparison | WER/CER CSV, per-utterance outputs, error analysis | Every number is regenerated from committed code |
| 6. Submission | Per-person contribution histories, repo audit, PPTX | Public clean repo and on-time commits; slide timing checked |

## Proposed experimental matrix — subject to lecturer baseline requirements
| Track | Model family | Treatment | Status |
| --- | --- | --- | --- |
| Vanilla | Multilingual Whisper, exact checkpoint TBD | Zero additional group training | Planned |
| Trained contribution | Same Whisper checkpoint | Swahili supervised fine-tuning | Planned |
| Vanilla comparator | Language-compatible wav2vec2-CTC ASR model (actual ASR head), exact checkpoint TBD | No additional group training | Optional until Week 9 baseline confirmation |
| Optional ablation | Fine-tuned Whisper | 25% / 50% / 100% training data, fixed validation/test | Planned only if feasible |

Do not call an encoder-only HuBERT/wav2vec2 checkpoint an ASR baseline: it requires a transcription head. Advanced pretrained models are still **baselines** without group training.

## Metrics / statistical interpretation
WER is the primary ASR task metric, not text-classification accuracy. CER helps with agglutinative/morphological spelling errors. Compute both with one frozen normalization policy, identical reference/hypothesis handling, and report sample counts. Inspect specific substitutions/deletions/insertions. Use latency only under consistent hardware, batch size and decoding parameters.

## Threats to validity
Crowdsourced transcript errors, text normalization bias, cross-split speaker leakage, dialect imbalance, near-duplicate utterances, dataset license restrictions, compute limitations, pretrained model exposure to data, and mismatch between Common Voice and external FLEURS domains.

## Ownership boundary
Beyza: repository structure, methodology/project flow, source evaluation, dataset candidate selection/provenance documentation, and initial helper scaffolding (not implementation/validation of cleaning or dataset downloads).
Other team members: dataset download/cleaning, actual audio/raw/cleaned uploads, real-world splitting/QA, model baselines, fine-tuning, evaluation and reproducibility; assign actual names only after agreement. Each member will create their **own** contribution file and commits. No contribution is attributed to another person at this stage.

## Before any training
- Obtain instructor's exhaustive Week 9 baseline list and minimum usable dataset size.
- Confirm no other group selected Swahili / identical training dataset.
- Record official Common Voice release, variant, actual download, licensing and exact file hashes.
- Determine audio subset size compatible with lecturer's data-in-repo rule, including potential Git LFS.
- Choose model ID/revision and training hardware; pin separate model-training dependencies after tested installation.
