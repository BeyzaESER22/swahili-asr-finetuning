# Project 3 — Team execution task cards

**Purpose:** A team member should be able to open the relevant card, find the input/source and precise output, implement and verify their part without relying on an oral explanation. Status on 2026-11-17: **only the initial scaffold and dataset research are in the repository**. No data or ASR model has been trained.

## Fixed scope and resource constraints

**Three runs:** E1 main required baseline `openai/whisper-tiny` pretrained (inference only); E2 additional Swahili-capable pretrained wav2vec2-CTC ASR comparator (inference only; a baseline-type model by the instructor's definition); E3 same Whisper-Tiny fine-tuned (one training job). **Two architectures, one required baseline, one comparator and one fine-tuned contribution.**

**Dataset budget: at most 10 hours total**, target 8h train / 1h validation / 1h test, while preventing speaker overlap. Confirm real clip hours; split targets are approximate due to speaker grouping. This does not establish an upper bound on GPU training time. The student confirmed there is no other-group dataset overlap; instructor minimum dataset size remains unspecified.

## Assignments — fill with actual GitHub users before work starts

| Card | Owner (GitHub username) | Status | Required PR |
| --- | --- | --- | --- |
| T0 — Project architecture, flow, candidate dataset research | `BeyzaESER22` | INITIAL DOCUMENTATION DONE; language uniqueness/approval still pending | Initial setup commits |
| T1 — Actual dataset acquisition, raw+cleaned assets and cleaning | UNASSIGNED | NOT STARTED | `data-acquisition-cleaning` |
| T2 — Dataset QA / speaker-disjoint splits and leak tests | UNASSIGNED | NOT STARTED | `data-splits-quality` |
| T3 — Whisper-Tiny main baseline + pretrained wav2vec2-CTC comparator | UNASSIGNED | NOT STARTED | `baseline-implementations` |
| T4 — Fine-tuning/training contribution | UNASSIGNED | NOT STARTED | `whisper-finetuning` |
| T5 — Unified metrics, benchmark, error analysis | UNASSIGNED | NOT STARTED | `benchmark-evaluation` |
| T6 — Final repo audit and technical presentation | SHARED, each member owns own artifacts | NOT STARTED | `submission-audit` |

**Members must not claim completed work solely because files exist**. Every card is considered done only when its evidence and acceptance criteria below are met. Owner fields are intentionally empty.

## T0 — Architecture + workflow + source research (Beyza Eser)
**Inputs:** instructor's repository/grading announcements; ASR/Week 9 announcement; official dataset catalog.
**Outputs:** `README.md`, `docs/PROJECT_PLAN.md`, `docs/DATASET_RESEARCH.md`, `docs/TEAM_WORKFLOW.md`, `docs/BASELINE_REGISTRY.md`, `docs/TEAM_TASKS.md`, structure and contribution file.
**Completed:** initial repo scaffold, workflow gates, Common Voice scripted speech v27.0 Swahili primary candidate with official catalog reference, FLEURS as optional secondary/external reference.
**Pending coordination, not data-cleaning work:** dataset uniqueness confirmed by the user; confirm instructor minimum usable data size and review terms/Git LFS feasibility. Student reports one required baseline, chosen as Whisper-Tiny.
**Boundary:** Beyza does **not** download, clean, split, upload or train data. Data utility source files are initial starter scaffolds handed to T1/T2.

## T1 — Dataset acquisition and real cleaning (future data engineering member)
**Inputs:** selected, versioned Common Voice Swahili `sw` dataset with source-approved real clip files, transcript TSV and `client_id`; Mozilla dataset terms; T0 research notes.
**Steps:**
1. Record the student's completed other-group overlap check; document source page, release number, license/terms, actual file hashes, original sample count, clip hours and speaker counts. Use trustworthy original reference transcripts only.
2. Obtain a representative subset with **no more than 10h of total usable audio**, reserving roughly 8h training, 1h validation, 1h test; instructor must clarify if this is too small for full credit. Avoid cherry-picking only short/easy clips. Measure duration from actual recordings and record sampling code, seed and selection IDs.
3. Preserve **actual original audio + original metadata** under `data/raw/`. Agree on Git LFS constraints before uploading. A Google Drive URL, ignored folder or download-only script **does not satisfy** the instructor's repo rule.
4. Extend `src/audit_manifest.py` beyond basic path/metadata checks: detect duplicate recordings, missing/corrupt audio, empty or misaligned references, abnormal durations/sample rates, clipping/silence/unintelligible segments; do not claim to have checked what the script does not implement.
5. Document **real fixes/removals**, not just file reformatting. Maintain `data/cleaned/cleaning_log.csv` with clip ID, issue, decision, reason and counts before/after. Never fabricate/correct text on model say-so. Retain **actual cleaned audio + aligned transcripts** in `data/cleaned/`.
6. Ensure `src/prepare_data.py` or documented commands reproduce raw -> cleaned outputs without manual Colab uploads. Add tests for key failure cases and demonstrate on the real subset.
**Outputs:** real `data/raw/` assets, `data/cleaned/` assets, cleaning script, cleaning log, provenance report, hashes, measured accepted hours/speakers; updated READMEs.
**Done only when:** team can reproduce the cleaning output from **committed raw data** and report exact genuine removals. No raw/cleaned data has been supplied at scaffold stage.
**Grading risk:** Data Novelty 0 if another group uses identical training dataset; large deductions if raw/cleaned absent or 'cleaning' is only converting file formats.

## T2 — Split and quality/contamination checks (future data engineer)
**Inputs:** cleaned audio + reference transcript + stable speaker IDs from T1.
**Steps:**
1. Complete and test `src/split_manifest.py`; adapt official split policy and GroupShuffleSplit as needed. **Target ~8h train/~1h validation/~1h test**, not necessarily exactly 80/10/10 by row count. Split by real audio duration while enforcing disjoint speakers and <=10h total. The starter only groups by speaker and currently DOES NOT enforce duration targets or the 10h ceiling; extend it before claiming this is done.
2. Freeze train/validation/test file manifests, speaker sets and checksums; prove zero shared `client_id` and zero reused clip IDs across splits.
3. Check near-duplicate audio/text leakage across splits; duplicated transcripts *alone* may be legitimate ASR readings, so review rather than blindly removing.
4. Ensure all baselines and fine-tuned models use **identical test IDs** and frozen reference transcript version. Record sample/hours/speaker counts for every split.
**Outputs:** `data/splits/{train,validation,test}.tsv`, split configuration/seed, test proofs, duplicate/leak report, README commands.
**Done only when:** assertions pass on actual corpus, train/test leak audit is recorded and split files are committed.

## T3 — Main Whisper-Tiny baseline and additional Wav2Vec2-CTC comparator (future ML implementer)
**Inputs:** student's clarification that one baseline is required, agreed model registry in `docs/BASELINE_REGISTRY.md`, frozen T2 data and scoring policy.
**Steps:**
1. Make a model table: method family, concrete checkpoint ID/revision (or full trainable implementation), preprocessing/tokenizer, train/inference command, compute requirements, and baseline status.
2. Run **E1: vanilla pretrained Whisper-Tiny** as the one required main baseline; run **E2: pretrained Swahili-compatible Wav2Vec2-CTC ASR checkpoint** as an extra comparator. Confirm the latter's ASR head and language support before using it. Both are pretrained inference only; in terminology both count as baselines, but E1 is the designated course baseline. Revisit only if newer explicit instructor instructions conflict.
3. For trainable baselines include **real baseline training code**, not just a result or a notebook reference. For pretrained inference baselines, pin model checkpoint and revision and ensure a valid language-compatible ASR head/decoder. A bare wav2vec2 encoder cannot transcribe.
4. Save model config, seed, package versions, original predictions, run logs, weights/checkpoint provenance, WER/CER computed by shared T5 evaluator. No manual results.
**Outputs:** versioned baseline configs/scripts, exact checkpoint IDs, predictions, logs, machine-generated metrics, tests and run commands.
**Done only when:** E1 and E2 have independently repeatable inference and identical test IDs/scorer, and the results are separately labeled main baseline versus additional pretrained comparator.
**Grading risk:** missing baseline loses Correctness + Rigor. 'Qwen3-ASR is advanced' alone is **not** a student contribution.

## T4 — Pretrained-model fine-tuning contribution (future ML implementer)
**Inputs:** frozen T2 data (maximum 10 total hours: ~8h training/~1h validation/~1h test); exact `openai/whisper-tiny` base checkpoint revision; T3 E1 baseline; compute allocation.
**Steps:**
1. Explain full ASR path: waveform -> resampling -> acoustic features -> encoder -> autoregressive decoder/tokenizer -> transcript. Show tensor shapes; training objective/cross-entropy, teacher forcing, gradient flow and decoding constraints.
2. Create `src/train.py` with training hyperparameters, random seeds, optimizer/scheduler, batch sizing, checkpoints, selection using validation set only, logging and resumption.
3. Train on the **training split only**; pick checkpoint on validation. Keep test split untouched until final evaluation. Save base revision and fine-tuned checkpoint separately.
4. Evaluate original pretrained checkpoint **without group training** (vanilla baseline), then fine-tuned checkpoint on identical test utterances; save raw predictions, WER/CER and explain changes.
5. Optional controlled training-size ablation (25/50/100% of **the <=8h training split**) only if compute allows; do not compromise required 3 runs. Pilot measured training throughput first, and constrain the training job to resource budget.
**Outputs:** train script, pinned model ID/revision, configs, training logs, checkpoint evidence, inference command, provenance, validation curves.
**Done only when:** training is runnable from repository, contribution is explicitly distinguished from baseline, and claimed scores are regenerated by code.

## T5 — Shared evaluation and error analysis (future evaluation owner)
**Inputs:** E1 Whisper-Tiny main baseline, E2 wav2vec2-CTC comparator and E3 fine-tuned Whisper-Tiny predictions, all with utterance IDs and immutable test reference transcripts.
**Steps:**
1. Finalize text normalization policy **before seeing test scores**; apply same rule to both references and predictions. The current `src/metrics.py` only uses NFC/lowercase/space collapse: review whether punctuation/number treatment needs documented adjustments for all experiments.
2. Explain **WER = (S+D+I)/N** and **CER = character edits/reference character count**. WER is task-level primary; CER aids inspection of Swahili morphology/spelling. Do not use classification accuracy as substitute.
3. Add unit tests with hand-checked substitutions, insertions, deletions, blank hypotheses, mismatched/empty references, and normalization cases.
4. Generate `results/predictions/<model>.tsv` per model (ID, reference, hypothesis), `results/metrics.csv` directly from code, and reproducible subgroup/error reports (speaker, utterance length, noisy clips when labels available).
5. Check test ID/order equality, metric denominators, model-version provenance, and runtime settings; do not hand-edit scores or copy numbers from Colab into README.
**Outputs:** frozen scoring script, tests, machine-generated benchmark table, errors and sample audit.
**Done only when:** new teammate can reproduce the final table from committed predictions and script.

## T6 — Submission / presentation (all members)
**Steps:**
1. Each member creates `contributions/<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` with accurate, personally verifiable work and makes commits **from their own account**.
2. Run all tests; pin every package version; record dataset/checkpoint revisions and seeds; make README explain every root file and folder.
3. Ensure actual raw and cleaned data (not links/scripts), licenses, training scripts, baselines and result provenance have been committed **before** deadline. Check public URL in incognito.
4. Upload PPTX and full `https://github.com/BeyzaESER22/swahili-asr-finetuning` link as assigned. Prime submitter submits once.
5. Prepare to describe ASR pipeline, model architecture, fine-tuning, dataset cleaning, bias/leakage, metrics and result differences without reading. Instructor's earlier best pace is **under two minutes**; verify whether limit applies per speaker/group.
**Done only when:** `docs/GRADING_AUDIT.md` is fully evidenced and the exact graded pre-deadline main commit is identified.

## How a contributor should proceed
- Read *this card*, then the linked source files and README.
- Create a branch for your card; implement only requirements claimed as completed.
- Run relevant tests and end-to-end commands. Document inputs, outputs, and known limitations.
- Update your **own** contribution file, README and PR with exact test command/results.
- Do not mark any unfinished checkpoint, dataset or model as completed.
