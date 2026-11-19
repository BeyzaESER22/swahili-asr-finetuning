# Project 3 — Team execution task cards

**Purpose:** A team member should be able to open the relevant card, find the input/source and precise output, implement and verify their part without relying on an oral explanation. Status on 2026-11-17: **only the initial scaffold and dataset research are in the repository**. No data or ASR model has been trained.

## Fixed scope and resource constraints

**Three runs:** E1 main required baseline `openai/whisper-tiny` pretrained (inference only); E2 additional Swahili-capable pretrained wav2vec2-CTC ASR comparator (inference only; a baseline-type model by the instructor's definition); E3 same Whisper-Tiny fine-tuned (one training job). **Two architectures, one required baseline, one comparator and one fine-tuned contribution.**

**Dataset budget: at most 10 hours total**, target 8h train / 1h validation / 1h test, while preventing speaker overlap. Confirm real clip hours; split targets are approximate due to speaker grouping. This does not establish an upper bound on GPU training time. The student confirmed there is no other-group dataset overlap; instructor minimum dataset size remains unspecified.

## Assignments — fill with actual GitHub users before work starts

| Card | Owner (GitHub username) | Status | Required PR |
| --- | --- | --- | --- |
| T0 — Project architecture, flow, candidate dataset research | `BeyzaESER22` | INITIAL DOCUMENTATION DONE; language uniqueness/approval still pending | Initial setup commits |
| T1 — Actual dataset acquisition, raw+cleaned assets and cleaning | `BeyzaESER22` | PREPARED AND COMMITTED LOCALLY; REMOTE LFS PUSH PENDING | `data-acquisition-cleaning` |
| T2 — Dataset QA / official-boundary and leak tests | `meltemcelik` | DONE: `src/verify_splits.py` + tests; all checks H1–H8 PASS on real data incl. 2,364 WAV hashes; `data/splits/qa_report.json` | `data-splits-quality` |
| T3 — Whisper-Tiny main baseline + pretrained wav2vec2-CTC comparator | `berkotun` | DONE: `src/run_baseline.py`, `src/score_predictions.py` + tests; E1 `openai/whisper-tiny`, E2 `facebook/mms-1b-all` (`swh`); results in `results/metrics.csv` | `baseline-implementations` |
| T4 — Fine-tuning/training contribution | UNASSIGNED | NOT STARTED | `whisper-finetuning` |
| T5 — Unified metrics, benchmark, error analysis | UNASSIGNED | NOT STARTED | `benchmark-evaluation` |
| T6 — Final README verification and release-ready documentation | SHARED; final editor/reviewer TBD | NOT STARTED | `final-readme` |
| T7 — Technical PPTX presentation, rehearsals and submission | SHARED; slide owner/presenter TBD | NOT STARTED | `presentation` |
| T8 — Final repo audit, on-time submission | SHARED; prime submitter TBD | NOT STARTED | `submission-audit` |

**Members must not claim completed work solely because files exist**. Every card is considered done only when its evidence and acceptance criteria below are met. Owner fields are intentionally empty.

## T0 — Architecture + workflow + source research (Beyza Eser; initial work documented)
**Inputs:** instructor's repository/grading announcements; ASR/Week 9 announcement; official dataset catalog.
**Outputs:** `README.md`, `docs/PROJECT_PLAN.md`, `docs/DATASET_RESEARCH.md`, `docs/TEAM_WORKFLOW.md`, `docs/BASELINE_REGISTRY.md`, `docs/TEAM_TASKS.md`, structure and contribution file.
**Completed:** initial repo scaffold, workflow gates and final FLEURS `sw_ke` selection with source/license references.
**Pending coordination, not data-cleaning work:** dataset uniqueness confirmed by the user; confirm instructor minimum usable data size and review terms/Git LFS feasibility. Student reports one required baseline, chosen as Whisper-Tiny.
**Updated boundary:** Beyza owns acquiring/cleaning and uploading real data plus preparation scripts. The full local preparation ran on real audio; T1 remains remotely incomplete only until the LFS objects are authenticated, pushed and fresh-clone verified. T2 independent QA and T3–T5 model/benchmark work belong to other members.

## T1 — Dataset acquisition and real cleaning (assigned to Beyza Eser; locally prepared)
**Inputs:** Google FLEURS `sw_ke` train/dev/test WAV archives and seven-field TSVs; FLEURS CC BY 4.0 terms; T0 research notes.
**Steps:**
1. Record the student's completed other-group overlap check; document source page, license/terms, actual file hashes, original sample count and clip hours. Speaker counts are unavailable because this TSV release omits speaker IDs. Use trustworthy original reference transcripts only.
2. Obtain a representative subset with **no more than 10h of total usable audio**, reserving roughly 8h training, 1h validation, 1h test; instructor must clarify if this is too small for full credit. Avoid cherry-picking only short/easy clips. Measure duration from actual recordings and record sampling code, seed and selection IDs.
3. Preserve **actual original audio + original metadata** under `data/raw/`. Agree on Git LFS constraints before uploading. A Google Drive URL, ignored folder or download-only script **does not satisfy** the instructor's repo rule.
4. Use `src/prepare_fleurs.py` for the full metadata/WAV audit, duration cap and extraction. It checks structural/format/sample-count/signal conditions and logs clipping/silence review thresholds; it does not claim listening-based intelligibility review.
5. Document real exclusions in `data/cleaned/cleaning_log.tsv`. Never fabricate/correct text on model say-so. Retain actual cleaned audio + aligned transcripts in `data/cleaned/`.
6. Reproduce raw -> cleaned outputs without manual Colab uploads and run the test suite.
**Outputs:** real `data/raw/` assets, `data/cleaned/` assets, cleaning script/log, provenance report, hashes and measured accepted hours; updated READMEs.
**Done only when:** the LFS commit is pushed and a fresh clone retrieves the real WAVs. Local preparation and commit are complete; remote verification remains pending.
**Grading risk:** Data Novelty 0 if another group uses identical training dataset; large deductions if raw/cleaned absent or 'cleaning' is only converting file formats.

## T2 — Split and quality/contamination checks (future data engineer)
**Inputs:** cleaned audio, reference transcripts, official split-boundary provenance and T1 audit evidence. Stable speaker IDs are unavailable in this TSV release.
**Steps:**
1. Independently rerun `src/prepare_fleurs.py` and confirm the 7.999300/0.789900/0.997716667-hour manifests and <=10h ceiling. Do not use the generic `src/split_manifest.py` for FLEURS because it requires unavailable `client_id` values.
2. Verify zero reused filenames and zero normalized-transcript overlap across official boundaries; review FLEURS paper evidence for speaker-disjoint source splits. Explicitly record that direct speaker-ID proof is unavailable.
3. Check near-duplicate audio/text leakage across splits; duplicated transcripts *alone* may be legitimate ASR readings, so review rather than blindly removing.
4. Ensure all baselines and fine-tuned models use identical test IDs and frozen reference transcript version. Record clip/hour counts and the speaker-ID limitation for every split.
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

## T6 — Finalize README.md (shared; final editor and independent reviewer TBD)

**Inputs:** completed T1–T5 PRs and reproducible experiment artifacts.
**Steps:**
1. Replace all scaffold-only descriptions and TODOs with the *actual* dataset version, clip hours, speaker/split counts, cleaning reasons, audio license and Git/LFS details.
2. Explain every root file and directory, exact model architecture vs checkpoint IDs/revisions, training/evaluation entry points, pinned environment and seeds.
3. Include a table automatically generated from committed results (WER/CER for E1, E2, E3). **Do not manually invent or transcribe scores from a notebook**; link a reproducible command and inputs.
4. Document real caveats, negative results, resource costs, limitations and team contribution links; ensure install/run scripts work without manual cloud uploads.
5. Ask a different team member to **clone and verify README commands end to end**, fix discrepancies and document the verification commit.
**Output:** final `README.md` matching repository, `docs/GRADING_AUDIT.md` evidence, reviewer verification report / PR.
**Done only when:** reviewer runs installation/tests/necessary experiment reproduction, and every link, output and file description is accurate.
**Grading risk:** missing explanation of files or manually edited results costs correctness/rigor.

## T7 — Prepare PPTX technical presentation (shared; owners TBD)
**Inputs:** actual T1–T5 experiments and generated benchmark results, NOT forecasts or fabricated metrics.
**Steps:**
1. Prepare `presentation/project3.pptx` and commit it to GitHub; assign each member concrete slide and speaking responsibilities.
2. Include dataset rationale and true cleaning examples; the 3 experiments (E1 Whisper-Tiny main vanilla baseline, E2 additional Wav2Vec2-CTC pretrained comparator, E3 fine-tuned Whisper-Tiny); frozen split; training loss; WER/CER and actual error examples.
3. For technical depth, explain acoustic features/encoder/decoder, tokenizer, CTC vs autoregressive decoding and fine-tuning/backpropagation, plus fair benchmark details. Use real model IDs, revision info and measured results.
4. Write concise presenter notes but **do not read** from the slides. Rehearse to the instructor's time limit; earlier announcement awards best pace below two minutes. Confirm per-speaker vs group interpretation.
5. Rehearse Q&A on normalization, speaker leakage, why fine-tuning is the contribution, and why E2 is an extra pretrained comparator, not a new model.
**Output:** committed editable PPTX, assigned speaking roles, rehearsal checklist and final presentation upload.
**Done only when:** slide claims reconcile with `results/`, all speakers can explain their parts, timing is met, PPTX is submitted.
**Grading risk:** copied/unexplained results, reading text, or exceeding time.

## T8 — Submission audit (shared; prime submitter TBD)
**Steps:**
1. Each member creates `contributions/<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` with accurate, personally verifiable work and makes commits **from their own account**.
2. Run all tests; pin every package version; record dataset/checkpoint revisions and seeds; make final README explain every root file and folder.
3. Confirm real raw and cleaned data (not links/scripts), correct licenses, reproducible baseline/training results, and PR evidence have been committed **before the deadline**.
4. Verify the GitHub repository is public in an incognito window; submit the full URL and PPTX on time through the required course platform. Prime submitter submits once.
**Done only when:** the final `docs/GRADING_AUDIT.md` is evidenced and pre-deadline main commit SHA is recorded.

## How a contributor should proceed
- Read *this card*, then the linked source files and README.
- Create a branch for your card; implement only requirements claimed as completed.
- Run relevant tests and end-to-end commands. Document inputs, outputs, and known limitations.
- Update your **own** contribution file, README and PR with exact test command/results.
- Do not mark any unfinished checkpoint, dataset or model as completed.
