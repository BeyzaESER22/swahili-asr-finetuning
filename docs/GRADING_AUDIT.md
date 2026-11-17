# Grading and repository audit (from instructor announcements)

This document is the **acceptance checklist**. It does not claim the instructor assessed this repository or that future points are guaranteed.

| Instructor concern | What must exist in the submitted repo | Current status |
| --- | --- | --- |
| One main required course baseline + optional comparator | E1 vanilla Whisper-Tiny as main baseline, E2 pretrained Wav2Vec2-CTC ASR comparator, same test IDs and scoring | Student confirmed one required baseline; both NOT RUN |
| One individually named contributions file per member | `contributions/NAME_STUDENTID_GITHUBUSERNAME.md`, own changes visible in Git history | Beyza file exists; future members pending |
| Raw AND cleaned datasets within repo | Real audio + metadata in both locations, cleaning script and log, valid permitted Git/LFS packaging | Prepared and committed locally with LFS; remote push/fresh-clone verification pending |
| Each member commits from their account | Authored PRs/commits per actual participant; co-author trailers alone not enough | Beyza initial commit only |
| Repo clean / no unexplained files | README describes all tracked root paths; remove temp/unnecessary files | Scaffold documented; repeat before deadline |
| Code license separate from dataset | Root LICENSE for own scripts, dataset license/terms recorded | MIT code license plus `data/LICENSE.md` CC BY 4.0 attribution/change notice |
| Metrics justified | WER primary, CER secondary, explicit formulas/normalization, test scripts | Documented concept; real-data evaluation pending |
| No irrelevant engineering | Prefer data quality, model code and benchmark to backend, release pipelines, database tools | Scaffold aligns; monitor future PRs |
| Fully reproducible | Fixed seeds, pinned versions, configs, checkpoints or exact reproducible training route, generated final numbers | Dataset pipeline seed/config/evidence complete locally; no model training yet |
| Notebooks and Colab | All notebooks/scripts committed, no manual uploads | No notebooks yet |
| Tests test actual ML/data work | Dataset integrity, split leakage, WER/CER, baseline prediction and train smoke tests | Dataset parser/WAV/selection tests run; model tests pending |
| Unique dataset | Document group-wide dataset coordination | User confirmed no other-group overlap; documentary evidence pending |
| Actual cleaning, not reformatting | Verified real data problems, decisions, removal log, enough usable speech | 3,768 WAV audit; 73 >30s exclusions; 9.786916667h frozen selection |
| Public repo and deadline | Public incognito link, complete PPTX + GitHub URL, last commit before deadline | Public repo exists; submission pending |

## Final documentation and slide quality checks

- [ ] Assign a final README editor and an independent reviewer. Reconcile every file description, run command, model checkpoint and WER/CER number against committed artifacts.
- [ ] Commit and submit `presentation/project3.pptx`; assign speakers, explain technical differences and rehearse the lecturer's presentation time constraint.
- [ ] Reflect Beyza's updated ownership accurately: local source acquisition, cleaning code/log and data artifacts completed; remote LFS upload not yet verified.
- [ ] Keep the final README and presentation tasks separate from model-training task completion.

## Critical quality gates
- **G0:** One required baseline confirmed by the student; designated E1 vanilla Whisper-Tiny; user confirmed dataset uniqueness. Reconcile if newer written instructor instructions conflict.
- **G1:** Dataset owner commits real raw and cleaned audio+metadata and measured cleaning evidence, **<=10h total audio (~8h train, ~1h val, ~1h test)**, with instructor minimum accepted.
- **G2:** Frozen official-boundary split, no filename/text overlap, paper-backed speaker-disjoint policy, explicit absence of speaker IDs, metric unit tests.
- **G3:** E1 main vanilla Whisper-Tiny baseline and E2 additional pretrained wav2vec2-CTC ASR comparator reproducible on same test. Both are technically baseline-type experiments.
- **G4:** E3 fine-tuned Whisper-Tiny contribution trained independently once, evaluated against E1 (and E2) on the same test set.
- **G5:** Final benchmark generated directly from code/predictions; audit each filename and contribution commit.
- **G6:** Incognito public-link verification and deadline-stamped commit; PPTX submitted.

## Instructor rubric (initial announcement)
Presentation: 50% of total project score — Technical Depth 50% of presentation, Fluency 25%, Pace 25%; questions may adjust technical depth.
Repository: 50% of total project score — Correctness 20% of repository, Data Novelty 30%, Task Novelty 30%, Rigor 20%.
These weights reflect the *initial announcement*, and a later project-specific rule may supersede them.

**Interpretation for this project:** Low-resource language or domain-specific audio may earn additional credit **only** with sufficient authentic references and appropriate evidence, not by labeling Swahili automatically low-resource. Model sophistication without training is not a contribution.

## Final PR review questions
1. Can I run your branch without manual cloud file uploads?
2. Does every displayed number come from a committed executable path?
3. Does the code enforce the same reference/transcript normalization and exact test IDs?
4. Do the tracked artifacts actually exist, or are they only README promises?
5. Can I verify what each member actually contributed through account commits?
6. Does the repository contain both real raw and cleaned data, compliant with its licenses and size constraints?
