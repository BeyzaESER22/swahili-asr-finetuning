# Grading and repository audit (from instructor announcements)

This document is the **acceptance checklist**. It does not claim the instructor assessed this repository or that future points are guaranteed.

| Instructor concern | What must exist in the submitted repo | Current status |
| --- | --- | --- |
| All course/project baselines in vanilla form, training code included | Exact Week 9 list, one runnable baseline per required method, same test IDs and scoring | PENDING Week 9 list; NOT IMPLEMENTED |
| One individually named contributions file per member | `contributions/NAME_STUDENTID_GITHUBUSERNAME.md`, own changes visible in Git history | Beyza file exists; future members pending |
| Raw AND cleaned datasets within repo | Real audio + metadata in both locations, cleaning script and log, valid permitted Git/LFS packaging | NOT COMPLETED; placeholder README files only |
| Each member commits from their account | Authored PRs/commits per actual participant; co-author trailers alone not enough | Beyza initial commit only |
| Repo clean / no unexplained files | README describes all tracked root paths; remove temp/unnecessary files | Scaffold documented; repeat before deadline |
| Code license separate from dataset | Root LICENSE for own scripts, dataset license/terms recorded | Initial MIT code license exists; dataset files not yet acquired |
| Metrics justified | WER primary, CER secondary, explicit formulas/normalization, test scripts | Documented concept; real-data evaluation pending |
| No irrelevant engineering | Prefer data quality, model code and benchmark to backend, release pipelines, database tools | Scaffold aligns; monitor future PRs |
| Fully reproducible | Fixed seeds, pinned versions, configs, checkpoints or exact reproducible training route, generated final numbers | Initial utilities only; no training yet |
| Notebooks and Colab | All notebooks/scripts committed, no manual uploads | No notebooks yet |
| Tests test actual ML/data work | Dataset integrity, split leakage, WER/CER, baseline prediction and train smoke tests | Starter unit tests available; **not run here** |
| Unique dataset | Document group-wide dataset coordination | NOT CONFIRMED |
| Actual cleaning, not reformatting | Verified real data problems, decisions, removal log, enough usable speech | NOT COMPLETED |
| Public repo and deadline | Public incognito link, complete PPTX + GitHub URL, last commit before deadline | Public repo exists; submission pending |

## Critical quality gates
- **G0:** Instructor's exact Week 9 mandatory list + group dataset uniqueness confirmed.
- **G1:** Dataset owner commits real raw and cleaned audio+metadata and measured cleaning evidence, size approved.
- **G2:** Frozen speaker-disjoint split, no overlap, metric unit tests.
- **G3:** Every mandatory vanilla baseline, **including training code when needed**, reproducible.
- **G4:** Fine-tuned contribution trained independently and evaluated vs vanilla on same test set.
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
