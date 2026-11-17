# Collaboration, reproducibility and submission workflow

## Ownership and task handoff
- Beyza Eser: initial architecture, project flow and dataset source research only; does **not** own data download, cleaning, upload, split generation, baseline training, fine-tuning or evaluation execution.
- Dataset engineering, baselines, fine-tuning and evaluation must be assigned to actual individual members in `docs/TEAM_TASKS.md` and each member's own contribution file. There are no invented owner names.
- Each implementation task has a definition of done and expected files. Read `docs/TEAM_TASKS.md` and `docs/GRADING_AUDIT.md` before starting.

## Git rules
1. Every student clones the **public** GitHub repository using their **own account**, creates a feature branch, makes substantive commits themselves and opens a pull request. Do not push the whole team's work under one person's identity.
2. Every actual member creates their own `contributions/FIRST_LAST_STUDENTID_GITHUBUSERNAME.md` (not `contributors/`). Describe actual file paths, experiments, code and linked commit/PR evidence.
3. Keep main runnable. Review changes before merge; record the owner of new code or dataset assets.
4. Delete temp files, `__pycache__`, `.pyc`, unneeded weights and old exports; explain any necessary files in README.

## Definition of done for every experiment
- Exact dataset version, split IDs, cleaning summary and artifact hashes are known.
- Exact pretrained model ID **and revision** are documented, separately from architecture.
- Exact environment package versions, seed and compute details are recorded.
- Baseline script (including training for trained vanilla models), fine-tuning script if relevant, config, checkpoint provenance and scoring script are committed.
- Test-set predictions and reported WER/CER are generated from code and retained; README does not contain manually invented metrics.
- Unit/smoke tests cover preprocessing and evaluation; train/test speaker overlap is zero.
- Run commands work without manual Colab uploads.

## Before submission
- Verify each member has own commits and correctly named contribution file.
- Verify raw and cleaned datasets are **inside repo** and their licensing/storage arrangements comply.
- Compare README against exact tracked files and tested commands.
- Confirm public GitHub URL from incognito and submit full https://github.com/... link.
- Submit PPTX and URL before deadline; lecturer grades last **pre-deadline** commit.
- Presentation: explain architecture, tokenizer/decoder, ASR objective/loss, fine-tuning gradients, preprocessing, metrics and errors without reading. Instructor's earlier pace scale rewards under two minutes; verify per-person vs group timing.
