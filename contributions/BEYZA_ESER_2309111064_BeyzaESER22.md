# Individual contribution — Beyza Eser

- Student name: Beyza Eser
- Student ID: 2309111064
- GitHub account: `BeyzaESER22`
- Project role: **Repository Architecture, Project Workflow, Dataset Research & Selection, and Dataset Acquisition/Preprocessing & Upload**

## Initial, attributable work (2026-11-17)
- Planned and initialized the GitHub project layout to separate raw data, cleaned data, split manifests, reproducible scripts, configs, tests and results.
- Defined the research workflow, acceptance gates and the mandatory separation of vanilla baselines from any trained or fine-tuned model.
- Researched candidate Swahili ASR corpora: Mozilla Common Voice Scripted Speech 27.0 and Google FLEURS; recorded sources, storage/license constraints and unresolved dataset selection decisions.
- Defined the requirements and added initial scaffolding for data-manifest audit, split handling and WER/CER utility tests.
- Documented team commit/PR practices and repository submission audit criteria based on instructor announcements.

## Completed local dataset preparation (2026-11-17)

I acquired the Google FLEURS `sw_ke` archives and metadata, recorded all six source SHA-256 hashes, and implemented `src/prepare_fleurs.py`. It audited all 3,768 WAVs, measured 16.200350 source hours, verified Float32/mono/16 kHz format and TSV frame counts, logged 73 clips longer than 30 seconds, and produced a deterministic 2,364-clip, 9.786916667-hour subset. I prepared raw and cleaned byte-identical audio trees, frozen manifests, a complete audit table, cleaning log, tests, CC BY 4.0 attribution and Git LFS rules.

The FLEURS seven-field TSV contains sentence IDs but not speaker IDs. I therefore preserved the official train/dev/test boundaries documented by FLEURS as speaker-disjoint and did not claim a speaker count or an independent ID-level overlap proof. Listening-based intelligibility/language review remains outside the completed automated checks.

Remote upload is still pending because the current machine has no GitHub Git credential; the local commit must not be described as pushed until authentication and LFS quota are confirmed.

Another team member will independently review split/leakage QA (T2); model training, baseline implementation and benchmark execution are not my responsibility. The team will separately assign final README preparation and technical PPTX presentation.

## Attribution limitations
Initial scaffold/code/docs were prepared with AI assistance, under my direction and review. Other students' code, datasets, experiments and results are **not** claimed as my work. Work is at the **planning / scaffold** stage; no download, training, speaker validation on real audio or verified experiment results have been performed.

## Evidence
Check this file's Git history and the associated initial project-structure commit(s). Add links to personally authored follow-up PRs and concrete dataset preparation outcomes after completion.
