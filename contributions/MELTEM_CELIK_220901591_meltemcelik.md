# Individual contribution — Meltem Çelik

- Student name: Meltem Çelik
- Student ID: 220901591
- GitHub account: `meltemcelik`
- Project role: **Independent Dataset Quality Assurance, Split-Boundary Verification and Leakage Testing (T2)**

## Scope of my work (2026-11-19)
- Took over the T2 handoff after the FLEURS `sw_ke` dataset was prepared in T1.
- Reviewed the T1 split policy and confirmed that the data should **not** be re-split: the FLEURS TSVs carry no speaker IDs, so the official train/dev/test boundaries (documented by FLEURS as speaker-disjoint) are the correct basis. The generic `src/split_manifest.py` was therefore not used for FLEURS.
- Designed an independent verification step so that T1's claims are re-checked by code written separately from `src/prepare_fleurs.py`, before any baseline or fine-tuned model is evaluated on the splits.

## Completed split QA and leakage audit (2026-11-19)

I implemented `src/verify_splits.py`. It re-reads the frozen manifests in `data/splits/`, the cleaned manifest, the cleaning log and `data/audit/summary.json`, and runs eight hard checks: required structure and no repeated files (H1); every clip taken only from its official source split, with validation drawn from `dev` (H2); only audit-accepted clips and none of the 73 rejected clips (H3); the three splits together equal to `data/cleaned/manifest.tsv` row for row (H4); zero cross-split overlap of filename, sentence ID, normalised transcript and audio SHA-256 (H5); 16 kHz mono format, duration consistent with sample count and at most 30 seconds (H6); clip counts and hours matching the T1 summary within the 10-hour budget (H7); and, on the real Git LFS audio, file existence, SHA-256 agreement and an independently parsed IEEE-Float32/mono/16 kHz WAV header with the expected sample count (H8). It also reports review items: near-duplicate transcripts across splits (character-trigram Jaccard ≥ 0.8), gender balance and the speaker-ID limitation.

I ran the script on my own computer against the committed manifests and the downloaded LFS audio. **All checks H1–H8 passed.** All 2,364 WAV files were verified with no missing files and no unresolved LFS pointers. The splits contain 1,902 / 210 / 252 clips and 7.9993 / 0.7899 / 0.9977 hours (9.7869 h in total). Cross-split overlap was zero for every pair and every key, and no near-duplicate transcript pair was found. The machine-generated result is committed as `data/splits/qa_report.json`.

I added `tests/test_verify_splits.py` (9 tests). They confirm that the real frozen splits pass and that injected faults are detected: a test clip copied into train, a rejected clip inside a split, an edited reference transcript, an hours mismatch against the summary, a Git LFS pointer in place of audio, and malformed WAV headers. I documented the commands and results in the README T2 section and updated the T2 row in `docs/TEAM_TASKS.md`.

The validation and test splits are male-dominated (151 male / 59 female and 184 male / 68 female). This is inherited from the official FLEURS split, not introduced by selection, and should be considered when WER/CER results are interpreted.

Baseline inference, fine-tuning and benchmark evaluation are not my responsibility and are assigned to other team members (T3–T5). Final README preparation and the technical PPTX presentation remain shared deliverables.

## Attribution limitations
The QA script and its tests were prepared with AI assistance, under my direction and review; I executed them on the real dataset myself and committed the work from my own GitHub account. The dataset acquisition and preparation (T1) is Beyza Eser's work and is **not** claimed here. Speaker disjointness cannot be proven by ID because FLEURS exposes no speaker identifiers; this check relies on the official FLEURS split design. The QA covers metadata, hashes and file format only; listening-based review of transcript correctness was not performed.

## Evidence
- Pull request [#1](https://github.com/BeyzaESER22/swahili-asr-finetuning/pull/1) from branch `data-splits-quality`, merged into `main` on 2026-11-19.
- `src/verify_splits.py`, `tests/test_verify_splits.py`, `data/splits/qa_report.json` and this file's Git history.
- Reproduce: `git lfs pull`, then `python src/verify_splits.py --audio-dir data/cleaned/audio`.
