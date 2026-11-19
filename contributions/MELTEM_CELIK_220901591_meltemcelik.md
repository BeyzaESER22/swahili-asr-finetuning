# Individual contribution — Meltem Çelik

- Student name: Meltem Çelik
- Student ID: 220901591
- GitHub account: `meltemcelik`
- Project role: **T2 — Independent dataset QA: official split boundaries and leakage tests**

## What I did
The T1 dataset (Google FLEURS `sw_ke`, prepared by Beyza Eser) keeps the official FLEURS train/dev/test boundaries, so my task was not to re-split the data but to **independently verify** that the frozen splits are correct and leak-free before any model is evaluated on them.

- Wrote `src/verify_splits.py`, a QA script implemented separately from `src/prepare_fleurs.py` so that T1's claims are re-checked by independent code. It runs eight hard checks:
  - **H1** structure: required columns, no repeated file inside a split
  - **H2** every clip comes from its official FLEURS source split (validation ← dev)
  - **H3** only audit-accepted clips are used; none of the 73 rejected clips appears in a split
  - **H4** the three splits together equal `data/cleaned/manifest.tsv`, row for row
  - **H5** zero cross-split overlap of filename, sentence ID, normalised transcript and audio SHA-256
  - **H6** clip format: 16 kHz, mono, duration = samples / rate, duration ≤ 30 s
  - **H7** clip counts and hours match `data/audit/summary.json`; total ≤ 10 h
  - **H8** every WAV exists, its SHA-256 matches the manifest, and its header is IEEE-float32 mono 16 kHz with the expected sample count (uses its own minimal RIFF parser)

  It also reports review items: near-duplicate transcripts across splits (character-trigram Jaccard ≥ 0.8), gender balance per split, and the speaker-ID limitation.
- Wrote `tests/test_verify_splits.py` (9 tests): the real frozen splits pass, and injected faults are caught (a test clip copied into train, a rejected clip in a split, an edited reference, an hours mismatch, a Git LFS pointer instead of audio, a WAV header check).
- Ran the full QA on the real data and committed the machine-generated report `data/splits/qa_report.json`.
- Documented the commands and results in `README.md` (T2 section) and updated the T2 row in `docs/TEAM_TASKS.md`.

## Verified result (run locally on commit `d8ebe3e` data with Git LFS audio)
| Split | Clips | Hours | Sentence IDs | Female | Male |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,902 | 7.9993 | 918 | 1,023 | 879 |
| validation | 210 | 0.7899 | 133 | 59 | 151 |
| test | 252 | 0.9977 | 162 | 68 | 184 |
| **total** | **2,364** | **9.7869** | | | |

- **All checks H1–H8 PASS.** 2,364 / 2,364 WAV files verified; 0 missing; 0 LFS pointers.
- Cross-split overlap of filenames, sentence IDs, normalised transcripts and audio hashes: **0** for every pair.
- Near-duplicate transcript pairs across splits: **0**.

## Limitations found
- FLEURS TSVs contain **no speaker ID**, so speaker disjointness between splits relies on the official FLEURS split design and cannot be proven by ID in this repository.
- The validation and test splits are **male-dominated** (151/59 and 184/68), inherited from the official FLEURS split. This should be considered when interpreting WER/CER.
- QA checks metadata, hashes and file format; it does not include listening-based review of transcript correctness.

## How to reproduce
```bash
git lfs pull
python src/verify_splits.py --audio-dir data/cleaned/audio
python -m unittest tests.test_verify_splits -v
```

## Evidence
- Pull request: https://github.com/BeyzaESER22/swahili-asr-finetuning/pull/1 (branch `data-splits-quality`, merged into `main`)

## Attribution
The QA script and tests were prepared with AI assistance (Claude), under my direction. I ran them on the real dataset on my own computer, reviewed the output, and committed the work from my own GitHub account. The dataset preparation (T1) is Beyza Eser's work; baselines, fine-tuning and evaluation belong to other team members and are not claimed here.
