# Prepared FLEURS `sw_ke` data

This directory records the reproducible preparation of a maximum-10-hour
Swahili ASR subset from the six source files placed in
`/Users/beyza/Desktop/Audio` on 2026-11-17.

## Verified result

| Split | Selected clips | Selected duration |
| --- | ---: | ---: |
| train | 1,902 | 7.999300 h |
| validation (source `dev`) | 210 | 0.789900 h |
| test | 252 | 0.997716667 h |
| **total** | **2,364** | **9.786916667 h** |

The full downloaded `sw_ke` material contains 3,768 clips and 16.200350 h of
audio. All WAVs were readable IEEE Float32, mono, 16 kHz files and their frame
counts matched the TSV `num_samples` values. Seventy-three recordings were
excluded from the eligible pool because they exceed 30 seconds (72 train, one
dev). No high-silence or high-clipping threshold was triggered. These checks
do not substitute for listening-based intelligibility, language-ID, or
transcript-correctness review.

## Speaker policy

The seven-field source TSVs do not contain a speaker ID. Their first field is
a sentence ID and repeats when different recordings read the same prompt.
Therefore this project does not claim an independently regenerated
speaker-disjoint split or a measured speaker count. It preserves the official
FLEURS train/dev/test boundaries, which the FLEURS paper documents as having
disjoint speakers, and samples only within those boundaries.

## Files

- `raw/manifest.tsv`: selected source metadata before derived audit fields.
- `cleaned/manifest.tsv`: selected metadata plus measured waveform properties
  and hashes; waveform bytes are unchanged.
- `cleaned/cleaning_log.tsv`: every rejected or review-flagged recording.
- `splits/*.tsv`: frozen model-facing manifests.
- `audit/full_audit.tsv`: all 3,768 source records and their audit outcome.
- `audit/summary.json`: source checksums, counts, hours, policy, and limits.
- `LICENSE.md`: required FLEURS attribution and modification notice.

Audio paths under `raw/audio/` and `cleaned/audio/` are tracked by Git LFS.
The cleaned files are byte-identical to the selected raw files; local
preparation uses hard links where supported to avoid duplicating disk space.

## Reproduce

```bash
python src/prepare_fleurs.py \
  --input-dir /path/to/FLEURS-sw_ke-source-files \
  --output-root data \
  --seed 42 \
  --extract-audio
```

The input directory must contain `train/dev/test.tsv` and
`train/dev/test.tar`. The script rejects requested targets above ten total
hours.
