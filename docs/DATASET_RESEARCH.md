# Dataset decision and verified preparation — FLEURS `sw_ke`

Updated 2026-11-17 from the actual files under `/Users/beyza/Desktop/Audio`.
This document separates measured evidence from source documentation and from
remaining limitations.

## Final dataset decision

- Dataset: Google FLEURS, Swahili–Kenya configuration `sw_ke`.
- Source card: <https://huggingface.co/datasets/google/fleurs>.
- Paper: <https://arxiv.org/abs/2205.12446>.
- Data license: CC BY 4.0; repository attribution/change notice:
  `data/LICENSE.md`.
- Project cap: at most ten total selected hours; target approximately 8/1/1.
- Selection: seed 42, preserve official split boundaries, select complete
  sentence-ID groups in deterministic hash order without exceeding each cap.

Common Voice was considered earlier but is not the selected dataset. Documents
and configs must not describe it as the primary corpus.

## Measured source evidence

| Source split | Clips | Real WAV duration |
| --- | ---: | ---: |
| train | 3,070 | 13.471050 h |
| dev | 211 | 0.801683333 h |
| test | 487 | 1.927616667 h |
| **total** | **3,768** | **16.200350 h** |

The preparation script opened every WAV payload, rather than inferring duration
from file size or clip count. For every file it checked RIFF structure, IEEE
Float32 encoding, mono channel count, 16 kHz sample rate, finite samples,
nonzero energy and equality between actual frame count and TSV `num_samples`.
It also computed peak, RMS, silence ratio, clipping ratio and per-WAV SHA-256.
All files passed those structural checks.

The audit found 73 recordings longer than 30 seconds: 72 train and one dev.
They were excluded from the eligible pool and remain visible in
`data/cleaned/cleaning_log.tsv`. No clip crossed the configured 98% silence or
1% clipping review thresholds. This is waveform evidence, not proof that every
utterance is intelligible, correctly labeled or linguistically aligned.

All six original source hashes, byte sizes and exact results are machine-written
to `data/audit/summary.json`; per-record evidence is in
`data/audit/full_audit.tsv`.

## Frozen maximum-10-hour subset

| Model split | Source boundary | Clips | Sentence IDs | Duration |
| --- | --- | ---: | ---: | ---: |
| train | train | 1,902 | 918 | 7.999300 h |
| validation | dev | 210 | 133 | 0.789900 h |
| test | test | 252 | 162 | 0.997716667 h |
| **total** |  | **2,364** |  | **9.786916667 h** |

Validation is below one hour because the single >30-second dev file is rejected
and the complete eligible dev split totals 0.7899 h. The selection does not
fill the unused budget with test material, because doing so would distort the
agreed 8/1/1 intent. The total remains safely below ten hours.

## Speaker evidence: important limitation

The first source TSV field is a sentence/prompt ID, not a speaker ID; repeated
values have identical transcripts recorded by different voices. The current
seven-field TSV therefore cannot support a speaker count or an independent
speaker-ID intersection test. The FLEURS paper states that the official
train/dev/test sets use disjoint speakers. This preparation preserves those
boundaries exactly and only subsamples within each boundary.

Confirmed source claim: official boundaries are speaker-disjoint. Independently
verified repository claim: no audio filename or normalized transcript crosses
the three source splits. Unavailable proof: direct speaker-ID comparison.
These claims must not be conflated.

## Cleaning and preprocessing scope

Implemented:

- literal seven-field TSV parsing without CSV quote corruption;
- full tar-to-TSV membership and duplicate-filename audit;
- full WAV/header/frame/signal inspection and SHA-256;
- auditable rejection and review reasons;
- deterministic duration-capped manifests;
- selected raw audio extraction and byte-identical cleaned audio tree;
- Git LFS tracking, CC BY attribution, tests and reproducible CLI.

Not implemented or claimed:

- listening to all 3,768 recordings;
- automatic or human language identification;
- human transcript correction/alignment review;
- denoising, silence trimming or resampling;
- independent speaker-ID proof or speaker counts.

No model prediction is used as reference text. Duplicated normalized text within
a split is expected because FLEURS collects multiple readings of a prompt; it
is not blindly removed as duplicate audio.

## GitHub/LFS assessment

GitHub blocks ordinary Git objects over 100 MiB and recommends Git LFS for
binary data. GitHub Free and Pro currently include 10 GiB LFS storage and
10 GiB monthly bandwidth; the selected unique audio occupies about 2.1 GiB
locally. Raw and cleaned paths have identical content hashes, so LFS content
addressing should not store duplicate waveform objects. Each WAV is below the
plan's per-file limit.

The size and license are therefore technically compatible with Git LFS, subject
to the repository owner's *remaining* quota and budget settings. The current
Codex environment has no usable GitHub Git credential, so remote upload is not
confirmed. Do not claim that the data is on GitHub until `git push` and `git lfs
push` succeed and a fresh clone retrieves real WAVs rather than only pointers.
