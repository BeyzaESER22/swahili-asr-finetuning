# Dataset research and selection — Swahili ASR

Research notes prepared 2026-11-17 by Beyza Eser as the **dataset research/selection** deliverable. The downloading, cleaning, clip QA, splitting and upload work is explicitly delegated to the team's future data engineering owner (not Beyza). **Selection is provisional: no audio or transcript files have been downloaded to this repository.**

## Confirmed group selection and corpus budget

The student confirmed no other group chose the selected dataset. **Maximum selected usable audio: 10 hours total**, with ~8 hours for training, ~1 hour validation and ~1 hour test. Duration must be measured from real clips (not converted from number of rows). Speaker-disjoint selection may make split lengths approximate. Instructor minimum-hours/full-credit criteria remain unconfirmed. This dataset research is Beyza's responsibility; actual downloading and cleaning remain **assigned to other members**.

## Primary candidate: Mozilla Common Voice Scripted Speech v27.0 — Swahili
- Locale: `sw` (Kiswahili); task: read-speech ASR; formats include MP3 clips and tabular metadata.
- Official dataset catalog identifies **CC0-1.0** and **20.89 GB** for the full Swahili package. This is the *published archive size*, not the target training subset.
- Source: https://commonvoice.mozilla.org/en/datasets
- Release announcement (2026-09-18): https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490
- Dataset version and archive hash must be recorded after download. Use only validated reference transcripts and verify the data source's acceptable-use terms.
- Example Common Voice metadata may include `path`, `sentence` and `client_id`, plus accent/locale fields. Inspect the actual downloaded schema rather than relying on assumptions.
- Speaker-disjoint evaluation requires nonempty `client_id`; exclude or separately review missing IDs.

### Why this is the leading option
A non-English language, transcribed speech, a well-documented source, opportunity for meaningful audio/transcript integrity checks, and sufficient *potential* material for a nontrivial cleaned subset. **No low-resource or novelty bonus is guaranteed.** Swahili may be regarded as better-resourced than some alternatives.

### Practical blocker
The full package is not appropriate for ordinary Git objects. The lecturer explicitly requires raw **and** cleaned data to be present in the submitted repository. Verify whether a **meaningful, licensed small subset with actual audio** can be placed under `data/raw/` and `data/cleaned/` and whether Git LFS storage/bandwidth is accepted; neither a Google Drive link nor a download-only script suffices. Do not upload until terms, disk quota and instructor acceptance are confirmed. Use a representative, speaker-disjoint subset; avoid cherry-picking easy clips.

## Secondary candidate: Google FLEURS (Swahili Kenya)
- Source: https://huggingface.co/datasets/google/fleurs
- Config typically `sw_ke`; verify exact revision, configuration, split counts and licensing **from the authoritative card before actual usage**.
- Suitable potential external-domain evaluation set; do not mix with Common Voice training without a documented reason, license confirmation and group dataset-novelty check.
- FLEURS results must be labeled **out-of-domain**, not quietly merged into the same test benchmark.

## Proposed cleaning audit
1. Verify archive version, original TSV columns, source split and SHA256.
2. Reject empty reference, missing clip, missing speaker identifier, duplicate clip path and duplicate exact path+reference.
3. Check duration, sample rate, clipping/silence, inaudible/corrupt clips, wrong-language and transcript/audio alignment **with actual implemented validation**.
4. Log for each exclusion: original identifier, specific reason, and source. Do not invent corrections or reference words.
5. Save original and cleaned **audio plus metadata**, not only manifest. Record accepted/rejected duration and speaker statistics.
6. Freeze speaker-disjoint train/val/test manifests with checks for duplicate audio or repeated content across splits.
7. Preserve immutable original references; define text normalization centrally for both hypotheses and references.

The current `src/audit_manifest.py` is a **limited metadata/path audit only**; it does **not** implement waveform/audio quality checks or create cleaned audio files. The additional stages above are requirements, not completed work.

## Dataset decision record
| Decision | Value |
| --- | --- |
| Primary dataset | Common Voice Scripted Speech v27.0 Swahili (candidate) |
| Language availability confirmed with classmates | YES — user confirmed no overlap |
| Training dataset uniqueness confirmed | YES — user confirmed no overlap |
| Source archive downloaded and hashed | NO |
| Dataset terms individually reviewed before sharing | PENDING |
| Selected dataset target / Git LFS | <=10 total hours targeted; Git LFS compliance PENDING |
| Audio quality audit and actual cleaned files | NOT STARTED |
| Speaker-disjoint splits validated on real data | NOT STARTED |
| Instructor baseline and dataset-size rules | PENDING |

## Source ethics
Avoid speaker re-identification. Do not use model predictions as gold transcripts. Correct suspect transcripts only when supported by a trustworthy original/source or documented human verification protocol.
