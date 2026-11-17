# FLEURS data license and attribution

The audio and transcripts under `data/raw/`, `data/cleaned/`, and `data/splits/`
are a selected subset of **FLEURS (Few-shot Learning Evaluation of Universal
Representations of Speech), `sw_ke` configuration**.

- Source: <https://huggingface.co/datasets/google/fleurs>
- Paper: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod,
  Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna, “FLEURS:
  Few-shot Learning Evaluation of Universal Representations of Speech,” 2022.
- License: [Creative Commons Attribution 4.0 International
  (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Changes made for this project: the official `train`, `dev`, and `test`
boundaries were preserved; clips were structurally audited; recordings longer
than 30 seconds were excluded from the eligible pool; and a deterministic
maximum-10-hour subset was selected. The selected audio bytes and reference
transcripts were not denoised, edited, or corrected. Derived manifests add
duration, integrity, waveform-review, and SHA-256 fields. See
`data/audit/summary.json` and `data/cleaned/cleaning_log.tsv` for the exact
decisions and limitations.

The repository's root MIT license covers project-authored source code only. It
does not replace or relicense FLEURS data.
