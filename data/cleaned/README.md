# Cleaned selected dataset

`manifest.tsv` is the audited, frozen 9.786916667-hour selection and
`cleaning_log.tsv` records the 73 duration exclusions. Audio under `audio/` is
byte-identical to raw audio: this pipeline performs validation and selection,
not denoising or transcript rewriting. The absence of listening-based review is
an explicit remaining limitation, not a completed check.
