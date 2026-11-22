# Individual contribution — Nilasu Yıldız

- Student name: Nilasu Yıldız
- Student ID: 2309011057
- GitHub account: `xyzta`
- Project role: **Unified Benchmark, Statistical Testing and Error Analysis (T5)**

## Scope of my work (2026-11-22)
- Took over the T5 handoff after the frozen FLEURS `sw_ke` split (T1/T2), the E1/E2 baselines (T3) and the E3 fine-tuned Whisper-Tiny predictions (T4) were committed.
- Owned the final comparison of all three runs: one shared scorer, one normalisation policy, the same 252 test clips, and every reported number regenerated from committed predictions by code.
- Did **not** run or train any model. All analysis uses the committed prediction files `results/predictions/E1.tsv`, `E2.tsv` and `E3.tsv`.

## Completed benchmark and error analysis (2026-11-22)

**Audit of the T4 hand-off.** Before scoring E3, I checked the committed T4 artefacts and turned the checks into `tests/test_benchmark_artifacts.py` (11 tests). E3 was trained on the frozen train/validation manifests (SHA-256 match), was a full run rather than the smoke test (1,902 + 210 clips), started from the same `openai/whisper-tiny` commit as E1 (`169d4a4341`), logged the hyperparameters recorded in `configs/experiment.json`, and no test clip appears in train or validation. All three prediction files cover exactly the 252 frozen test IDs with unchanged references. The audit found that `results/metrics.csv` had not been regenerated after T4 and still lacked E3; I regenerated it with `src/score_predictions.py` instead of editing it, and the test now checks that the committed table matches a fresh rescoring. I also set `lineterminator="\n"` in `src/score_predictions.py` so the table no longer changes line endings between Windows and macOS/Linux. `src/run_baseline.py` wrote `"group_training": false` for every run, so the correct value for E3 had been entered in `results/runs/E3.json` by hand; I made the flag follow the run's role (`group_training()`), with a test, so a rerun of E3 inference writes the correct value.

**Scorer tests.** `tests/test_metrics.py` (14 tests) hand-checks the shared scorer in `src/metrics.py`: substitution, pure deletion, pure insertion with WER above 1, blank and punctuation-only hypotheses, pooled corpus WER versus per-clip average, CER, rejection of blank references and count mismatches, NFC, and the documented digit policy. The existing normalisation was kept as written in T3, before E3 was scored; I did not change it after seeing test scores.

**Error analysis.** I wrote `src/error_analysis.py`. It re-uses the same ID/reference check as `score_predictions.py` and writes ten tables to `results/error_analysis/`: per-run summary (S/D/I, insertion share, median clip WER, clips with WER > 1, repetition loops, empty and perfect outputs), per-clip errors with test-set metadata, subgroups (gender, clip duration, digits, loudness tercile), pairwise comparison against E1, a sentence-clustered bootstrap, train-vocabulary (OOV) error rates, word-boundary (space) errors, substitution categories, the most frequent substitutions and best/median/worst examples. The bootstrap resamples whole `sentence_id` groups (252 clips are 162 sentences read by different speakers) with a fixed seed, so the committed tables are reproduced exactly. `tests/test_error_analysis.py`, `tests/test_error_analysis_extended.py` and `tests/test_error_analysis_t5_extra.py` (41 tests) cover per-clip counts, loop detection (including Swahili reduplication such as `pole pole`, which must not count as a loop), bootstrap behaviour, OOV splitting, space edits, substitution categories and the output files on the real predictions. One test caught an error in my first substitution heuristic, which labelled unrelated words as prefix errors; I fixed the rule after the failing test.

| Run | WER | CER | WER 95% CI |
| --- | ---: | ---: | --- |
| E1 Whisper-Tiny, vanilla (main baseline) | 1.1176 | 0.4891 | 1.051 – 1.197 |
| E2 MMS-1B-all + `swh` (comparator) | 0.1483 | 0.0367 | 0.132 – 0.165 |
| E3 Whisper-Tiny, fine-tuned (contribution) | 0.4804 | 0.1627 | 0.416 – 0.562 |

Main findings: E3 − E1 is −0.637 WER (95% CI −0.732 to −0.553, paired bootstrap p ≤ 0.001) and E3 is better on 246 of 252 clips; repetition loops fall from 24 clips to 5. Fine-tuning mainly helps on words seen in training: E3's error is 27% on seen words but 70% on the 21% of test words never seen in training (E1: 90% / 92%). E3's substitutions become near misses (unrelated-word substitutions fall from 69% to 31%), and quiet recordings are the weakest subgroup for every model (E3: 0.62 vs 0.37 WER). I wrote the T5 section of the README and updated the T5 row in `docs/TEAM_TASKS.md`. The full suite (100 tests) passes.

## Attribution limitations
The analysis script and tests were prepared with AI assistance, under my direction and review; I ran them on the committed predictions myself and committed the work from my own GitHub account. The dataset (T1, Beyza Eser), split QA (T2, Meltem Çelik), baselines (T3, Berk Ötün) and fine-tuning (T4, Asya Oral) are **not** claimed here. Limitations of the analysis: FLEURS has no speaker IDs, so no per-speaker analysis is possible; some subgroups have only 30–39 clips; the substitution categories are a character-overlap heuristic, not a morphological parser; the bootstrap covers test-set sampling, not training randomness, because E3 was trained once with one seed; no listening-based review of clips was performed.

## Evidence
- Branch `t5-evaluation`, merged into `main` through a pull request.
- `src/error_analysis.py`, the `lineterminator` change in `src/score_predictions.py`, `group_training()` in `src/run_baseline.py`, `tests/test_metrics.py`, `tests/test_error_analysis.py`, `tests/test_error_analysis_extended.py`, `tests/test_error_analysis_t5_extra.py`, `tests/test_benchmark_artifacts.py`, `results/metrics.csv`, `results/error_analysis/`, the README T5 section and this file's Git history.
- Reproduce: `python src/score_predictions.py --predictions results/predictions/E1.tsv results/predictions/E2.tsv results/predictions/E3.tsv`, then `python src/error_analysis.py --predictions results/predictions/E1.tsv results/predictions/E2.tsv results/predictions/E3.tsv`, then `python -m unittest discover -s tests -v`.
