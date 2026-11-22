# Results

Every file here is machine-generated from committed code. Do not edit numbers by hand or copy them from Colab.

| Path | Produced by | Content |
| --- | --- | --- |
| `predictions/E1.tsv`, `E2.tsv`, `E3.tsv` | `src/run_baseline.py` (T3, T4) | Per-clip `id`, `reference`, `hypothesis` on the 252 frozen test clips |
| `runs/E1.json`, `E2.json`, `E3.json` | `src/run_baseline.py` | Model ID and commit, decoding, versions, device, runtime, test-manifest SHA-256 |
| `runs/E3_train.json` | `src/train.py` (T4) | E3 training configuration, manifest hashes, losses and validation WER/CER |
| `metrics.csv` | `src/score_predictions.py` (T5) | WER/CER of E1, E2 and E3 with the shared scorer |
| `error_analysis/` | `src/error_analysis.py` (T5) | Bootstrap CIs, per-clip errors, subgroups, OOV, word-boundary and substitution tables; see the README T5 section |

The fine-tuned E3 checkpoint (`checkpoints/E3/best`, ~151 MB) is not committed.
