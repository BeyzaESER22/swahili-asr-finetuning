# Individual contribution — Berk Ötün
 
- Student name: Berk Ötün
- Student ID: 220901605
- GitHub account: `berkotun`
- Project role: **Baseline Inference, Pretrained Comparator Evaluation and Shared Scoring (T3)**
 
## Scope of my work (2026-11-19)
- Took over the T3 handoff after the FLEURS `sw_ke` splits were frozen in T1 and independently verified in T2.
- Ran the agreed vanilla baselines on the frozen 252-clip test split only: **E1** `openai/whisper-tiny` as the single required course baseline, and **E2** `facebook/mms-1b-all` with its Swahili adapter `swh` as an additional pretrained wav2vec2-CTC comparator. Neither model was trained or fine-tuned by the group.
- Confirmed that E2 is a full CTC ASR checkpoint with a transcription head and a Swahili language adapter, not an encoder-only wav2vec2 model.
 
## Completed baseline inference and scoring (2026-11-19)
 
I added `src/run_baseline.py`. It reads `data/splits/test.tsv`, uses `raw_transcription` as the human reference, loads each WAV as 16 kHz mono float32 and stops with a clear message if a file is still a Git LFS pointer. E1 runs Whisper encoder-decoder generation with `language=sw`, `task=transcribe` and greedy decoding; E2 runs greedy CTC decoding with the MMS `swh` adapter. Each run resolves the requested model revision to an immutable commit hash and writes per-utterance predictions to `results/predictions/<RUN>.tsv` plus provenance to `results/runs/<RUN>.json` (model commit, package versions, device, runtime, real-time factor and the SHA-256 of the test manifest).
 
I added `src/score_predictions.py`, which refuses to score unless a prediction file covers exactly the 252 frozen test IDs with unchanged reference text, and then computes WER/CER through the shared `src/metrics.py`. I updated the shared normalisation in `src/metrics.py` so that references and hypotheses are treated identically: Unicode NFC, lower case, punctuation removed except the apostrophe used in Swahili spelling, whitespace collapsed; digits are left unchanged. The final normalisation policy remains subject to T5 review.
 
I ran E1 on my own computer (CPU) and E2 on Google Colab (T4 GPU), copied the E2 outputs into the repository, and generated `results/metrics.csv` with the scoring script:
 
| Run | Role | Model (resolved commit) | Test clips | WER | CER |
| --- | --- | --- | ---: | ---: | ---: |
| E1 | Main baseline | `openai/whisper-tiny` (`169d4a4341`) | 252 | 1.1176 | 0.4891 |
| E2 | Additional pretrained comparator | `facebook/mms-1b-all` + `swh` (`3d33597edb`) | 252 | 0.1483 | 0.0367 |
 
E1's WER is above 1.0 because Whisper-Tiny frequently hallucinates repeated tokens on Swahili audio (for example long loops of a single word), so insertions exceed the reference word count; its CER shows that roughly half of the characters are still recovered. E1 was deliberately run without repetition penalties or other decoding changes so that it remains a vanilla baseline for the E3 fine-tuning comparison.
 
E2's much lower error must be interpreted with care: MMS-1b-all was trained on FLEURS training data among other corpora, so it is in-domain for this dataset even though our test split was held out. E2 is a comparator, not a group contribution.
 
I added `tests/test_baseline.py` (10 tests) covering the run registry, manifest reading and duplicate IDs, the real 252-clip test split, the LFS-pointer guard, prediction round-trips, refusal to score mismatched or edited test sets, and hand-checked normalisation and WER cases. The full suite (29 tests) passes. I also added `requirements-models.txt`, recorded the E2 checkpoint in `configs/experiment.json`, documented the commands in the README T3 section and updated the T3 row in `docs/TEAM_TASKS.md`.
 
Fine-tuning (T4), the final benchmark and error analysis (T5) are assigned to other team members. Final README preparation and the technical PPTX presentation remain shared deliverables.
 
## Attribution limitations
The inference and scoring scripts and their tests were prepared with AI assistance, under my direction and review; I executed both runs myself, checked the outputs and committed the work from my own GitHub account. The dataset (T1, Beyza Eser) and split verification (T2, Meltem Çelik) are **not** claimed here. Inference used package versions recorded in `results/runs/*.json`; E1 ran on CPU and E2 on a Colab GPU, so the recorded runtimes are not directly comparable.
 
## Evidence
- Pull request [#2](https://github.com/BeyzaESER22/swahili-asr-finetuning/pull/2) (E1 baseline and scoring) from branch `baseline-implementations`, merged into `main`; a follow-up pull request from the same branch adds the E2 results.
- `src/run_baseline.py`, `src/score_predictions.py`, `tests/test_baseline.py`, `results/predictions/E1.tsv`, `results/predictions/E2.tsv`, `results/runs/E1.json`, `results/runs/E2.json`, `results/metrics.csv` and this file's Git history.
- Reproduce: `python src/run_baseline.py --run E1`, `python src/run_baseline.py --run E2` (GPU recommended), then `python src/score_predictions.py --predictions results/predictions/E1.tsv results/predictions/E2.tsv`.
