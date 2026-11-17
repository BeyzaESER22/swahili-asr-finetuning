# Project 3 baseline register — awaiting exact Week 9 requirement

The instructor says **all mandatory Project 3 baselines must be included in vanilla form, with training code when applicable**. The supplied announcement names **families/examples**, not necessarily the final exhaustive project-specific set:

- MFCC + GMM-HMM
- CTC
- Attention-based encoder-decoder
- Transducer
- HuBERT / wav2vec 2.0 based ASR
- Whisper
- Canary
- Parakeet
- Qwen3-ASR

**Do not silently assume that every named family is compulsory**, or conversely assume only Whisper is sufficient. Obtain the instructor's exact Week 9 list and check off every applicable item. An advanced pretrained model used unchanged is a baseline, not a new contribution.

| Item | Exact checkpoint/implementation | Vanilla result | Training script if needed | Decision |
| --- | --- | --- | --- | --- |
| Instructor-required baselines | PENDING Week 9 list | NOT RUN | NOT PREPARED | BLOCKED |
| Proposed Whisper baseline | PENDING ID / revision | NOT RUN | N/A for inference-only | CANDIDATE |
| Proposed fine-tuned Whisper | Same source checkpoint | NOT RUN | NOT PREPARED | CONTRIBUTION CANDIDATE |
| Proposed wav2vec2-CTC ASR comparator | PENDING Swahili-compatible ASR-head checkpoint | NOT RUN | Only if training | OPTIONAL |

For all rows, preserve common test IDs and scoring policies; report WER/CER and run provenance. Never call a bare wav2vec2 encoder a ready-to-transcribe ASR model.
