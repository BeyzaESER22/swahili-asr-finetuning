# Project 3 model and baseline registry — agreed scope (2026-11-17)

**User's clarification:** the instructor requests **one required baseline**. The team chose Whisper-Tiny as that main baseline. The project retains the earlier broader comparison: a pretrained Wav2Vec2-CTC ASR model will be an **additional pretrained comparator**, and a fine-tuned Whisper-Tiny will be the **trained contribution**.

| Experiment | Role | Model | Group training | State |
| --- | --- | --- | --- | --- |
| E1 | **Main mandatory baseline** | `openai/whisper-tiny` (pretrained, vanilla); pin model revision | No, inference only | NOT RUN |
| E2 | Additional pretrained comparator (**technically also a baseline**, never call this new model contribution) | Swahili-capable Wav2Vec2-CTC ASR checkpoint *with ASR head*; exact model ID/revision TBD | No, inference only | CHECKPOINT NOT SELECTED |
| E3 | **Group's model contribution** | Fine-tuned copy of E1 Whisper-Tiny checkpoint | Yes, supervised fine-tuning on selected training data | NOT RUN |

**Two model families, three runs, one fine-tuning training job.** The E2 checkpoint is not approved until the team has checked its Swahili support, tokenizer/CTC vocabulary and actual transcription head. An encoder-only wav2vec2 model cannot transcribe.

## Comparison protocol
- Dataset: at most 10 hours total validated audio, with *target* 8h train / 1h val / 1h test (actual grouped split can deviate).
- Evaluate E1, E2, E3 on identical frozen test utterance IDs and gold transcripts; record WER and CER from one normalization/scoring implementation.
- Save decoded predictions, exact checkpoint IDs + revisions, model-specific preprocessing, decoding hyperparameters, seeds and machine-generated evaluation outputs.
- E1 vs E3 is the principal contribution analysis; E2 provides cross-architecture context.
- Data budget does not guarantee a maximum training wall-clock time. Perform a throughput pilot before full E3 training.

## Instructor context
The instructor's earlier Week 9 note lists families such as GMM-HMM, CTC, encoder-decoder and transducer, and checkpoints such as Whisper, wav2vec2, Canary, Parakeet and Qwen3-ASR. These were described as examples of *baseline methods*, not a requirement in the material provided to train all of them. If a subsequent explicit project-specific baseline list conflicts with the user's confirmation of one required baseline, update the scope before completing work.

## Pre-submission checklist
- [ ] Exact E1 Whisper-Tiny revision is pinned.
- [ ] E2 model ID and *Swahili-compatible* ASR head is verified.
- [ ] Raw/cleaned audio (up to 10h final selected corpus) and transcripts included in repository according to lecturer rules.
- [ ] E1 and E2 inference reproduce predictions using committed scripts.
- [ ] E3 fine-tuning script, settings, logs and reproducible checkpoint path are committed.
- [ ] All three run on exact same test IDs, normalized consistently.
- [ ] Final results come from evaluation code, not manual copy/paste.
