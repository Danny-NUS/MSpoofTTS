# Developer README

## 1) Discriminator Training

Use these two scripts:

- `train_segment_discriminator.py`
  - Model: `SegmentTokenDiscriminator`
  - Default `SEGMENT_LEN = 10`
  - Trains a segment-level real/fake classifier.

- `train_strided_discriminator.py`
  - Model: `StridedSegmentTokenDiscriminator`
  - Default `SEGMENT_LEN = 50`, `SCALE = 10`
  - Trains a strided segment discriminator used by hierarchical/ranking inference.

Shared behavior in both scripts:

- Real label = `1`, synthetic label = `0`
- Data is loaded from Kaldi-style folders (`wav.scp`, `text`, `utt2codes`)
- Each sample randomly selects one token segment per call
- Training uses PyTorch Lightning (`max_epochs=5`, GPU, TensorBoard logger, checkpoints every epoch)

Quick run:

```bash
python train_segment_discriminator.py
python train_strided_discriminator.py
```

## 2) Inference and `neutts/neutts.py`

Main torch inference routing is in `_infer_torch(...)` and dispatches by `sampling_scheme`.

- `orig`
  - Original baseline generation path.
  - Uses `backbone.generate(...)` with standard sampling settings.

- `ras_k50_win25`
  - Repetition-Aware Sampling (RAS).
  - Uses nucleus-style sampling, and if repetition in the recent window is too high, it suppresses the repeated token and resamples.

- `eas`
  - Entropy-Aware Sampling (EAS).
  - Applies a decaying temporal penalty memory over recently favored tokens before sampling.

- `rank_ras_hier`
  - Hierarchical beam process with RAS token expansion.
  - Multi-stage filtering/ranking with discriminators at segment lengths 10 -> 25 -> 50, then final rank-based selection.

- `rank_eas_hier`
  - Hierarchical beam process with EAS token expansion.
  - Same staged discriminator selection, but token expansion uses EAS memory/penalty dynamics.

Example usage:

- See `test.ipynb` (there are cells for `orig`, `eas`, `ras_k50_win25`, `rank_ras_hier`, `rank_eas_hier`).
