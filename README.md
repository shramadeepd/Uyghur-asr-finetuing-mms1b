<div align="center">

# Uyghur ASR — 0.0517 CER with 46K Trainable Parameters

**Fine-tuning a 1B-parameter speech model on a low-resource language by retraining 0.005% of it.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.56.2-FFD21E)](https://huggingface.co/docs/transformers)
[![CER](https://img.shields.io/badge/CER-0.0517-2ea44f)](#results)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Notebook](notebooks/uyghur-asr-mms1b-annotated.ipynb) ·
[Model](https://huggingface.co/Shramadeepd/wav2vec-ug-finetuned-1b) ·
[Approach](#approach) ·
[Results](#results) ·
[Quick start](#quick-start)

</div>

---

## Overview

An end-to-end Uyghur automatic speech recognition system built for the **NPPE-2 ASR Challenge**,
achieving **0.0517 Character Error Rate** — roughly one incorrect character in every nineteen.

The notable part is not the score but the cost: the model has **964,694,692 parameters**, of
which **46,116 were trained** — 0.005% — in a **single epoch on 2 × Tesla T4 GPUs in under an
hour**.

```
audio (16 kHz waveform)
  └─► Wav2Vec2FeatureExtractor      per-utterance normalisation
      └─► CNN feature encoder       ❄️  FROZEN   waveform → 20 ms frame vectors
          └─► 48-layer Transformer  ❄️  FROZEN   contextual acoustic representations
              └─► lm_head           🔥  TRAINED  Linear(1280 → 36) character logits
                  └─► CTC greedy decode ──► transcription
```

---

## Approach

Uyghur is a low-resource language: ~23 hours of labelled audio is nowhere near enough to train
an acoustic model from scratch. The entire problem is therefore transfer learning — and the
question is *how much* of the network actually needs to change.

The base checkpoint, [`ixxan/wav2vec2-large-mms-1b-uyghur-latin`](https://huggingface.co/ixxan/wav2vec2-large-mms-1b-uyghur-latin),
had **already been fine-tuned on Uyghur Latin speech**. Its encoder already understood Uyghur
phonetics. The only component that did not transfer was the **output alphabet** — the checkpoint
exposes 34 character logits, while this corpus requires 36.

So rather than fine-tune a billion parameters, the solution:

1. Replaces `Linear(1280 → 34)` with a freshly initialised `Linear(1280 → 36)`
2. Freezes **everything else** in the network
3. Trains the head alone at an aggressive `lr = 1e-3` for one epoch

The head only needs to learn the projection from an already-correct acoustic representation onto
a new symbol set — close to learning a permutation.

### Why it works

Training loss collapsed from **12.51 → below 1.0 within 50 optimizer steps**:

| Step | 1 | 30 | 50 | 100 | 426 |
|---|---|---|---|---|---|
| Loss | 12.51 | 2.62 | 0.64 | 0.49 | ~0.35 |

That trajectory is the evidence: the frozen encoder's frame representations were already close
to linearly separable into the correct characters. The head only had to find the projection.

### What freezing buys

| Benefit | Detail |
|---|---|
| **~7 GB VRAM saved** | Adam keeps two moment buffers per trainable parameter; 964M of them do not exist here. This is the difference between fitting on a 14 GB T4 and not. |
| **No backward pass through 48 layers** | Gradients terminate at the head. |
| **No catastrophic forgetting** | A 1e-3 learning rate with a randomly initialised head producing loss-12.5 gradients would damage an unfrozen backbone within a few dozen steps. Here it is mathematically impossible. |
| **Gradient checkpointing unnecessary** | Recomputing frozen activations is pure waste, so it stays disabled. |

---

## Results

| | |
|---|---|
| **Character Error Rate** | **0.0517** |
| Trainable parameters | 46,116 / 964,694,692 (0.005%) |
| Training time | 59 min 43 s |
| Hardware | 2 × Tesla T4 (14 GB each), 16 GB system RAM |
| Epochs / steps | 1 / 426 |

---

## Training specification

<table>
<tr><th colspan="2" align="left">Hardware</th></tr>
<tr><td>GPU</td><td>2 × NVIDIA Tesla T4, 14 GB usable VRAM each</td></tr>
<tr><td>System RAM</td><td>16 GB</td></tr>
<tr><td>Precision</td><td>fp16 mixed precision</td></tr>
<tr><th colspan="2" align="left">Hyperparameters</th></tr>
<tr><td>Epochs</td><td>1 (426 optimizer steps)</td></tr>
<tr><td>Per-device batch size</td><td>8</td></tr>
<tr><td>Learning rate</td><td>1e-3, linear decay, 5% warmup</td></tr>
<tr><td>Optimizer</td><td><code>adamw_torch_fused</code></td></tr>
<tr><td>CTC loss reduction</td><td><code>mean</code></td></tr>
<tr><td>Length bucketing</td><td><code>group_by_length=True</code></td></tr>
<tr><td>Seed</td><td>42</td></tr>
<tr><th colspan="2" align="left">Data</th></tr>
<tr><td>Total audio</td><td>~23 hours</td></tr>
<tr><td>Train / validation</td><td>6,816 / 758 clips (90/10 split of 7,574)</td></tr>
<tr><td>Test</td><td>1,894 clips</td></tr>
<tr><td>Audio</td><td>16 kHz mono WAV</td></tr>
<tr><td>Vocabulary</td><td>36 tokens — 33 letters, <code>|</code>, <code>[UNK]</code>, <code>[PAD]</code> (CTC blank)</td></tr>
</table>

---

## Repository structure

```
uyghur-asr-mms1b/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── uyghur-asr-mms1b-annotated.ipynb    # full annotated pipeline
└── submissions/
    └── submission.csv                       # final predictions (0.0517 CER)
```

---

## Quick start

### Installation

```bash
git clone https://github.com/<your-username>/uyghur-asr-mms1b.git
cd uyghur-asr-mms1b
pip install -r requirements.txt
```

### Inference with the trained model

```python
import torch, librosa
from transformers import AutoModelForCTC, Wav2Vec2Processor

MODEL_ID = "Shramadeepd/wav2vec-ug-finetuned-1b"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda").eval()

speech, _ = librosa.load("sample.wav", sr=16000, mono=True)   # 16 kHz is mandatory
inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
inputs = {k: (v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda"))
          for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs).logits

print(processor.batch_decode(torch.argmax(logits, dim=-1))[0])
```

### Reproducing training

Open `notebooks/uyghur-asr-mms1b-annotated.ipynb` on Kaggle or Colab with a T4 runtime, point
`DATA_DIR` at the dataset, and run top to bottom. Every cell is documented with the reasoning
behind it.

---

## Implementation notes

Details that matter more than they look:

- **16 kHz is non-negotiable.** The CNN feature encoder has fixed strides `(5,2,2,2,2,2,2)` and
  consumes exactly 320 samples per output frame — one frame per 20 ms at 16 kHz. Any other
  sample rate stretches every learned temporal pattern and produces silently wrong output.
- **Never lowercase the transcripts.** `A G H J N O U` are distinct phonemes in this
  transliteration scheme, not capitalisation.
- **`[PAD]` doubles as the CTC blank (ε).** The blank is what allows a short transcript to align
  to a long frame sequence, and what lets genuine double letters (`ll`) survive the
  repeat-collapse step.
- **Label padding is masked to `-100`.** Otherwise the CTC loss treats padding as real characters
  the model must predict, teaching it to append garbage to every transcription.
- **`group_by_length=True` is the largest throughput win.** Without it, one 10-second clip in a
  batch of 2-second clips forces everything to be padded to 10 s, and most GPU work goes into
  processing zeros.
- **`remove_unused_columns=False` is mandatory.** `Trainer` otherwise strips `input_length` and
  silently disables length bucketing.

---

## Roadmap

Ordered by expected CER reduction per unit of effort:

- [ ] **KenLM beam-search decoding** via `pyctcdecode` — greedy decoding discards all linguistic
      context; typically a 10–20% relative CER gain
- [ ] **Unfreeze the top 8 encoder layers** at `lr ≈ 1e-5` after the head converges
- [ ] **Proper validation** — per-epoch CER with `load_best_model_at_end` and early stopping
- [ ] **SpecAugment + speed perturbation** (only valuable once the encoder is unfrozen)
- [ ] **Ensemble** MMS-1B with MMS-300M by averaging per-frame log-probabilities

---

## Acknowledgements

- [`ixxan/wav2vec2-large-mms-1b-uyghur-latin`](https://huggingface.co/ixxan/wav2vec2-large-mms-1b-uyghur-latin) — base checkpoint
- [Massively Multilingual Speech (MMS)](https://arxiv.org/abs/2305.13516), Meta AI
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477), Baevski et al.
- NPPE-2 Uyghur ASR Challenge organisers

## License

MIT — see [LICENSE](LICENSE). Model weights and dataset are subject to their own respective
licenses.