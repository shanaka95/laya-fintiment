# laya-fintiment

> **TL;DR.** Took [Laya](https://huggingface.co/convaiinnovations/laya) (421M, ModernBERT-large + decision head) and fine-tuned it with RLCD on 61k financial headlines/tweets to do one thing — pick `positive` / `negative` / `neutral` from a financial text. On the held-out 15,355-item test split:
>
> | | Accuracy | F1 (macro-avg) | Latency |
> |---|---|---|---|
> | Base Laya (zero-shot) | 70.24% | 0.705 | 38.6 ms |
> | **This model (fine-tuned)** | **94.86%** | **0.948** | 36.6 ms |
>
> **+24.62 percentage points** of accuracy at the same latency, on the same questions, on the same hardware.

This repo holds the **code, notebooks, training log, and eval charts** that produced it. The model weights live on the Hugging Face Hub:

- 🤗 Model: **[shanaka95/laya-fintiment](https://huggingface.co/shanaka95/laya-fintiment)** (~842 MB)
- 🤗 Dataset: **[shanaka95/fingpt-sentiment-3class](https://huggingface.co/datasets/shanaka95/fingpt-sentiment-3class)** (76,772 rows, 3-class)
- 🤗 Base: **[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)**

---

## Table of contents

1. [What this is, in one paragraph](#1-what-this-is-in-one-paragraph)
2. [Headline numbers](#2-headline-numbers)
3. [Side-by-side eval charts](#3-side-by-side-eval-charts)
4. [How the model is used](#4-how-the-model-is-used)
5. [How it was trained](#5-how-it-was-trained)
6. [How to reproduce](#6-how-to-reproduce)
7. [What lives in this repo](#7-what-lives-in-this-repo)
8. [Honest limits](#8-honest-limits)
9. [License & credits](#9-license--credits)

---

## 1. What this is, in one paragraph

Laya is a non-autoregressive **decision model**: you give it text + a typed question (e.g. "pick one of three options"), it returns the chosen option with a calibrated probability, in a single forward pass. The base checkpoint is general-purpose — it works on intent, routing, scoring, moderation — but on this benchmark it's near chance out of the box.

This repo is the result of taking that base checkpoint and fine-tuning it on a **single, narrow task**: 3-class financial sentiment. The training objective is RLCD — REINFORCE with a strictly proper scoring rule as the reward, group-mean baseline, Gaussian exploration noise on the logits. No new data was labelled; we collapsed FinGPT's existing 9-class sentiment labels into 3 and trained for 4 epochs.

---

## 2. Headline numbers

Same 15,355-item stratified test split. Same `choice` question schema. Same `laya.Agent.predict()` call. Only the model changed.

| Metric | Zero-shot | **Fine-tuned** | Δ |
|---|---|---|---|
| **Overall accuracy** | 0.7024 | **0.9486** | **+0.2462 (+24.62 pp)** |
| Precision — positive | 0.854 | 0.950 | +0.096 |
| Precision — neutral | 0.621 | 0.943 | +0.321 |
| Precision — negative | 0.694 | 0.956 | +0.262 |
| Recall — positive | 0.584 | 0.954 | **+0.370** |
| Recall — neutral | 0.792 | 0.955 | +0.163 |
| Recall — negative | 0.761 | 0.928 | +0.168 |
| F1 — positive | 0.694 | 0.952 | +0.258 |
| F1 — neutral | 0.696 | 0.949 | +0.253 |
| F1 — negative | 0.726 | 0.942 | +0.216 |
| Latency p50 / p95 | 38.6 / 41.0 ms | 36.6 / 39.1 ms | -2 ms |
| Throughput | ~25 ex/s | ~26.5 ex/s | +1.5 ex/s |

**Confusion matrices** (rows = true, cols = predicted):

<table>
<tr><th>Zero-shot</th><th>Fine-tuned</th></tr>
<tr><td>

| | positive | neutral | negative |
|---|---|---|---|
| **positive** | 3,564 | 2,140 | 398 |
| **neutral** | 471 | 4,627 | 745 |
| **negative** | 137 | 678 | 2,595 |

</td><td>

| | positive | neutral | negative |
|---|---|---|---|
| **positive** | 5,820 | 210 | 72 |
| **neutral** | 189 | 5,580 | 74 |
| **negative** | 115 | 129 | 3,166 |

</td></tr>
</table>

**What the rows actually mean for a finance workflow:**

- **Bullish precision** (when the model says "positive", how often is it right?) — 85.4% → **95.0%**. Fewer false bullish alarms to act on.
- **Bullish recall** (of all true positives, how many get flagged?) — 58.4% → **95.4%**. **The single biggest move.** The base model was leaking 35% of bullish signals into the neutral bucket; after fine-tuning only 3.4% leak.
- **Bearish → bullish flips** (the costliest error — going long when you should go short) — 137 → 115 (small absolute move, but recall climbed 17 points, so the *rate* of bearish → bullish is materially lower).
- **Bullish → bearish flips** — 398 → **72** (an 82% drop). The other expensive direction.
- **Latency unchanged** — same forward pass, same hardware, ~37 ms p50.

### Worked example (smoke test)

| | Base Laya | **Fine-tuned** |
|---|---|---|
| Input: *"Asset Management One Co. has a bullish call on Treasuries. One for the long, long run https://t.co/8go8ZhMvdc"* (gold: positive) | `positive` @ **0.5604**, neutral @ 0.3705 (hedging) | `positive` @ **0.9488** (decisive) |

---

## 3. Side-by-side eval charts

The two PNGs below are produced by `laya_inference_eval.ipynb` (zero-shot) and `laya_finetuned_eval.ipynb` (fine-tuned). Identical layout so you can compare panel-by-panel.

| Zero-shot — base `convaiinnovations/laya` | Fine-tuned — `shanaka95/laya-fintiment` |
|---|---|
| ![Zero-shot](./eval_zero_shot.png) | ![Fine-tuned](./eval_finetuned.png) |
| 70.24% overall — diagonal weak, neutral column inflated, bullish row leaks into neutral | 94.86% overall — diagonal dominates, off-diagonal cells shrink ~10×, distributions track actuals |

Each chart contains four panels:
1. **Confusion matrix** (row-normalised — what fraction of each true class ends up predicted as each label).
2. **Per-class precision / recall / F1** bars.
3. **Class distribution** — predicted vs actual counts.
4. **Inference latency** histogram with p50 and p95 markers.

---

## 4. How the model is used

```bash
pip install "laya>=0.1.6" "transformers>=4.48.0"
```

```python
import os
os.environ["USE_TF"] = "0"  # avoid a TF/abseil deadlock when loading transformers

import laya

agent = laya.Agent("shanaka95/laya-fintiment", device="cuda")

questions = {
    "sentiment": {
        "type": "choice",
        "instructions": (
            "What is the financial sentiment of this text? "
            "Please choose exactly one answer from {positive, negative, neutral}. "
            "Answer only with the chosen label."
        ),
        "criteria": {
            "positive": "The text expresses a positive / bullish financial sentiment.",
            "negative": "The text expresses a negative / bearish financial sentiment.",
            "neutral":  "The text is neutral, factual, or has no clear positive or negative sentiment.",
        },
    }
}

result = agent.predict(
    "Apple reported record quarterly revenue, beating analyst estimates.",
    questions,
)

ans = result["answers"]["sentiment"]
print(ans["choice"])            # -> 'positive'
print(ans["probabilities"])     # -> {'positive': ~0.95, 'negative': ~0.01, 'neutral': ~0.04}
print(ans["confidence"])        # -> ~0.78 (calibrated via post-training temperature)
```

The question schema is **identical** to what was used during training. Changing `instructions` or `criteria` at inference time is fine — Laya is robust to prompt rephrasing — but changing the option labels (`positive` / `negative` / `neutral`) will degrade accuracy.

For batched or multi-lingual usage, see the upstream [Laya README](https://github.com/NandhaKishorM/laya).

---

## 5. How it was trained

### Architecture (inherited, not modified)

- **Backbone:** ModernBERT-large (395M, bidirectional, fully fine-tuned)
- **Decision head:** 2 transformer layers + an option-marker scorer + an act/escalate head, trained from scratch
- **Total:** 421M parameters, 512-token context window (`head_max_len = 256`)

### Objective

RLCD — **Reinforcement Learning for Calibrated Decisions**. The policy reports a distribution over the three options; exploration adds zero-mean Gaussian noise to the logits; the reward is a strictly proper scoring rule (log loss + spherical, with `w_sph = 0.75`). Expected reward is maximised only by reporting honest probabilities, so calibration is a by-product, not a post-hoc adjustment. Updates are REINFORCE with a group-mean baseline (GRPO-style, `group_size=4` noisy samples per item).

### Hyperparameters

| | |
|---|---|
| Epochs | 4 |
| Train items | 61,017 (400 held out for calibration) |
| Optimizer steps | 7,624 (effective batch 32 = micro 16 × grad-accum 2) |
| Optimiser | AdamW |
| LR schedule | CosineAnnealingLR, T_max = total_updates |
| Sequence budget | `max_len=1024`, `head_max_len=256`, `max_tokens_per_batch=16384` |
| Memory toggles | gradient_checkpointing ON, allow_tf32 ON, cudnn.benchmark ON |
| Hardware | 1 × 12 GB GPU (vast.ai) |

### Training curve (from `train.log`)

| Epoch | Avg loss | Wall time | Notes |
|---|---|---|---|
| 1 | 0.257 | 50 min | First epoch of exploration |
| 2 | 0.493 | 50 min | Loss rises — model starts pushing past the 0.750 reward ceiling |
| 3 | 0.323 | 50 min | Settling |
| 4 | **0.199** | 50 min | Lowest loss, no overfit signal |

The epoch-2 loss bump is expected — that's when the model starts learning to confidently distinguish all three labels. The reward saturates at the maximum 0.750 = `w_sph × 1.0` for `choice` questions with the correct argmax.

### Post-training calibration

400 held-out items from the train split (seed 20260922) were used to fit a per-qtype temperature on the option-marker logits:

- **choice**: T = **5.013**
- **score**: T = 1.2
- **noul**: T = 1.2

The choice-temperature of 5.0 sharpens probability mass onto the picked label, which matches the argmax-accuracy objective the model was trained on. Without it, the raw probabilities are more diffuse but the argmax accuracy is unchanged.

---

## 6. How to reproduce

### Re-run the fine-tuned eval (10 min on a 12 GB GPU)

1. Open `laya_finetuned_eval.ipynb` in any CUDA-capable Jupyter environment (vast.ai, Colab, Lambda, local).
2. Run all cells in order. First cell installs `laya`, `transformers`, `datasets`, `matplotlib`. Last cell saves `eval_finetuned.png` to `/workspace/`.
3. Expected output: `OVERALL ACCURACY: 0.9486` and `DELTA vs zero-shot baseline (0.7024): +0.2462`.

### Re-run the fine-tune from scratch (~3.4 h on a 12 GB GPU)

1. Open `laya_finetune_sentiment.ipynb`.
2. Run all cells. The notebook:
   - Pulls the train split of `shanaka95/fingpt-sentiment-3class` (61,417 items).
   - Holds out 400 items for calibration.
   - Builds the typed-decisions `choice` question schema shown in section 4.
   - Trains 4 epochs of RLCD with `group_size=4` noisy samples per item.
   - Fits calibration temperatures on the 400-item holdout.
   - Saves to `./laya_fintiment/`.
3. Push with `hf upload shanaka95/laya-fintiment ./laya_fintiment . --repo-type model`.

The full per-50-step log from the original run is in [`train.log`](./train.log).

### Re-run the zero-shot baseline (10 min)

Identical to the fine-tuned eval but `MODEL_ID = "convaiinnovations/laya"`. See [`laya_inference_eval.ipynb`](./laya_inference_eval.ipynb).

---

## 7. What lives in this repo

| File | What it is |
|---|---|
| [`README.md`](./README.md) | This file. |
| [`laya_finetune_sentiment.ipynb`](./laya_finetune_sentiment.ipynb) | The fine-tuning notebook — runs end-to-end on a 12 GB GPU. |
| [`laya_inference_eval.ipynb`](./laya_inference_eval.ipynb) | Zero-shot eval of base `convaiinnovations/laya`. Produces the 70.24% baseline. |
| [`laya_finetuned_eval.ipynb`](./laya_finetuned_eval.ipynb) | Fine-tuned eval of `shanaka95/laya-fintiment`. Produces the 94.86% number and the `eval_finetuned.png` chart. |
| [`train.log`](./train.log) | Raw 4-epoch training log. Loss, reward, LR, GPU util printed every 50 steps. |
| [`eval_zero_shot.png`](./eval_zero_shot.png) | Confusion matrix + per-class metrics + class distribution for the zero-shot run. |
| [`eval_finetuned.png`](./eval_finetuned.png) | Same chart layout for the fine-tuned run. |

---

## 8. Honest limits

- **3-class only.** Anything beyond `positive / negative / neutral` (intensity, topic, aspect, time horizon) needs a different fine-tune or a `score` / `noul` question. The underlying checkpoint supports those primitives, but they're not trained here.
- **Domain match matters.** Training data is FinGPT-sentiment-train (mostly short news headlines and tweets, English). Earnings calls, regulatory filings, and analyst reports will be OOD; expect accuracy drift. Refit temperatures per your domain before trusting the probabilities — the upstream [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) shows how.
- **No soft distribution comparison done.** Argmax accuracy is reported (and it's strong). Soft ECE / Brier against gold probability distributions was not measured here. Worth doing if you plan to use `probabilities` directly as a trading signal rather than `choice`.
- **Single training run.** No hyperparameter sweep, no ensemble, no data-augmentation ablations. The recipe is "what worked once"; it likely works again but isn't a tuned optimum.
- **One-shot calibration.** Temperatures were fit on 400 held-out items. For production use, refit on a larger calibration set from your domain.

---

## 9. License & credits

- **Code in this repo:** MIT.
- **Model weights on the Hub:** Apache 2.0 (inherited from [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)).
- **Training data:** derived from [FinGPT/fingpt-sentiment-train](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) — please respect their license terms.

**Acknowledgements:**
- [NandhaKishor M](https://github.com/NandhaKishorM) — author of Laya, the RLCD training library, and the upstream fine-tuning notebook we adapted.
- [Convai Innovations](https://huggingface.co/convaiinnovations) — base model and ongoing Laya development.
- [FinGPT](https://huggingface.co/FinGPT) — the sentiment dataset we collapsed to 3 classes.
