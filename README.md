# Rumor Stance Detection with Twitter-RoBERTa

Improving rumor stance detection on the **SemEval-2017 Task 8 (RumourEval)** dataset by replacing the Branch-LSTM state-of-the-art with a domain-specific transformer, contextual branch encoding, and class-balanced training.

## Overview

Stance detection is a NLP technique that identifies a person’s attitude/position regarding a specific topic. The SemEval-2017 Task 8 dataset contains a series of twitter threads regarding many different news events, which provides a benchmark for this task. Because of the way this dataset is constructed, a model needs to evaluate the stance of replies that can be nested in several tweet depths from the original source tweet. The [SOTA](https://aclanthology.org/S17-2006.pdf) for this [dataset](https://figshare.com/articles/dataset/PHEME_rumour_scheme_dataset_journalism_use_case/2068650?file=4988998) utilizes an LSTM-based approach to model a conversation tree where the stance of an individual tweet depends on previous tweets in the tree. This Branch-LSTM approach allows the model to see the parent’s hidden state and even earlier context. However, it relies on static word embeddings and struggles on underrepresented classes, namely predicting **zero** "Deny" tweets in its reported confusion matrix.

## Approach

Our method combines three ideas:

1. **Domain-specific pre-training.** Instead of fine-tuning a general-purpose language model, we use [`cardiffnlp/twitter-roberta-base`](https://huggingface.co/cardiffnlp/twitter-roberta-base), a RoBERTa model pre-trained on ~58M tweets. This gives the encoder an inherent understanding of platform-specific phenomena such as hashtags, emojis, slang, and sarcasm; cues that frequently signal stance that may be missed by general LMs.
2. **Contextual branch encoding.** For each target tweet, we concatenate its preceding branch (source tweet → ancestor replies → target) into a single input sequence. This preserves conversational order in the same spirit as the Branch-LSTM, but unlike bag-of-words / averaged static embeddings it also preserves **word order within tweets**, so "man eats dog" is no longer representationally identical to "dog eats man".
3. **Weighted Cross-Entropy Loss.** RumourEval-2017 is heavily imbalanced: the "Comment" class makes up roughly 65% of all labels, while Support / Deny / Query (SDQ) are comparatively rare. We apply class-weighted cross-entropy so the model pays disproportionate attention to these high-value SDQ signals during training.

---

## Project Structure

```
RoBERTa-stance-analysis/
├── data/
│   ├── raw/
│   │   └── pheme-rumour-scheme-dataset/
│   │       ├── annotations/
│   │       │   └── en-scheme-annotations.json   # per-tweet stance labels (SDQC)
│   │       └── threads/en/{event}/{thread_id}/
│   │           ├── source-tweets/{id}.json       # root tweet JSON
│   │           ├── reactions/{id}.json           # reply tweet JSONs
│   │           └── structure.json               # nested dict encoding reply tree
│   └── processed/                               # (populated at runtime)
├── outputs/                                     # checkpoints and results (runtime)
├── notebooks/
│   └── colab_train.ipynb                        # end-to-end training on Google Colab
├── scripts/
│   └── download_data.sh
├── src/
│   ├── config.py
│   ├── data.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```

---

## Dataset

The [PHEME rumour scheme dataset](https://figshare.com/articles/dataset/PHEME_rumour_scheme_dataset_journalism_use_case/2068650) contains 297 Twitter threads across 8 English news events (charliehebdo, ferguson, germanwings-crash, ottawashooting, putinmissing, sydneysiege, ebola-essien, prince-toronto). Each thread is a conversation tree rooted at a source tweet with nested reply chains.

_EXTRACT DATASET INTO data/raw/_

Stance labels come from `en-scheme-annotations.json`, a newline-delimited JSON file with one record per tweet:

| Field                    | Description                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `responsetype-vs-source` | Reply stance: `agreed` → **support**, `disagreed` → **deny**, `appeal-for-more-information` → **query**, `comment` → **comment** |
| `support`                | Source-tweet stance toward the rumour (not used for SDQC classification)                                                         |

**Label distribution across 4,263 annotated reply tweets:**

| Class   | Count | %     |
| ------- | ----- | ----- |
| comment | 2,923 | 68.6% |
| support | 645   | 15.1% |
| query   | 361   | 8.5%  |
| deny    | 334   | 7.8%  |

**Evaluation protocol:** Leave-One-Event-Out (LOEO) — train on 7 events, test on the 8th, repeat for all 8. Primary metric is macro-F1 (treats all four classes equally regardless of frequency), matching the original Branch-LSTM paper.

---

## Source Code

### `src/config.py`

Central configuration for all paths and hyperparameters. Edit this file to change the model, batch size, learning rate, sequence length, etc. Key settings:

```python
MODEL_NAME   = "cardiffnlp/twitter-roberta-base"
NUM_LABELS   = 4
MAX_SEQ_LEN  = 256
BATCH_SIZE   = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS   = 6
LR_DECAY     = 0.9
```

---

### `src/data.py`

Loads and parses the raw PHEME dataset into a list of training examples. Key functions:

**`load_pheme_dataset(events=None) → list[dict]`**
Parses `en-scheme-annotations.json`, reads each tweet's text from its JSON file, and reconstructs the branch path (ancestor chain from source tweet down to immediate parent) for every annotated reply by recursively walking `structure.json`. Returns one dict per reply tweet:

```python
{
  "tweet_id":     str,
  "thread_id":    str,
  "event":        str,          # e.g. "charliehebdo"
  "text":         str,          # target tweet text
  "branch_texts": list[str],    # [root_text, ..., parent_text]
  "label":        int,          # 0=support 1=deny 2=query 3=comment
}
```

**`loeo_splits(examples) → list[tuple]`**
Groups examples by event and returns all 8 `(held_out_event, train_examples, test_examples)` folds for LOEO cross-validation.

---

### `src/dataset.py`

PyTorch `Dataset` and batch collator. Key components:

**`_build_input_ids(branch_texts, target_text, tokenizer, max_len)`**
Formats the branch + target into a single token sequence following RoBERTa's multi-segment convention:

```
<s> [root] </s></s> [ancestor] </s></s> ... </s></s> [parent] </s></s> [target] </s>
```

Truncation preserves the most semantically important context: if the full branch exceeds `max_len`, middle ancestors are dropped first; root and immediate parent are always kept. If even root + parent + target exceeds budget, the root is truncated rather than dropped entirely.

**`PhemeDataset(examples, tokenizer, max_len=256)`**
Standard `Dataset` returning `input_ids`, `attention_mask`, and `label` tensors per example.

**`collate_fn(batch, pad_token_id)`**
Pads variable-length sequences within a batch to the same length. Intended for use with `functools.partial`:

```python
from functools import partial
loader = DataLoader(ds, batch_size=16, collate_fn=partial(collate_fn, pad_token_id=tok.pad_token_id))
```

---

### `src/model.py`

Defines the `StanceClassifier` nn.Module — a thin wrapper around `cardiffnlp/twitter-roberta-base` with a classification head.

**Architecture:**

```
twitter-roberta-base encoder  (~124.6M params)
  → last_hidden_state[:, 0, :]   (<s> token, shape B × 768)
  → Dropout(p=0.1)
  → Linear(768 → 768)
  → tanh
  → Dropout(p=0.1)
  → Linear(768 → 4)
  → logits  (shape B × 4)
```

This is the standard `RobertaForSequenceClassification` head — the extra hidden layer + non-linearity gives the encoder room to specialise its `<s>` representation for the SDQC label space, which consistently helps on small fine-tuning datasets like RumourEval over a bare linear projection.

`forward(input_ids, attention_mask, labels=None)` returns a dict with key `logits` (always) and `loss` (only when `labels` is passed). When used in training, loss is **not** passed through `forward` — it is computed externally in `train.py` using weighted cross-entropy.

---

### `src/train.py`

End-to-end fine-tuning loop. Entry point:

**`run_loeo(output_dir=OUTPUTS)`**
Runs all 8 LOEO folds and prints a summary table of per-class F1 and macro-F1 for each held-out event. Saves per-fold artefacts under `outputs/fold_{event}/`:

| File                   | Contents                                                        |
| ---------------------- | --------------------------------------------------------------- |
| `best_model.pt`        | State dict of the epoch with highest macro-F1 on the test split |
| `results.json`         | Accuracy, macro-F1, and per-class F1 for the best epoch         |
| `confusion_matrix.png` | Confusion matrix heatmap                                        |

**Key training details:**

- **Optimizer:** AdamW with layer-wise LR decay (see below)
- **Scheduler:** Linear warmup for 10% of steps, then linear decay to 0. Skips a scheduler step whenever AMP detects inf/NaN grads and skips the optimizer step, keeping LR aligned with actual updates.
- **Loss:** `CrossEntropyLoss` with per-class weights computed as `sqrt(total / (4 × class_count))` from the training split only — weights are recomputed fresh for each fold. Plain inverse frequency over-corrected and starved the comment class; the sqrt variant keeps the max class-weight ratio under ~1.8x.
- **Sampler:** Uniform shuffle (`shuffle=True`). Earlier experiments combined `WeightedRandomSampler` with the loss weighting and the compounded ~80:1 effective ratio caused the model to never predict comment; the sampler was removed in favour of the milder loss-side correction above.
- **Mixed precision:** `torch.amp.autocast` + `GradScaler` when a CUDA device is available
- **Gradient clipping:** max norm 1.0
- **Tokenizer:** loaded with `add_prefix_space=True` so that each ancestor / target segment is encoded as if it were a continuation of running text — preserves the leading-space (`Ġ`) BPE markers that the encoder saw during pretraining.

To run:

```bash
cd src && python3 train.py
```

---

## Fine-tuning Improvements

### Layer-wise Learning Rate Decay

When fine-tuning a pre-trained transformer, all 124M parameters are updated, but not all layers should change at the same rate. Lower layers encode general linguistic knowledge (syntax, token meaning) learned from 58M tweets; those representations are already well-suited to the task and should be disturbed as little as possible. Upper layers encode task-specific abstractions and benefit from larger updates. Updating every layer with the same learning rate risks _catastrophic forgetting_, overwriting the Twitter-specific knowledge that makes `twitter-roberta-base` valuable.

Layer-wise LR decay assigns each layer a learning rate scaled by a fixed factor relative to the layer above it:

```
lr(depth) = base_lr × decay^(num_layers + 1 − depth)
```

With `base_lr = 1e-5` and `decay = 0.9` over 12 transformer layers:

| Component                | Depth | Learning Rate    |
| ------------------------ | ----- | ---------------- |
| Classifier head + pooler | 13    | `1.00e-5` (base) |
| Transformer layer 11     | 12    | `9.00e-6`        |
| Transformer layer 10     | 11    | `8.10e-6`        |
| ⋮                        | ⋮     | ⋮                |
| Transformer layer 0      | 1     | `2.82e-6`        |
| Embeddings               | 0     | `2.54e-6`        |

The head trains at the full rate while the embeddings train at ~25% of it. This is implemented in `get_layerwise_optimizer()`, which also applies the standard transformer convention of zeroing weight decay on bias and `LayerNorm` parameters.

The decay factor is controlled by `LR_DECAY` in `config.py` (default `0.9`). Setting it to `1.0` disables layer-wise decay and reverts to uniform learning rates.

---

### `src/evaluate.py`

Evaluation utilities used by `train.py` and usable standalone.

**`compute_metrics(true_labels, pred_labels) → dict`**
Returns `accuracy`, `macro_f1`, and `per_class_f1` (list of 4 values in support/deny/query/comment order).

**`print_report(true_labels, pred_labels)`**
Prints a full sklearn `classification_report` with per-class precision, recall, and F1.

**`save_confusion_matrix(true_labels, pred_labels, path)`**
Saves a seaborn heatmap of the 4×4 confusion matrix as a PNG.

After polishing our model more we can add extra metrics/graphs here for a more comprehensive evaluation.
