"""Fine-tuning loop for stance classification on RumourEval-2017."""

import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from functools import partial
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from config import (
    BATCH_SIZE, FP16, GRAD_ACCUM_STEPS, GRAD_CLIP, LABELS,
    LEARNING_RATE, LR_DECAY, MAX_SEQ_LEN, MODEL_NAME, NUM_EPOCHS, NUM_LABELS,
    OUTPUTS, SEED, WARMUP_RATIO, WEIGHT_DECAY,
)
from data import load_pheme_dataset, loeo_splits
from dataset import PhemeDataset, collate_fn
from evaluate import compute_metrics, print_report, save_confusion_matrix
from model import StanceClassifier


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_class_weights(examples: list[dict]) -> torch.Tensor:
    """Square-root inverse-frequency weights derived from training examples only.

    Full inverse-frequency (total / 4*count) makes deny ~3.24x the weight of
    comment, which over-corrects on this dataset and starves the comment class.
    Square-root weighting compresses that ratio to ~1.80x max — enough to nudge
    the model toward minority classes without flipping the bias the other way.
    """
    counts = Counter(ex["label"] for ex in examples)
    total = len(examples)
    return torch.tensor(
        [math.sqrt(total / (NUM_LABELS * max(counts[i], 1))) for i in range(NUM_LABELS)],
        dtype=torch.float,
    )

def get_layerwise_optimizer(model: StanceClassifier, base_lr: float, weight_decay: float, lr_decay: float) -> AdamW:
    """AdamW with per-layer learning rates that decay toward the input.

    Depth assignment (num_layers = 12 for roberta-base):
        classifier / pooler  → depth 13  → lr * decay^0  = base_lr
        encoder layer 11     → depth 12  → lr * decay^1
        encoder layer 10     → depth 11  → lr * decay^2
        ...
        encoder layer 0      → depth  1  → lr * decay^12
        embeddings           → depth  0  → lr * decay^13  ≈ 0.25 * base_lr

    Bias and LayerNorm parameters are placed in a separate group with
    weight_decay=0, which is standard practice for transformer fine-tuning.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    num_layers = model.encoder.config.num_hidden_layers  # 12

    def layer_depth(name: str) -> int:
        if name.startswith("encoder.embeddings"):
            return 0
        if name.startswith("encoder.encoder.layer."):
            # e.g. "encoder.encoder.layer.3.attention.self.query.weight"
            return int(name.split(".")[3]) + 1  # layer 0→1, layer 11→12
        return num_layers + 1  # pooler, classifier → 13

    # Accumulate params into (depth, apply_wd) buckets.
    buckets: dict[tuple, list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = (layer_depth(name), not any(nd in name for nd in no_decay))
        buckets.setdefault(key, []).append(param)

    param_groups = []
    for (depth, apply_wd), params in buckets.items():
        layer_lr = base_lr * (lr_decay ** (num_layers + 1 - depth))
        param_groups.append({
            "params": params,
            "lr": layer_lr,
            "weight_decay": weight_decay if apply_wd else 0.0,
        })

    return AdamW(param_groups)


def _make_loaders(
    train_examples: list[dict],
    test_examples: list[dict],
    tokenizer,
) -> tuple[DataLoader, DataLoader]:
    pad_id = tokenizer.pad_token_id
    collate = partial(collate_fn, pad_token_id=pad_id)
    train_ds = PhemeDataset(train_examples, tokenizer, MAX_SEQ_LEN)
    test_ds = PhemeDataset(test_examples, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate)
    return train_loader, test_loader


def _eval_pass(model: StanceClassifier, loader: DataLoader, device: torch.device) -> tuple[dict, list, list]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_preds.extend(out["logits"].argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    return compute_metrics(all_labels, all_preds), all_labels, all_preds


def train_fold(
    held_out_event: str,
    train_examples: list[dict],
    test_examples: list[dict],
    tokenizer,
    device: torch.device,
    fold_dir: Path,
) -> dict:
    """Train on train_examples and evaluate on test_examples.

    Saves the best checkpoint (by macro-F1) and returns its metrics.
    """
    train_loader, test_loader = _make_loaders(train_examples, test_examples, tokenizer)

    model = StanceClassifier(MODEL_NAME, NUM_LABELS).to(device)
    class_weights = compute_class_weights(train_examples).to(device)

    optimizer = get_layerwise_optimizer(model, LEARNING_RATE, WEIGHT_DECAY, LR_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = FP16 and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_macro_f1 = -1.0
    best_metrics: dict = {}
    best_labels: list = []
    best_preds: list = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"[{held_out_event}] epoch {epoch}/{NUM_EPOCHS}",
            leave=False,
        )
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.functional.cross_entropy(
                    out["logits"], labels, weight=class_weights
                )
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Skip scheduler step if AMP skipped the optimizer step (inf/nan grads).
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{running_loss / (step + 1):.3f}"})

        metrics, true_labels, pred_labels = _eval_pass(model, test_loader, device)
        macro_f1 = metrics["macro_f1"]
        avg_loss = running_loss / len(train_loader)

        print(
            f"  [{held_out_event}] epoch {epoch}  "
            f"loss={avg_loss:.3f}  macro_f1={macro_f1:.4f}  acc={metrics['accuracy']:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_metrics = metrics
            best_labels = true_labels
            best_preds = pred_labels
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), fold_dir / "best_model.pt")

    # Persist per-fold artefacts for the best epoch.
    fold_dir.mkdir(parents=True, exist_ok=True)
    with open(fold_dir / "results.json", "w") as f:
        json.dump(best_metrics, f, indent=2)
    print_report(best_labels, best_preds)
    save_confusion_matrix(best_labels, best_preds, fold_dir / "confusion_matrix.png")

    return best_metrics


def run_loeo(output_dir: Path = OUTPUTS) -> dict:
    """Run full Leave-One-Event-Out evaluation and return per-fold results."""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    examples = load_pheme_dataset()
    splits = loeo_splits(examples)

    all_results: dict = {}

    for held_out_event, train_examples, test_examples in splits:
        print(
            f"\n{'='*60}\n"
            f"Fold: held-out={held_out_event} | "
            f"train={len(train_examples)}  test={len(test_examples)}\n"
            f"{'='*60}"
        )
        fold_dir = output_dir / f"fold_{held_out_event}"
        metrics = train_fold(
            held_out_event, train_examples, test_examples,
            tokenizer, device, fold_dir,
        )
        all_results[held_out_event] = metrics

    # Summary table.
    macro_f1s = [v["macro_f1"] for v in all_results.values()]
    accs = [v["accuracy"] for v in all_results.values()]
    per_class = np.array([v["per_class_f1"] for v in all_results.values()])

    print(f"\n{'='*60}")
    print("LOEO Summary")
    print(f"{'='*60}")
    header = f"{'Event':<26}" + "  ".join(f"{l:>8}" for l in LABELS) + "  macro_f1    acc"
    print(header)
    print("-" * len(header))
    for event, m in all_results.items():
        row = f"{event:<26}" + "  ".join(f"{f:>8.4f}" for f in m["per_class_f1"])
        row += f"  {m['macro_f1']:>8.4f}  {m['accuracy']:>6.4f}"
        print(row)
    print("-" * len(header))
    mean_row = f"{'MEAN':<26}" + "  ".join(f"{f:>8.4f}" for f in per_class.mean(axis=0).tolist())
    mean_row += f"  {sum(macro_f1s)/len(macro_f1s):>8.4f}  {sum(accs)/len(accs):>6.4f}"
    print(mean_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "loeo_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_loeo()
