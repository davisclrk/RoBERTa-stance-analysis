"""PyTorch Dataset and collator for PHEME stance classification."""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def _build_input_ids(
    branch_texts: list[str], target_text: str, tokenizer, max_len: int,
) -> tuple[list[int], list[int]]:
    """Construct token IDs and a target-tokens mask for a branch + target input.

    Returns:
        input_ids:   the full token sequence (CLS, ancestors with sep pairs,
                     target, EOS).
        target_mask: 1 over the target tweet's content tokens, 0 elsewhere.
                     Used by `model.StanceClassifier` when POOLING="target_mean"
                     to mean-pool hidden states over only the target positions.

    Output structure:
        <s> [root] </s></s> [anc_1] </s></s> ... </s></s> [parent] </s></s> [target] </s>

    If the full branch exceeds the token budget, ancestors are dropped starting
    from the middle outward, always preserving the root and the immediate parent.
    If root + parent + target still exceeds budget, the root is truncated to fit.
    """
    cls_id = tokenizer.cls_token_id   # 0
    sep_id = tokenizer.sep_token_id   # 2

    # Encode target; cap at half the budget so it never crowds out all context.
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_ids = target_ids[: max_len // 2]

    # Overhead layout: CLS + [ancestors + their SEP pairs] + SEP SEP + target + EOS
    # For k selected ancestors the special-token cost is: 1 + 2k + 1 = 2k + 2.
    # We compute how many ancestor tokens we can fit and then select accordingly.

    if not branch_texts:
        # No context: <s> target </s>
        ids = [cls_id] + target_ids + [sep_id]
        mask = [0] + [1] * len(target_ids) + [0]
        return ids[:max_len], mask[:max_len]

    # Encode every ancestor; total length is computed by `_total_len()` below.
    all_anc_ids = [tokenizer.encode(t, add_special_tokens=False) for t in branch_texts]

    def _total_len(anc_id_lists: list[list[int]]) -> int:
        # CLS + each-anc-content + 2-seps-per-anc + target + EOS
        return 1 + sum(len(a) + 2 for a in anc_id_lists) + len(target_ids) + 1

    # Try full branch first.
    if _total_len(all_anc_ids) <= max_len:
        selected = all_anc_ids
    else:
        # Drop middle ancestors, keep root + immediate parent.
        root_ids = all_anc_ids[0]
        parent_ids = all_anc_ids[-1]
        if len(all_anc_ids) == 1:
            selected = [root_ids]
        elif _total_len([root_ids, parent_ids]) <= max_len:
            # Greedily reinsert middle ancestors nearest-first.
            selected = [root_ids, parent_ids]
            for anc_ids in reversed(all_anc_ids[1:-1]):
                candidate = [root_ids] + [anc_ids] + selected[1:]
                if _total_len(candidate) <= max_len:
                    selected = candidate
                else:
                    break
        else:
            # root + parent alone still doesn't fit — truncate root.
            # Budget for root: max_len - CLS - (2+len(parent)) - target - EOS
            # = max_len - 1 - 2 - len(parent_ids) - 2 - len(target_ids) - 1
            root_budget = max_len - 1 - 2 - len(parent_ids) - 2 - len(target_ids) - 1
            if root_budget > 0:
                selected = [root_ids[:root_budget], parent_ids]
            else:
                # Even parent alone might not fit; truncate it.
                parent_budget = max_len - 1 - 2 - len(target_ids) - 1
                selected = [parent_ids[:max(1, parent_budget)]]

    # Build the final token sequence and the target-tokens mask in lockstep.
    ids: list[int] = [cls_id]
    mask: list[int] = [0]
    for anc_ids in selected:
        ids.extend(anc_ids)
        mask.extend([0] * len(anc_ids))
        ids.extend([sep_id, sep_id])
        mask.extend([0, 0])
    ids.extend(target_ids)
    mask.extend([1] * len(target_ids))
    ids.append(sep_id)
    mask.append(0)

    return ids[:max_len], mask[:max_len]  # safety clamp


class PhemeDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_len: int = 256):
        # Tokenize once at construction so each epoch is a tensor lookup, not a
        # re-tokenization. Also makes num_workers > 0 safe (no shared tokenizer).
        # target_mask is always computed (cheap); the model only consults it when
        # POOLING="target_mean".
        self._items: list[dict] = []
        for ex in examples:
            input_ids, target_mask = _build_input_ids(
                ex["branch_texts"], ex["text"], tokenizer, max_len,
            )
            self._items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "target_mask": torch.tensor(target_mask, dtype=torch.long),
                "label": torch.tensor(ex["label"], dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Pad a batch of variable-length sequences to the same length."""
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    target_mask = pad_sequence(
        [item["target_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.stack([item["label"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "labels": labels,
    }
