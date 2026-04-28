"""Twitter-RoBERTa with a 4-way stance classification head."""

import torch
import torch.nn as nn
from transformers import AutoModel


class StanceClassifier(nn.Module):
    """twitter-roberta-base encoder + RoBERTa-style classification head.

    The head matches `transformers.RobertaForSequenceClassification`:
        pooled hidden state → dropout → dense(H→H) → tanh → dropout → out_proj(H→C)
    The extra hidden layer + non-linearity is the standard sequence-classification
    head for RoBERTa and consistently outperforms a bare linear projection on
    small fine-tuning datasets like RumourEval.

    Pooling strategies:
        "cls"          — use the <s> token's hidden state (standard).
        "target_mean"  — mean-pool hidden states over only the target tweet's
                         token positions (requires `target_mask` in forward).
                         Sometimes helps when the input is mostly context.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()
        if pooling not in {"cls", "target_mean"}:
            raise ValueError(f"unknown pooling strategy: {pooling!r}")
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

        # Match RobertaPreTrainedModel._init_weights: N(0, initializer_range^2),
        # biases zero. Default initializer_range is 0.02 for RoBERTa-base.
        std = self.encoder.config.initializer_range
        for layer in (self.dense, self.out_proj):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            target_mask:    (B, L) optional; required when pooling="target_mean".
            labels:         (B,) optional; if given, loss is computed and returned.

        Returns dict with keys:
            logits: (B, num_labels)
            loss:   scalar tensor, present only when labels is not None
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

        if self.pooling == "cls":
            x = hidden[:, 0, :]
        else:  # "target_mean"
            if target_mask is None:
                raise ValueError("target_mean pooling requires target_mask")
            mask = target_mask.to(hidden.dtype).unsqueeze(-1)  # (B, L, 1)
            denom = mask.sum(dim=1).clamp(min=1.0)             # (B, 1)
            x = (hidden * mask).sum(dim=1) / denom             # (B, H)

        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
