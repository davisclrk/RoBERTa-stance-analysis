"""Twitter-RoBERTa with a 4-way stance classification head."""

import torch
import torch.nn as nn
from transformers import AutoModel


class StanceClassifier(nn.Module):
    """twitter-roberta-base encoder + linear classification head.

    Pools the CLS token representation from the last hidden state and
    projects it to num_labels classes via dropout + linear.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            labels:         (B,) optional; if given, loss is computed and returned.

        Returns dict with keys:
            logits: (B, num_labels)
            loss:   scalar tensor, present only when labels is not None
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token is at position 0
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_repr))

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
