"""Twitter-RoBERTa with a 4-way stance classification head."""

import torch
import torch.nn as nn
from transformers import AutoModel


class StanceClassifier(nn.Module):
    """twitter-roberta-base encoder + RoBERTa-style classification head.

    The head matches `transformers.RobertaForSequenceClassification`:
        <s> hidden state → dropout → dense(H→H) → tanh → dropout → out_proj(H→C)
    The extra hidden layer + non-linearity is the standard sequence-classification
    head for RoBERTa and consistently outperforms a bare linear projection on
    small fine-tuning datasets like RumourEval.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

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
        x = outputs.last_hidden_state[:, 0, :]  # <s> token
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
