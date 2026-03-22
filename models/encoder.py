"""
models/encoder.py

RoBERTa span encoder. Takes a list of EDU text strings, tokenises them,
runs RoBERTa, and returns the [CLS] embedding for each EDU as a (N, 768) tensor.

Used in the forward pass to replace the placeholder node features in HeteroData.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from typing import List


class SpanEncoder(nn.Module):
    """
    Encodes a batch of short text spans (EDUs) with RoBERTa-base.

    Architecture:
      - RoBERTa-base (frozen by default, unfrozen after warmup_epochs)
      - Takes [CLS] token embedding as span representation
      - Output: (N, 768) where N = number of EDUs in the document

    Freezing strategy:
      During the first `warmup_epochs` epochs only the R-GCN and task heads
      are trained. After that, the encoder is unfrozen with lr_encoder = 1e-4
      (10× smaller than the R-GCN lr). This prevents catastrophic forgetting
      of RoBERTa's pretrained representations early in training.
    """

    def __init__(
        self,
        model_name:   str  = "roberta-base",
        freeze:       bool = True,
        max_length:   int  = 128,     # EDUs are short — 128 tokens is enough
        device:       str  = "cpu",
    ):
        super().__init__()
        self.max_length = max_length
        self.device     = device
        self.hidden_dim = 768

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.roberta   = RobertaModel.from_pretrained(model_name)

        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.roberta.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.roberta.parameters():
            p.requires_grad = True

    def forward(self, edu_texts: List[str]) -> torch.Tensor:
        """
        Args:
            edu_texts: list of N EDU strings
        Returns:
            embeddings: (N, 768) float tensor
        """
        if not edu_texts:
            return torch.zeros(0, self.hidden_dim, device=self.device)

        # Tokenise all EDUs as a batch
        encoded = self.tokenizer(
            edu_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        _device = next(self.roberta.parameters()).device
        encoded = {k: v.to(_device) for k, v in encoded.items()}

        outputs = self.roberta(**encoded)
        # [CLS] token is at position 0
        cls_embeddings = outputs.last_hidden_state[:, 0, :]   # (N, 768)
        return cls_embeddings
