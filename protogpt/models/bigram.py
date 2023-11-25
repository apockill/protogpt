from typing import Generator

import torch
import torch.nn as nn
from torch.nn import functional

from .base import BaseGenerativeTextModel


class BigramLanguageModel(BaseGenerativeTextModel):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape

            # Reshape logits and targets so pytorch can operate on them
            logits_reshaped = logits.view(b * t, c)
            targets_reshaped = targets.view(b * t)
            loss = functional.cross_entropy(logits_reshaped, targets_reshaped)
        return logits, loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int
    ) -> Generator[torch.Tensor, None, None]:
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)

            # Focus on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)

            # Apply softmax to get probabilities
            probabilities = functional.softmax(logits, dim=-1)  # (B, C)

            # Sample from this distribution
            new_column = torch.multinomial(probabilities, num_samples=1)  # (B, 1)

            # Shift the current context by one, and append new newly generated tokens
            idx[:, :-1] = idx[:, 1:].clone()
            idx[:, -1] = new_column
            yield new_column
