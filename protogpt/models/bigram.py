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
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)

            loss = functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)

            # Focus on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)

            # Apply softmax to get probabilities
            probabilities = functional.softmax(logits, dim=-1)  # (B, C)

            # Sample from this distribution
            idx_next = torch.multinomial(probabilities, num_samples=1)  # (B, 1)

            # Append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
