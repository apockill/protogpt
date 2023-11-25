from typing import Generator

import torch
import torch.nn as nn
from torch.nn import functional

from .base import BaseGenerativeTextModel


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(
        self, head_size: int, input_features: int, block_size: int, dropout: float
    ):
        super().__init__()
        self.key = nn.Linear(input_features, head_size, bias=False)
        self.query = nn.Linear(input_features, head_size, bias=False)
        self.value = nn.Linear(input_features, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, embedding_size = x.shape

        keys = self.key(x)  # (B, T, C)
        queries = self.query(x)  # (B, T, C)

        # Compute affinities (attention scores). (B, T, H) @ (B, H, T) --> (B, T, T)
        affinities = queries @ keys.transpose(-2, -1)
        # You multiply by 1/sqrt(embedding_size) because otherwise affinities will scale
        # with the size of the embeddings. This is bad for softmax, and leads to very
        # spiky, one-hot vectors. By dividing, you diffuse the influence of the size.
        # This is called "scaled attention"
        affinities *= embedding_size**-0.5
        affinities = affinities.masked_fill(
            self.tril[:n_tokens, :n_tokens] == 0, float("-inf")
        )  # B, T, T)
        affinities = functional.softmax(affinities, dim=-1)
        affinities = self.dropout(affinities)

        # Perform weighted aggregation of values
        values = self.value(x)  # (B, T, C)
        output = affinities @ values  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return output  # type: ignore


class MultiHeadAttention(nn.Module):
    """One head of self-attention"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        input_features: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size, input_features, block_size, dropout=dropout)
                for _ in range(num_heads)
            ]
        )

        # Useful so that the multiple heads always output input_features len
        self.projection = nn.Linear(input_features, input_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate all the outputs on the channel dimension
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.projection(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity, with a higher dimension.
    Then followed by another linear layer, going back to the original dimensionality.
    """

    def __init__(
        self,
        embedding_size: int,
        dropout: float,
        inner_dimension_scaling_factor: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, inner_dimension_scaling_factor * embedding_size),
            nn.ReLU(),
            # Project back into the residual pathway
            nn.Linear(inner_dimension_scaling_factor * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore


class Block(nn.Module):
    """A transformer block: communication followed by computation.

    Uses residuals to help training
    """

    def __init__(
        self,
        num_heads: int,
        input_features: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()

        # Create n heads that add up to a given embedding size
        head_size = input_features // num_heads
        self.self_attention_heads = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            input_features=input_features,
            block_size=block_size,
            dropout=dropout,
        )

        # Useful so that all the data extracted from the attention can coalesce more
        self.feed_forward = FeedForward(input_features, dropout=dropout)

        self.layer_norm_1 = nn.LayerNorm(input_features)
        self.layer_norm_2 = nn.LayerNorm(input_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum X to allow residual connections
        x = self.layer_norm_1(x)
        x = x + self.self_attention_heads(x)
        x = self.layer_norm_2(x)
        x = x + self.feed_forward(x)
        return x


class ProtoGPTModel(BaseGenerativeTextModel):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        device: torch.device,
        embedding_size: int = 384,
        num_heads: int = 6,
        num_block_layers: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device

        # Encodes the "identity" of the tokens
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)

        # Encodes the "position" of the tokens
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(
            *[
                Block(
                    num_heads=num_heads,
                    input_features=embedding_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_block_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n_batches, n_tokens = idx.shape
        assert (
            n_tokens == self.block_size
        ), f"Invalid context length: {n_tokens}, expected {self.block_size}"

        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(n_tokens, device=self.device)
        )  # (T, C)
        x = token_embeddings + pos_embeddings  # (B, T, C)
        x = self.blocks(x)  # Apply multiple layers of self-attention (B, T, C)
        x = self.final_layer_norm(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape logits and targets so pytorch can operate on them
            logits_reshaped = logits.view(n_batches * n_tokens, self.vocab_size)
            targets_reshaped = targets.view(n_batches * n_tokens)
            loss = functional.cross_entropy(logits_reshaped, targets_reshaped)
        return logits, loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int
    ) -> Generator[torch.Tensor, None, None]:
        """Generate tokens and yield them as-generated"""

        for _ in range(max_new_tokens):
            # Get predictions, make sure to select only up to the context length
            logits, loss = self(idx[:, -self.block_size :])

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
