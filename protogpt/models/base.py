from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseGenerativeTextModel(nn.Module, ABC):
    """

    Nomenclature:
    B: Batch size
    T: Time, or the length of the context
    C: Channel
    """

    @abstractmethod
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """

        :param idx: Idx and targets are both (batch_size, time) tensors of integers
        :param targets: Same shape as idx
        :return: A (batch_size, time, channel) shaped tensor and the loss
        """

    @abstractmethod
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        :param idx: A (B, T) array of indices in the current context
        :param max_new_tokens: How many tokens to produce, maximum
        :return: Idx, with the newly generated tokens concatenated
        """
