from abc import ABC, abstractmethod
from typing import Iterator

import torch

from protogpt.tokenizer import BaseTokenizer


class BaseTextDataset(ABC):
    name = "UnnamedDataset"

    def __repr__(self) -> str:
        return f"{self.name}(n={len(self)})"

    @abstractmethod
    def __len__(self) -> int:
        """The number of characters in the dataset"""

    @property
    @abstractmethod
    def unique_characters(self) -> set[str]:
        """Return the unique characters in this dataset"""

    @abstractmethod
    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a batch for training or validation

        :param batch_size: The size of the batch
        :return: the X, Y tensors of shape (batch_size, block_size) each
        """


class InMemoryTextDataset(BaseTextDataset):
    def __init__(
        self, text: str, tokenizer: BaseTokenizer, device: torch.device, block_size: int
    ):
        self._text = text
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Cache the text on the target device memory
        self._encoded_text = self.tokenizer.encode(self._text).to(device)

    def __len__(self) -> int:
        return len(self._text)

    @property
    def unique_characters(self) -> set[str]:
        return set(self._text)

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._encoded_text
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y


class DatasetSplits:
    def __init__(self, train: BaseTextDataset, val: BaseTextDataset):
        self.train = train
        self.val = val

        train.name = "Train"
        val.name = "Val"

    def __iter__(self) -> Iterator[BaseTextDataset]:
        yield self.train
        yield self.val
