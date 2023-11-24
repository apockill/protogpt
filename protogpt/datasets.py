from abc import ABC, abstractmethod

import torch

from protogpt.tokenizer import BaseTokenizer


class BaseTextDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """The number of characters in the dataset"""

    @property
    @abstractmethod
    def unique_characters(self) -> set[str]:
        """Return the unique characters in this dataset"""

    @abstractmethod
    def get_batch(
        self, batch_size: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a batch for training or validation

        :param batch_size: The size of the batch
        :param block_size: The maximum context length of predictions
        :return: the X, Y tensors of shape (batch_size, block_size) each
        """


class InMemoryTextDataset(BaseTextDataset):
    def __init__(self, text: str, tokenizer: BaseTokenizer, device: torch.device):
        self._text = text
        self.tokenizer = tokenizer

        # Cache the text on the target device memory
        self._encoded_text = self.tokenizer.encode(self._text).to(device)

    def __len__(self) -> int:
        return len(self._text)

    @property
    def unique_characters(self) -> set[str]:
        return set(self._text)

    def get_batch(
        self, batch_size: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._encoded_text
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y


class TrainValDataset:
    def __init__(self, train: BaseTextDataset, val: BaseTextDataset):
        self.train = train
        self.val = val
