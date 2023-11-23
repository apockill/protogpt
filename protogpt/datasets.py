from abc import ABC, abstractmethod

from protogpt.tokenizer import BaseTokenizer


class BaseTextDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """The number of characters in the dataset"""

    @property
    @abstractmethod
    def unique_characters(self) -> set[str]:
        """Return the unique characters in this dataset"""


class InMemoryTextDataset(BaseTextDataset):
    def __init__(self, text: str, tokenizer: BaseTokenizer):
        self._text = text
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._text)

    @property
    def unique_characters(self) -> set[str]:
        return set(self._text)


class TrainValDataset:
    def __init__(self, train: BaseTextDataset, val: BaseTextDataset):
        self.train = train
        self.val = val
