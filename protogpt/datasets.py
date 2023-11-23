from abc import ABC, abstractmethod


class TextDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """The number of characters in the dataset"""

    @property
    @abstractmethod
    def unique_characters(self) -> set[str]:
        """Return the unique characters in this dataset"""


class InMemoryTextDataset(TextDataset):
    def __init__(self, text: str):
        self._text = text

    def __len__(self) -> int:
        return len(self._text)

    @property
    def unique_characters(self) -> set[str]:
        return set(self._text)
