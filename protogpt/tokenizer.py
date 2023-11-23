from abc import ABC, abstractmethod

TOKENS = list[int]


class BaseTokenizer(ABC):
    @abstractmethod
    def decode(self, tokens: TOKENS) -> str:
        pass

    @abstractmethod
    def encode(self, text: str) -> TOKENS:
        pass

    @classmethod
    @abstractmethod
    def create_from_corpus(cls, text: str) -> "BaseTokenizer":
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the unique characters in this dataset"""


class CharacterLevelTokenizer(BaseTokenizer):
    """The simplest tokenizer. Each character == 1 token"""

    def __init__(self, mapping: dict[str, int]):
        """
        :param mapping: A mapping of {character: token_id}
        """
        self._mapping_to_tokens = mapping
        self._mapping_to_text = {v: k for k, v in mapping.items()}
        assert len(self._mapping_to_tokens) == len(self._mapping_to_text)

    def decode(self, tokens: TOKENS) -> str:
        return "".join((self._mapping_to_text[t] for t in tokens))

    def encode(self, text: str) -> TOKENS:
        return [self._mapping_to_tokens[c] for c in text]

    @classmethod
    def create_from_corpus(cls, text: str) -> "CharacterLevelTokenizer":
        unique_chars = sorted(set(text))
        return CharacterLevelTokenizer(
            mapping={c: i for i, c in enumerate(unique_chars)}
        )

    @property
    def vocab_size(self) -> int:
        return len(self._mapping_to_text)
