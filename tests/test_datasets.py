from pathlib import Path

import torch

from protogpt.datasets import InMemoryTextDataset
from protogpt.tokenizer import CharacterLevelTokenizer


def test_in_memory_dataset() -> None:
    text = Path("datasets/tiny_shakespear.txt").read_text()
    dataset = InMemoryTextDataset(
        text=text,
        tokenizer=CharacterLevelTokenizer.create_from_corpus(text),
        device=torch.device("cpu"),
        block_size=200,
    )

    x, y = dataset.get_batch(3)
    assert x.shape == (3, 200)
    assert y.shape == (3, 200)
    assert len(dataset) == len(text)
    assert dataset.tokenizer.vocab_size == 65
