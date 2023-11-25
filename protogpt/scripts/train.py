from pathlib import Path
from typing import Literal

import torch
from argdantic import ArgParser
from pydantic import BaseModel, Field

from protogpt import training
from protogpt.datasets import DatasetSplits, InMemoryTextDataset
from protogpt.models import ProtoGPTModel
from protogpt.tokenizer import CharacterLevelTokenizer
from protogpt.training import TrainingLoopParams


class ScriptParams(BaseModel):
    dataset: Path
    device: Literal["cpu", "cuda", "cuda:0", "cuda:1"]
    learning_rate: float = 3e-4
    train_split_percent: float = 0.9
    block_size: int = Field(256, description="The context length")


class CombinedParams(ScriptParams, TrainingLoopParams):
    pass


parser = ArgParser(description="Train a smoll gpt!")


@parser.command(singleton=True)
def main(params: CombinedParams) -> None:
    torch.manual_seed(1337)
    device = torch.device(params.device)
    corpus = params.dataset.read_text()
    tokenizer = CharacterLevelTokenizer.create_from_corpus(corpus)

    # Create train and val datasets
    n_train_chars = int(len(corpus) * params.train_split_percent)
    dataset = DatasetSplits(
        train=InMemoryTextDataset(
            corpus[:n_train_chars], tokenizer, device, params.block_size
        ),
        val=InMemoryTextDataset(
            corpus[n_train_chars:], tokenizer, device, params.block_size
        ),
    )

    # Create the model and optimizer
    model = ProtoGPTModel(
        vocab_size=tokenizer.vocab_size, block_size=params.block_size, device=device
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)

    # Actually train
    training.simple_training_loop(
        model=model, dataset=dataset, optimizer=optimizer, params=params
    )

    # Try running some stuff
    xb, yb = dataset.val.get_batch(1)
    for token_batch in model.generate(xb, max_new_tokens=10000):
        print(tokenizer.decode(token_batch), end="")


if __name__ == "__main__":
    parser()
