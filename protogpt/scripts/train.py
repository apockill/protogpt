from pathlib import Path

import torch
from argdantic import ArgParser
from pydantic import BaseModel

from protogpt.datasets import InMemoryTextDataset, TrainValDataset
from protogpt.models import BigramLanguageModel
from protogpt.tokenizer import CharacterLevelTokenizer


class TrainParameters(BaseModel):
    dataset: Path
    learning_rate: float = 1e-3
    train_split_percent: float = 0.9


parser = ArgParser(description="Train a smoll gpt!")


@parser.command(singleton=True)
def main(params: TrainParameters) -> None:
    torch.manual_seed(1337)

    corpus = params.dataset.read_text()
    tokenizer = CharacterLevelTokenizer.create_from_corpus(corpus)

    # Create train and val datasets
    n_train_chars = int(len(corpus) * params.train_split_percent)
    dataset = TrainValDataset(
        train=InMemoryTextDataset(corpus[:n_train_chars], tokenizer),
        val=InMemoryTextDataset(corpus[n_train_chars:], tokenizer),
    )

    # Create the model and optimizer
    model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)

    # Actually train
    xb, yb = dataset.train.get_batch(4, 8)
    logits, loss = model(xb, yb)



if __name__ == "__main__":
    parser()
