from pathlib import Path
from typing import Literal

import torch
from argdantic import ArgParser
from pydantic import BaseModel

from protogpt import training
from protogpt.datasets import InMemoryTextDataset, TrainValDataset
from protogpt.models import BigramLanguageModel
from protogpt.tokenizer import CharacterLevelTokenizer
from protogpt.training import TrainingLoopParams


class ScriptParams(BaseModel):
    dataset: Path
    device: Literal["cpu", "cuda"]
    learning_rate: float = 1e-3
    train_split_percent: float = 0.9


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
    dataset = TrainValDataset(
        train=InMemoryTextDataset(corpus[:n_train_chars], tokenizer, device),
        val=InMemoryTextDataset(corpus[n_train_chars:], tokenizer, device),
    )

    # Create the model and optimizer
    model = BigramLanguageModel(tokenizer.vocab_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)

    # Actually train
    training.simple_training_loop(
        model=model, dataset=dataset, optimizer=optimizer, params=params
    )

    # Try running some stuff
    xb, yb = dataset.val.get_batch(4, 8)
    logits, loss = model(xb, yb)
    generated = model.generate(xb, max_new_tokens=500)
    generated_decoded = [tokenizer.decode(g) for g in generated]



if __name__ == "__main__":
    parser()
