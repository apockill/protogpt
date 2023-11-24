from argparse import ArgumentParser
from pathlib import Path

import torch

from protogpt.datasets import InMemoryTextDataset, TrainValDataset
from protogpt.models import BigramLanguageModel
from protogpt.tokenizer import CharacterLevelTokenizer


def main() -> None:
    torch.manual_seed(1337)

    parser = ArgumentParser(description="Train a smoll gpt!")
    parser.add_argument("-d", "--dataset", type=Path, required=True)
    args = parser.parse_args()

    corpus = args.dataset.read_text()
    tokenizer = CharacterLevelTokenizer.create_from_corpus(corpus)

    # Create train and val datasets
    n_train_chars = int(len(corpus) * 0.9)
    dataset = TrainValDataset(
        train=InMemoryTextDataset(corpus[:n_train_chars], tokenizer),
        val=InMemoryTextDataset(corpus[n_train_chars:], tokenizer),
    )

    # Create the model
    model = BigramLanguageModel(tokenizer.vocab_size)

    # Actually train
    xb, yb = dataset.train.get_batch(4, 8)
    logits, loss = model(xb, yb)



if __name__ == "__main__":
    main()
