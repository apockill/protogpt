from argparse import ArgumentParser
from pathlib import Path

from protogpt.datasets import InMemoryTextDataset, TrainValDataset
from protogpt.tokenizer import CharacterLevelTokenizer


def main() -> None:
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


if __name__ == "__main__":
    main()
