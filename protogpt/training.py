from pathlib import Path

import torch.optim
from pydantic import BaseModel
from tqdm import tqdm

from protogpt.datasets import BaseTextDataset, DatasetSplits
from protogpt.models import BaseGenerativeTextModel


class TrainingLoopParams(BaseModel):
    batch_size: int = 64
    training_steps: int = 5000

    # Eval params
    eval_iters: int = 200
    eval_interval: int = 500

    checkpoint_interval: int = 500
    checkpoint_dir: Path = Path("checkpoints")


@torch.no_grad()
def estimate_loss(
    model: BaseGenerativeTextModel, dataset: DatasetSplits, params: TrainingLoopParams
) -> dict[BaseTextDataset, torch.Tensor]:
    """Return the average loss for each split as a dict"""

    split_losses: dict[BaseTextDataset, torch.Tensor] = {}

    model.eval()
    for split in dataset:
        losses = torch.zeros(params.eval_iters)
        for k in range(params.eval_iters):
            xb, yb = split.get_batch(params.batch_size)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()

        split_losses[split] = losses.mean()
    model.train()

    return split_losses


def load_latest_checkpoint(model: BaseGenerativeTextModel, directory: Path) -> None:
    """Load the latest checkpoint from the directory into the model"""

    checkpoints = sorted(directory.glob("*.pt"))
    if len(checkpoints) == 0:
        return None
    latest_checkpoint = checkpoints[-1]
    print(f"Loading checkpoint {latest_checkpoint}")
    model.load_state_dict(torch.load(latest_checkpoint))


def save_checkpoint(model: BaseGenerativeTextModel, directory: Path, step: int) -> Path:
    """Save a checkpoint of the model to the directory"""
    checkpoint_filename = directory / f"checkpoint_{step}.pt"
    torch.save(model.state_dict(), checkpoint_filename)
    return checkpoint_filename


def simple_training_loop(
    model: BaseGenerativeTextModel,
    dataset: DatasetSplits,
    optimizer: torch.optim.Optimizer,
    params: TrainingLoopParams,
) -> None:
    params.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    load_latest_checkpoint(model, params.checkpoint_dir)

    for step in tqdm(range(params.training_steps)):
        # Every so often print out the losses against each split
        if step % params.eval_interval == 0 or step == params.training_steps - 1:
            losses = estimate_loss(model, dataset, params)
            loss_report = ", ".join(
                [f"{split.name}: {loss}" for split, loss in losses.items()]
            )
            print(loss_report)

        if step % params.checkpoint_interval == 0 or step == params.training_steps - 1:
            save_checkpoint(model, params.checkpoint_dir, step)

        # Sample a batch of data
        xb, yb = dataset.train.get_batch(params.batch_size)

        # Run inference the loss
        logits, loss = model(xb, yb)

        # Evaluate the loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
