import torch.optim
from pydantic import BaseModel
from tqdm import tqdm

from protogpt.datasets import BaseTextDataset, DatasetSplits
from protogpt.models import BaseGenerativeTextModel


class TrainingLoopParams(BaseModel):
    training_steps: int
    batch_size: int = 32
    block_size: int = 16
    eval_iters: int = 200
    eval_interval: int = 500


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
            xb, yb = split.get_batch(params.batch_size, params.block_size)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()

        split_losses[split] = losses.mean()
    model.train()

    return split_losses


def simple_training_loop(
    model: BaseGenerativeTextModel,
    dataset: DatasetSplits,
    optimizer: torch.optim.Optimizer,
    params: TrainingLoopParams,
) -> None:
    for step in tqdm(range(params.training_steps)):
        # Every so often print out the losses against each split
        if step % params.eval_interval == 0 or step == params.training_steps - 1:
            losses = estimate_loss(model, dataset, params)
            loss_report = ", ".join(
                [f"{split.name}: {loss}" for split, loss in losses.items()]
            )
            print(loss_report)

        # Sample a batch of data
        xb, yb = dataset.train.get_batch(params.batch_size, params.block_size)

        # Run inference the loss
        logits, loss = model(xb, yb)

        # Evaluate the loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
