import torch.optim
from pydantic import BaseModel

from protogpt.datasets import TrainValDataset
from protogpt.models import BigramLanguageModel


class TrainingLoopParams(BaseModel):
    training_steps: int
    batch_size: int = 32
    block_size: int = 8


def simple_training_loop(
    model: BigramLanguageModel,
    dataset: TrainValDataset,
    optimizer: torch.optim.Optimizer,
    params: TrainingLoopParams,
) -> None:

    for _ in range(params.training_steps):
        # Sample a batch of data
        xb, yb = dataset.train.get_batch(params.batch_size, params.block_size)

        # Run inference the loss
        logits, loss = model(xb, yb)

        # Evaluate the loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(loss.item())
