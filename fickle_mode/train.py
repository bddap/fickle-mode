from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.optim import Adadelta  # type: ignore
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import mnist_train, mnist_val
from .model import Model


def train(model, device, train_loader, optimizer):
    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader: DataLoader) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(
                output,
                target,
                reduction="sum",
            ).item()

            # get the index of the max log-probability
            pred = output.argmax(
                dim=1,
                keepdim=True,
            )
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # type: ignore
    return test_loss


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-cache", type=Path)
    args = parser.parse_args()

    dataset_train = mnist_train(args.dataset_cache)
    loader_train = DataLoader(dataset_train, batch_size=64)
    dataset_val = mnist_val(args.dataset_cache)
    loader_val = DataLoader(dataset_val, batch_size=1000)

    torch.manual_seed(1)
    device = torch.device("cpu")

    model = Model().to(device)
    optimizer = Adadelta(model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 14 + 1):
        loss = test(model, device, loader_val)
        print(f"val loss: {loss}, starting epoch {epoch}")
        train(model, device, loader_train, optimizer)
        scheduler.step()

    loss = test(model, device, loader_val)
    print(f"final val loss: {loss}")


if __name__ == "__main__":
    main()
