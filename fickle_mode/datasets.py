from pathlib import Path

from torchvision.datasets import MNIST
from torchvision import transforms

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def mnist_train(cache_dir: Path) -> MNIST:
    return MNIST(cache_dir, train=True, download=True, transform=mnist_transform)


def mnist_val(cache_dir: Path) -> MNIST:
    return MNIST(cache_dir, train=False, download=True, transform=mnist_transform)
