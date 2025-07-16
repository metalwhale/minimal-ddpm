import pathlib

from minimal_ddpm.data import MultimodalDistribution
from minimal_ddpm.model import MinimalDdpm, train


STORAGE_PATH = pathlib.Path(__file__).parent.parent / "storage"


def main():
    target_distribution = MultimodalDistribution([(-2.0, 1.0, 0.4), (2.0, 0.5, 0.6)])
    model = MinimalDdpm()
    train(STORAGE_PATH, target_distribution, model)


if __name__ == "__main__":
    main()
