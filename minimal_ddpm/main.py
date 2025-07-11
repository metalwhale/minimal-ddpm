import pathlib

from minimal_ddpm.data import MultimodalDistribution
from minimal_ddpm.model import MinimalDdpm, train


STORAGE_PATH = pathlib.Path(__file__).parent.parent / "storage"


def main():
    target_distribution = MultimodalDistribution([(-2.0, 1.0, 0.5), (1.0, 0.5, 0.5)])
    # target_distribution = SillyDistribution((-10, 10), 20, (1, 1000))
    model = MinimalDdpm()
    train(STORAGE_PATH, target_distribution, model)


if __name__ == "__main__":
    main()
