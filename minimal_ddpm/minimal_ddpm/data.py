import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class MixtureGaussian:
    _parameters: list[tuple[float, float]]
    _weights: list[float]

    def __init__(self, parameters: list[tuple[float, float, float]]):
        self._parameters = []
        self._weights = []
        assert isinstance(parameters, list)
        for parameter in parameters:
            assert isinstance(parameter, tuple)
            assert len(parameter) == 3
            m, s, w = parameter  # Mean, standard deviation and weight
            assert isinstance(m, float) and isinstance(s, float)
            self._parameters.append((m, s))
            assert isinstance(w, float)
            self._weights.append(w)
        assert sum(self._weights) == 1.0

    def sample(self, size: int) -> list[float]:
        samples = []
        for _ in range(size):
            parameter = self._parameters[np.random.choice(len(self._parameters), p=self._weights)]
            m, s = parameter
            samples.append(np.random.normal(loc=m, scale=s))
        return samples


class SillyDistribution:
    _bin_width: float
    _bin_centers: list[float]
    _bin_masses: list[float]

    def __init__(self, domain: tuple[float, float], bin_num: int, mass_range: tuple[float, float]):
        domain_min, domain_max = domain
        assert domain_min < domain_max
        mass_min, mass_max = mass_range
        assert 0 < mass_min < mass_max
        # Generate bins
        self._bin_width = (domain_max - domain_min) / (bin_num - 1)
        self._bin_centers = []
        self._bin_masses = []
        for i in range(bin_num):
            bin_center = domain_min + i * self._bin_width
            bin_mass = np.random.uniform(mass_min, mass_max)
            self._bin_centers.append(bin_center)
            self._bin_masses.append(bin_mass)
        # Normalize the masses so that they sum to 1
        total_mass = sum(self._bin_masses)
        for i in range(len(self._bin_masses)):
            self._bin_masses[i] /= total_mass

    def sample(self) -> float:
        bin_center = np.random.choice(self._bin_centers, p=self._bin_masses)
        value = np.random.uniform(bin_center - self._bin_width, bin_center + self._bin_width)
        return value


def plot_histogram(
    samples: list[float],
    bin_num: int,
    domain: tuple[float, float] | None = None,
    top: float | None = None,
) -> np.ndarray:
    if domain is None:
        domain = (min(samples), max(samples))
    plt.hist(samples, bins=bin_num, range=domain)
    if top is not None:
        plt.ylim(top=top)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.clf()
    buffer.seek(0)
    image = np.array(Image.open(buffer))
    return image
