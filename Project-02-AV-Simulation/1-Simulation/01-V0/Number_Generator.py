import numpy as np

class General_Generator:
    """Generating numbers according to any distribution, only by calling the object. According to SCIPY"""
    def __init__(self, distribution, seed:int=None):
        """Args:
            distribution (scipy distribution): You should first make your distirution using scipy, then pass it here to be set with a specefic seed,
                                               and can be used in the further classes for simulation.
                                               Example: 
                                               from scipy.stats import 
                                               bernoulli, binom, poisson, norm, lognorm, uniform, expon, 
                                               gamma, beta, weibull_max, weibull_min, t, chi2, triang

            seed (int, optional): Random seed (numpy). Defaults to None.
        """
        self.distribution = distribution
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self):
        return self.distribution.rvs(random_state=self.rng)

class Exponential_Generator:
    """Generating numbers according to Exponential distribution, only by calling the object."""
    def __init__(self, mean:float, seed:int=None) -> None:
        """Args:
            seed (int): Random seed (numpy). Defaults to None.
            mean (float): mean of the data.
        """
        self.mean = mean
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self):
        return self.rng.exponential(self.mean)

class Deterministic_Generator:
    """Generating a constant number(mean), only by calling the object."""
    def __init__(self, mean):
        """Args:
            mean (float): mean of the data.
        """
        self.mean = mean

    def __call__(self):
        return self.mean

