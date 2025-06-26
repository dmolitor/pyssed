from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Callable, Dict


class Bandit(ABC):
    """
    An abstract class for Bandit algorithms used in the MAD algorithm.

    Each bandit algorithm that inherits from this class must implement all the
    abstract methods defined in this class.

    Notes
    -----
    See the detailed method documentation for in-depth explanations.
    """

    @abstractmethod
    def control(self) -> int:
        """Get the index of the bandit control arm.

        Returns
        -------
        int
            The index of the arm that is the control arm. E.g. if the
            bandit is a 3-arm bandit with the first arm being the control arm,
            this should return the value 0.
        """

    @abstractmethod
    def k(self) -> int:
        """Get the number of bandit arms.

        int
            The number of arms in the bandit.
        """

    @abstractmethod
    def probabilities(self) -> Dict[int, float]:
        """Calculate bandit arm assignment probabilities.

        Returns
        -------
        Dict[int, float]
            A dictionary where keys are arm indices and values are the
            corresponding probabilities. For example, if the bandit algorithm
            is UCB with three arms, and the third arm has the maximum
            confidence bound, then this should return the following dictionary:
            `{0: 0., 1: 0., 2: 1.}`, since UCB is deterministic.
        """

    @abstractmethod
    def reward(self, arm: int) -> "Reward":
        """Calculate the reward for a selected bandit arm.

        Returns the reward for a selected arm.

        Parameters
        ----------
        arm : int
            The index of the selected bandit arm.

        Returns
        -------
        Reward
            The resulting Reward containing any individual-level covariates
            and the observed reward.
        """

    @abstractmethod
    def t(self) -> int:
        """Get the current time step of the bandit.

        This method returns the current time step of the bandit, and then
        increments the time step by 1. E.g. if the bandit has completed
        9 iterations, this should return the value 10. Time steps start
        at 1, not 0.

        Returns
        -------
        int
            The current time step.
        """


class Reward:
    """
    A simple class for reward functions.

    Each reward function should return a reward object. For covariate adjusted
    algorithms, the reward should contain both the outcome as well as the
    corresponding covariates. For non-covariate-adjusted algorithms, only the
    outcome should be specified.

    Attributes
    ----------
    outcome : float
        The outcome of the reward function.
    covariates : pd.DataFrame | None
        (Optional) The corresponding individual-level covariates.
    """

    def __init__(self, outcome: float, covariates: pd.DataFrame | None = None):
        self.outcome = outcome
        self.covariates = covariates


class TSBernoulli(Bandit):
    """
    A class implementing Thompson Sampling on Bernoulli data with a Beta prior.

    This is an example implementation of the `Bandit` meta-class.

    Parameters
    ----------
    k : int
        The number of bandit arms.
    control : int
        The (0-based) index of the control arm.
    reward : Callable[[int], Reward]
        This function should take one input (the selected arm index) and output
        a Reward object containing the reward and (optionally) any covariates.
    optimize : {"max", "min"}
        Should Thompson Sampling select the arm that maximizes or minimizes the
        posterior probability of success.
    """

    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], float],
        optimize: str = "max",
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._means = {x: 0.0 for x in range(k)}
        self._optimize = optimize
        self._params = {x: {"alpha": 1, "beta": 1} for x in range(k)}
        self._rewards = {x: [] for x in range(k)}
        self._reward_fn = reward
        self._t = 1

    def calculate_probs(self) -> Dict[int, float]:
        sample_size = 1
        samples = np.column_stack(
            [
                np.random.beta(
                    a=self._params[idx]["alpha"],
                    b=self._params[idx]["beta"],
                    size=sample_size,
                )
                for idx in self._active_arms
            ]
        )
        if self._optimize == "max":
            optimal_indices = np.argmax(samples, axis=1)
        elif self._optimize == "min":
            optimal_indices = np.argmin(samples, axis=1)
        else:
            raise ValueError("`self._optimal` must be one of: ['max', 'min']")
        win_counts = {
            idx: np.sum(optimal_indices == i) / sample_size
            for i, idx in enumerate(self._active_arms)
        }
        return win_counts

    def control(self) -> int:
        return self._control

    def k(self) -> int:
        return self._k

    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(
            self._active_arms
        ), "Mismatch in `len(self._active_arms)` and `self.k()`"
        probs = self.calculate_probs()
        return probs

    def reward(self, arm: int) -> Reward:
        reward: Reward = self._reward_fn(arm)
        if not isinstance(reward, Reward):
            raise ValueError(
                "The provided reward function must return a `Reward` object"
            )
        self._rewards[arm].append(reward.outcome)
        if reward.outcome == 1:
            self._params[arm]["alpha"] += 1
        else:
            self._params[arm]["beta"] += 1
        self._means[arm] = self._params[arm]["alpha"] / (
            self._params[arm]["alpha"] + self._params[arm]["beta"]
        )
        return reward

    def t(self) -> int:
        step = self._t
        self._t += 1
        return step
