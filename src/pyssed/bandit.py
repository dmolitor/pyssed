from abc import ABC, abstractmethod
from typing import Dict


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
    def reward(self, arm: int) -> float:
        """Calculate the reward for a selected bandit arm.

        Returns the reward for a selected arm.

        Parameters
        ----------
        arm : int
            The index of the selected bandit arm.

        Returns
        -------
        float
            The resulting reward.
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
