from abc import ABC, abstractmethod
from typing import Dict

class Bandit(ABC):
    """
    An abstract class for Bandit algorithms used in the MAD algorithm. Each
    bandit algorithm must implement the abstract methods defined below.
    """
    @abstractmethod
    def control(self) -> int:
        """
        Returns the index of the arm that is the control arm. E.g. if the
        bandit is a 3-arm bandit with the first arm being the control arm,
        this should return the value 0.
        """
    
    @abstractmethod
    def k(self) -> int:
        """This method that returns the number of arms in the bandit"""
    
    @abstractmethod
    def probabilities(self) -> Dict[int, float]:
        """
        Returns a dictionary with the arm indices as keys and 
        selection probabilities for each arm as values. For example,
        if the bandit algorithm is UCB with three arms, and the third arm has
        the maximum confidence bound, then this should return the following
        dictionary: {0: 0., 1: 0., 1: 1.}, since UCB is deterministic.
        """

    @abstractmethod
    def reward(self, arm: int) -> float:
        """
        Returns the reward for a selected arm.
        
        Parameters:
        - arm (int): The index of the selected bandit arm
        """
    
    @abstractmethod
    def t(self) -> int:
        """
        This method returns the current time step of the bandit, and then
        increments the time step by 1. E.g. if the bandit has completed
        9 iterations, this should return the value 10. Time step starts
        at 1, not 0.
        """