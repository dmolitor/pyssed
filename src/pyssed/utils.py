import numpy as np
import pandas as pd
from typing import Any, Dict, List


def check_shrinkage_rate(t: int, delta_t: float):
    """Checks the shrinkage rate delta_t defined in Liang and Bojinov"""
    assert t <= 1 or delta_t > 1 / (
        t ** (1 / 4)
    ), "Sequence is converging to 0 too quickly"


def cs_radius(
    var: List[float],
    t: int,
    t_star: int,
    alpha: float = 0.05,
    mc_adjust: str | None = None,
    n_arms: int = 1,
) -> float:
    """
    Confidence sequence radius

    Parameters:
    -----------
    var : List[float]
        A list of individual treatment effect variances (upper bounds)
    t : int
        The current time-step of the algorithm
    t_star : int
        The time-step at which we want to optimize the CSs to be tightest
    alpha : float
        The size of the statistical test
    mc_adjust : str or None
        The type of multiple comparison adjustment to perform
    n_arms : int
        The number of treatment arms. Necessary for multiple
        comparison adjustment

    Return:
    -------
    The radius of the Confidence Sequence. Aka the value V such that
    tau (ATE estimate) ± V is a valid alpha-level CS.
    """
    S = np.sum(var)
    if mc_adjust is None:
        alpha = alpha
    elif mc_adjust.lower() == "bonferroni":
        alpha = alpha / n_arms
    eta = np.sqrt((-2 * np.log(alpha) + np.log(-2 * np.log(alpha) + 1)) / t_star)
    rad = np.sqrt(
        2
        * (S * (eta**2) + 1)
        / ((t**2) * (eta**2))
        * np.log(np.sqrt(S * (eta**2) + 1) / alpha)
    )
    return rad.astype(float)


def last(x: List[Any]) -> Any:
    """Returns the last element of a list. Returns None if empty list."""
    if len(x) < 1:
        return None
    return x[len(x) - 1]


def prep_dummies(
    covariates: pd.DataFrame,
    categories: List,
    add_constant: bool = True,
    arm: None | Any = None,
) -> pd.DataFrame:
    """
    Prepare covariate data for model training by expanding treatment arm
    assignment into dummy variables.

    Parameters
    ----------
    covariates: pd.DataFrame
        A dataframe of individual-level covariates.
    categories: List
        A list of treatment arm indices.
    add_constant: bool
        Should an intercept term be added?
    arm: None | Any
        A provided value to append as the "arm" column.
    """
    if arm is not None:
        covariates["arm"] = arm
    covariates["arm"] = pd.Categorical(covariates["arm"], categories=categories)
    covariates = pd.concat(
        [
            covariates,
            pd.get_dummies(covariates["arm"], prefix="arm", drop_first=True, dtype=int),
        ],
        axis=1,
    ).drop("arm", axis=1)
    if add_constant:
        covariates["const"] = 1.0
    return covariates


def weighted_probs(
    bandit_probs: Dict[int, float], weights: Dict[int, float]
) -> Dict[int, float]:
    """
    Re-weight bandit probabilities based on importance weights.

    Parameters
    ----------
    bandit_probs : Dict[int, float]
        Original probabilities (must sum to 1).
    weights : Dict[int, float]
        Importance weights (values in [0, 1]).

    Returns
    -------
    Dict[int, float]
        Updated bandit probabilities.

    Notes
    -----
    This function adjusts a set of bandit probabilities using importance weights
    while ensuring that the total probability remains 1. It follows these steps:

        (1) Apply Importance Weights:
            Multiply each arm's bandit probability by its corresponding weight
            to get the initial adjusted probabilities.

            >>> updated probabilities = bandit_probs * weights

        (2) Calculate Lost Probability Mass:
            Compute the probability mass lost due to down-weighting by multiplying
            each probability by (1 - its weight). Sum the total lost probability
            mass across all arms.

            >>> lost mass = bandit_probs * (1 - weights)
            >>> total mass lost = sum(lost mass)

        (3) Redistribute Lost Probability Mass:
            Compute the relative weight of each arm by dividing its weight by the
            total sum of all weights. Redistribute the lost probability mass to
            each arm proportionally to its relative weight.

            >>> relative weights = weights / sum(weights)
            >>> redistributed mass = (total mass lost) * (relative weights)

        (4) Compute Final Probabilities:
            Add the redistributed loss to each arm’s updated probability.

            >>> re-weighted probabilities = (updated probabilities) + (redistributed mass)

    The function returns a dictionary where each key is a bandit arm
    index and the value is its updated probability.
    """
    assert np.isclose(
        sum(bandit_probs.values()), 1.0, rtol=0.01
    ), "Bandit probabilities should sum to 1"
    updated_probs = {arm: weights[arm] * bandit_probs[arm] for arm in bandit_probs}
    losses = {arm: bandit_probs[arm] * (1 - weights[arm]) for arm in bandit_probs}
    total_loss = sum(losses.values())
    weight_sum = sum(weights.values())
    relative_weights = {arm: (weights[arm] / weight_sum) for arm in weights}
    redistributed_loss = {arm: total_loss * relative_weights[arm] for arm in weights}
    updated_bandit_probs = {
        arm: updated_probs[arm] + redistributed_loss[arm] for arm in bandit_probs
    }
    assert np.isclose(
        sum(updated_bandit_probs.values()), 1.0, rtol=0.01
    ), "Updated bandit probabilities should sum to 1"
    return updated_bandit_probs
