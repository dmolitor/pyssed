from pyssed.bandit import Bandit
import numpy as np
from tqdm import tqdm
from typing import Callable
import pandas as pd
import plotnine as pn
from pyssed.utils import check_shrinkage_rate, cs_radius, ite, last, var

generator = np.random.default_rng(seed=123)


class MAD:
    """
    A class implementing Liang and Bojinov's Mixture-Adaptive Design (MAD).

    Parameters
    ----------
    bandit: pyssed.Bandit
        The underlying bandit algorithm on which the MAD design operates.
        This bandit class must implement several crucial methods/attributes.
        For more details on how to create a custom Bandit object, see the
        documentation of the `pyssed.Bandit` class.
    alpha : float
        The size of the statistical test (testing for non-zero ATEs).
    delta : Callable[[int], float]
        A function that generates the real-valued sequence delta_t in Liang
        and Bojinov (Definition 4 - Mixture Adaptive Design). This sequence
        determines the amount of random exploration that is infused into
        the bandit design, and it should converge to 0 slower than 1/t^(1/4)
        where t denotes the time step in {0, ... n}. This function should
        intake an integer (t) and output a float (the corresponding delta_t).
    t_star : int
        The time-step at which we want to optimize the CSs to be tightest.
        E.g. Liang and Bojinov set this to the max horizon of their experiment.
    """

    def __init__(
        self, bandit: Bandit, alpha: float, delta: Callable[[int], float], t_star: int
    ):
        self._alpha = alpha
        self._ate = []
        self._bandit = bandit
        self._cs_radius = []
        self._cs_width = []
        self._delta = delta
        self._ite = []
        self._ite_var = []
        self._n = []
        self._t_star = t_star
        for _ in range(bandit.k()):
            self._ite.append([])
            self._ite_var.append([])
            self._ate.append([])
            self._cs_radius.append([])
            self._cs_width.append(0)
            self._n.append([0])

    def estimates(self) -> pd.DataFrame:
        """
        Extract estimated ATEs and confidence sequences.

        Returns
        -------
        pandas.DataFrame
            A dataframe of ATE estimates and corresponding CS lower and
            upper bounds.
        """
        results = {"arm": [], "ate": [], "lb": [], "ub": []}
        for arm in range(len(self._ate)):
            if arm == self._bandit.control():
                continue
            ate = last(self._ate[arm])
            radius = last(self._cs_radius[arm])
            lb = ate - radius
            ub = ate + radius
            results["arm"].append(arm)
            results["ate"].append(ate)
            results["lb"].append(lb)
            results["ub"].append(ub)
        return pd.DataFrame(results)

    def fit(self, verbose: bool = True) -> None:
        """
        Fit the MAD algorithm for the full time horizon.

        Parameters
        ----------
        verbose : bool
            Whether to print progress of the algorithm

        Returns
        -------
        None
        """
        if verbose:
            fit_seq = tqdm(range(self._t_star), total=self._t_star)
        else:
            fit_seq = range(self._t_star)
        for _ in fit_seq:
            self.pull()

    def plot_ate(self) -> pn.ggplot:
        """
        Plot the ATE and CS paths for each arm of the experiment.

        Returns
        -------
        plotnine.ggplot
        """
        arms = list(range(len(self._ate)))
        arms.remove(self._bandit.control())
        estimates = []
        for arm in arms:
            ates = self._ate[arm]
            radii = self._cs_radius[arm]
            ubs = np.nan_to_num([x + y for (x, y) in zip(ates, radii)], nan=np.inf)
            lbs = np.nan_to_num([x - y for (x, y) in zip(ates, radii)], nan=-np.inf)
            estimates_df = pd.DataFrame(
                {
                    "arm": [arm] * len(ates),
                    "ate": ates,
                    "lb": lbs,
                    "ub": ubs,
                    "t": range(1, len(ates) + 1),
                }
            )
            estimates.append(estimates_df)
        estimates = (
            pd.concat(estimates, axis=0).reset_index(drop=True).dropna(subset="ate")
        )
        plt = (
            pn.ggplot(
                data=estimates,
                mapping=pn.aes(
                    x="t",
                    y="ate",
                    ymin="lb",
                    ymax="ub",
                    color="factor(arm)",
                    fill="factor(arm)",
                ),
            )
            + pn.geom_line(size=0.3, alpha=0.8)
            + pn.geom_ribbon(alpha=0.05)
            + pn.facet_wrap("~ arm", labeller=pn.labeller(arm=lambda v: f"Arm {v}"))
            + pn.theme_538()
            + pn.theme(legend_position="none")
            + pn.labs(y="ATE", color="Arm", fill="Arm")
        )
        return plt

    def plot_n(self) -> pn.ggplot:
        """
        Plot the total N assigned to each arm.

        Returns
        -------
        plotnine.ggplot
        """
        arm_n = pd.concat(
            [
                pd.DataFrame({"arm": [arm], "n": last(self._n[arm])})
                for arm in range(len(self._ate))
            ]
        )
        plt = (
            pn.ggplot(data=arm_n, mapping=pn.aes(x="factor(arm)", y="n"))
            + pn.geom_bar(stat="identity")
            + pn.labs(x="Arm", y="N")
            + pn.theme_538()
            + pn.theme(legend_position="none")
        )
        return plt

    def plot_sample_assignment(self) -> pn.ggplot:
        """
        Plot sample assignment to arms across time

        Returns
        -------
        plotnine.ggplot
        """
        sample_assignment = pd.concat(
            [
                pd.DataFrame(
                    {
                        "arm": [arm] * len(self._n[arm]),
                        "t": np.array(range(len(self._n[arm]))),
                        "n": self._n[arm],
                    }
                )
                for arm in range(len(self._ate))
            ]
        )
        plt = (
            pn.ggplot(
                data=sample_assignment,
                mapping=pn.aes(x="t", y="n", color="factor(arm)", group="factor(arm)"),
            )
            + pn.geom_line()
            + pn.facet_wrap(
                "~ arm", ncol=2, labeller=pn.labeller(arm=lambda v: f"Arm {v}")
            )
            + pn.theme_538()
            + pn.theme(legend_position="none")
            + pn.labs(y="N", color="Arm", fill="Arm")
        )
        return plt

    def pull(self) -> None:
        """
        Perform one full iteration of the MAD algorithm.

        Returns
        -------
        None
        """
        # Bandit parameters
        control = self._bandit.control()
        k = self._bandit.k()
        t = self._bandit.t()
        d_t = self._delta(t)
        check_shrinkage_rate(t, d_t)
        # Use the MAD algorithm to select the treatment arm
        arm_probs = self._bandit.probabilities()
        probs = [d_t / k + (1 - d_t) * p for p in arm_probs.values()]
        selected_index = generator.multinomial(1, pvals=probs).argmax()
        selected_arm = list(arm_probs.keys())[selected_index]
        # Update the counts of selected arms
        for arm in range(len(self._ate)):
            if arm == selected_arm:
                self._n[arm].append((last(self._n[arm]) + 1))
            else:
                self._n[arm].append((last(self._n[arm])))
        propensity = probs[selected_index]
        reward = self._bandit.reward(selected_arm)
        # Calculate the individual treatment effect estimate and variance
        treat_effect = ite(reward, int(selected_arm != control), propensity)
        treat_effect_var = var(reward, propensity)
        self._ite[selected_arm].append(treat_effect)
        self._ite_var[selected_arm].append(treat_effect_var)
        # For each arm calculate the ATE and corresponding Confidence Sequence (CS)
        for arm in arm_probs.keys():
            if arm == control:
                ites = self._ite[control]
                vars = self._ite_var[control]
            else:
                # Get the ITE and variance estimates for the current arm and control arm
                ites = self._ite[control] + self._ite[arm]
                vars = self._ite_var[control] + self._ite_var[arm]
            assert len(ites) == len(
                vars
            ), "Mismatch in dimensions of ITEs and Variances"
            if len(self._ite[arm]) < 1 or len(self._ite[control]) < 1:
                avg_treat_effect = np.nan
                conf_seq_radius = np.inf
            else:
                # Calculating np.mean(ites) effectively ignores any time periods (i)
                # where neither the current arm nor the control arm were selected.
                # This is WRONG! The unbiased ATE estimator includes the ITE for each
                # of those steps (in which case the ITE is just 0). This is why the
                # denominator is the full number of time steps, t.
                avg_treat_effect = np.sum(ites).astype(float) / t
                # The Confidence Sequence calculation is similar. Like the ATE estimator,
                # the estimated variance for any (i) where neither the current arm
                # nor the control arm were selected is just 0. The CS calculation simply
                # sums the estimated variances which is why we only need the non-zero
                # variance estimates.
                conf_seq_radius = cs_radius(vars, t, self._t_star, self._alpha)
            self._ate[arm].append(avg_treat_effect)
            self._cs_radius[arm].append(conf_seq_radius)
            self._cs_width[arm] = 2.0 * conf_seq_radius

    def summary(self) -> None:
        """
        Print a summary of ATEs and confidence bands.

        Returns
        -------
        None
        """
        print("Treatment effect estimates:")
        for arm in range(len(self._ate)):
            if arm == self._bandit.control():
                continue
            ate = last(self._ate[arm])
            radius = last(self._cs_radius[arm])
            lb = ate - radius
            ub = ate + radius
            print(f"- Arm {arm}: {round(ate, 3)} ({round(lb, 5)}, {round(ub, 5)})")
