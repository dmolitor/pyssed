from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Any


class Model(ABC):
    """
    A model class to support a variety of different models. This should accept
    an input matrix or dataframe `X` and response array `y`. This class must also
    implement the following methods:

    Notes
    -----
    The following methods must be implemented:

    fit()
        This method should fit the model and return the fitted model
        object. The class should also record the fitted model under the
        `self._fitted` attribute. That way the user can access the fitted
        model directly or after the fact.
    predict(X):
        This method should generate predicted values from the fitted
        model on a holdout set and return these predictions as a numpy array.
    """

    @abstractmethod
    def fit(self) -> Any:
        """
        Fit the underlying model and return the model object.

        This method should also store the model at the `self._fitted` attribute
        for direct user access.

        Returns
        -------
        Any
            An arbitrary fitted model.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame | npt.NDArray | Any) -> npt.NDArray[np.float64]:
        """
        Returns the predictions of the fitted model on a hold-out dataset.

        This function should return predictions as a numpy array.

        Parameters
        ----------
        X : pd.DataFrame | np.array | Any
            Any rectangular data structure that can generate predictions
            using the `np.dot(X, beta)` procedure.

        Returns
        -------
        np.array
            A one-dimension numpy array with predicted hold-out values.
        """


class FastOLSModel(Model):
    """
    An example Model class that fits a trimmed down OLS model.
    """

    def __init__(self, X: pd.DataFrame | npt.NDArray | Any, y: npt.NDArray):
        self._fitted = None
        self._X = X
        self._y = y

    def fit(self):
        self._fitted, _, _, _ = np.linalg.lstsq(self._X, self._y)
        return self

    def predict(self, X: pd.DataFrame | npt.NDArray):
        assert self._fitted is not None, "Attempting to predict before fitting a model"
        predictions = np.dot(X, self._fitted)
        return predictions
