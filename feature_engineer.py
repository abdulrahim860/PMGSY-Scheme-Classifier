import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["ROAD_COMPLETION_RATE"] = X["NO_OF_ROAD_WORKS_COMPLETED"] / (X["NO_OF_ROAD_WORK_SANCTIONED"] + 1e-6)
        X["BRIDGE_COMPLETION_RATE"] = X["NO_OF_BRIDGES_COMPLETED"] / (X["NO_OF_BRIDGES_SANCTIONED"] + 1e-6)
        X["COST_PER_KM"] = X["COST_OF_WORKS_SANCTIONED"] / (X["LENGTH_OF_ROAD_WORK_SANCTIONED"] + 1e-6)
        X["EXPENDITURE_RATIO"] = X["EXPENDITURE_OCCURED"] / (X["COST_OF_WORKS_SANCTIONED"] + 1e-6)
        return X
