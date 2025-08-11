from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, is_regression=False):
        self.is_regression = is_regression

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["ROAD_COMPLETION_RATE"] = X["NO_OF_ROAD_WORKS_COMPLETED"] / (X["NO_OF_ROAD_WORK_SANCTIONED"] + 1e-6)
        X["BRIDGE_COMPLETION_RATE"] = X["NO_OF_BRIDGES_COMPLETED"] / (X["NO_OF_BRIDGES_SANCTIONED"] + 1e-6)
        
        if not self.is_regression:
            X["COST_PER_KM"] = X["COST_OF_WORKS_SANCTIONED"] / (X["LENGTH_OF_ROAD_WORK_SANCTIONED"] + 1e-6)
            X["EXPENDITURE_RATIO"] = X["EXPENDITURE_OCCURED"] / (X["COST_OF_WORKS_SANCTIONED"] + 1e-6)
        else:
            X["COST_PER_KM"] = 0
            X["EXPENDITURE_RATIO"] = 0
            
        return X
