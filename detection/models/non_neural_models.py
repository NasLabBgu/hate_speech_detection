from detection.models.base_model import BaseModel
from sklearn.linear_model import LogisticRegressionCV
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class MyLogisticRegression(BaseModel):
    def __init__(self, **kwargs):
        kwargs["name"] = "LogisticRegression"
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.model = LogisticRegressionCV()
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

class MyCatboost(BaseModel):
    def __init__(self, **kwargs):
        kwargs["name"] = "CatBoost"
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.model = CatBoostClassifier()
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, verbose=False)
    def predict(self, X_test):
        return self.model.predict(X_test)
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

class MyLightGBM(BaseModel):
    def __init__(self, **kwargs):
        kwargs["name"] = "LightGBM"
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.model = LGBMClassifier()
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

class MyXGBoost(BaseModel):
    def __init__(self, **kwargs):
        kwargs["name"] = "XGBoost"
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.model = XGBClassifier()
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)