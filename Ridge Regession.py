import pandas as pd
from pandas.core.common import random_state
from rich.diagnose import report
from sklearn.linear_model import RidgeClassifier,Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import GloablVariableStorage
import Dataprep2


def ridge_classification(sheet_index: int,report:pd.DataFrame) -> pd.DataFrame:

    X, Y = Dataprep2.finalrunner(sheet_index)


    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42,
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    param_grid = {
        "model__alpha": [
            1e-5, 1e-4, 1e-3, 1e-2,
            3e-2, 1e-1, 3e-1,
            1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0
        ],
        "model__fit_intercept": [True, False],
        "model__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"],
        "model__tol": [1e-4, 1e-3],
        "model__max_iter": [5000],
    }
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="r2",n_jobs=-1)
    grid.fit(X_train, y_train)

    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    acc_train = r2_score(y_train, y_train_pred)
    acc_test =  r2_score(y_test, y_test_pred)

    print(f"Sheet {sheet_index}: RidgeClassifier acc train = {acc_train:.4f}")
    print(f"Sheet {sheet_index}: RidgeClassifier acc test  = {acc_test:.4f}")
    print(f"Sheet {sheet_index}: Best alpha = {grid.best_params_['model__alpha']}")

    return pd.DataFrame.empty


def runRidgeRegession():
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
          report = ridge_classification(i)
    except Exception as e:
        print(f"Ridge run failed: {e}")

runRidgeRegession()
