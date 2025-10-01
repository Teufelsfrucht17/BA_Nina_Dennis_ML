import numpy as np
import pandas as pd
import GloablVariableStorage
from DataPreperation import featuresplit, combine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score




def OLSRegression(ticker):
    X, Y = combine(ticker)

    # Drop non-numeric columns, keep only numeric
    if hasattr(X, 'columns'):
        X = X.drop(columns=[c for c in X.columns if c.lower() in ("ticker","symbol")], errors='ignore')
        date_col = X["date"].copy() if "date" in X.columns else None
        X = X.select_dtypes(include=[np.number]).copy()
    else:
        date_col = None

    Y = pd.to_numeric(Y.squeeze(), errors='coerce')
    mask = X.notna().all(axis=1) & Y.notna()
    X, Y = X.loc[mask].reset_index(drop=True), Y.loc[mask].reset_index(drop=True)
    if date_col is not None:
        date_col = pd.to_datetime(date_col.loc[mask]).reset_index(drop=True)

    if date_col is not None and len(date_col) == len(X):
        order = np.argsort(date_col.values)
        X, Y = X.iloc[order].reset_index(drop=True), Y.iloc[order].reset_index(drop=True)

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
    Y_train, Y_test = Y.iloc[:split].copy(), Y.iloc[split:].copy()

    # Ridge regression pipeline with scaling
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])
    param_grid = {"ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10,20,100,1000,10000,100000]}
    CV_ridge = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)
    CV_ridge.fit(X_train, Y_train)

    # Predictions
    y_train_pred = CV_ridge.predict(X_train)
    y_test_pred = CV_ridge.predict(X_test)

    r2_train = r2_score(Y_train, y_train_pred)
    r2_test = r2_score(Y_test, y_test_pred)

    print("Ridge R2 train =", r2_train)
    print("Ridge R2 test  =", r2_test)
    print("Best alpha:", CV_ridge.best_params_["ridge__alpha"])




OLSRegression(GloablVariableStorage.ListofStock)