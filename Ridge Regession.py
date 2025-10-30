import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from Dataprep2 import runningcycle, splitdataXY
import GloablVariableStorage


def ridge_classification(sheet_index: int) -> None:
    """RidgeClassifier mit y=change (0/1) und X=[momentum, change_dax, change_vdax]."""
    df = runningcycle(sheet_index)
    if df is None or df.empty:
        print(f"Sheet {sheet_index}: ⚠️ Keine Daten vorhanden nach runningcycle.")
        return

    X_change, Y_feats = splitdataXY(df)

    # Ziel/Features numerisch und ohne NaN
    y = pd.to_numeric(X_change.squeeze(), errors="coerce").astype("Int64")
    X = Y_feats.apply(pd.to_numeric, errors="coerce")

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True)
    if len(X) == 0 or y.nunique() < 2:
        print(f"Sheet {sheet_index}: ⚠️ Zu wenige Daten oder nur eine Klasse im Ziel.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifier())
    ])
    param_grid = {"clf__alpha": [0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    print(f"Sheet {sheet_index}: RidgeClassifier acc train = {acc_train:.4f}")
    print(f"Sheet {sheet_index}: RidgeClassifier acc test  = {acc_test:.4f}")
    print(f"Sheet {sheet_index}: Best alpha = {grid.best_params_['clf__alpha']}")


if __name__ == "__main__":
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            ridge_classification(i)
    except Exception as e:
        print(f"Ridge run failed: {e}")
