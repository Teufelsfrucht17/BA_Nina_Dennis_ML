from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

import GloablVariableStorage
from Dataprep2 import finalrunner
from createScoreModels import createscore


def load_xy(sheet: int):
    """Lädt Featurematrix und Ziel aus Dataprep2."""
    X_df, Y_df = finalrunner(sheet)
    if "change" not in Y_df.columns:
        raise ValueError("Target-Spalte 'change' fehlt in den gelieferten Daten.")

    X = X_df.to_numpy(dtype=np.float32)
    y = Y_df["change"].to_numpy(dtype=np.float32)
    return X, y, list(X_df.columns)


def time_series_split(X: np.ndarray, y: np.ndarray, val_split: float):
    """Chronologischer Split: frühe Daten -> Training, späte -> Validation."""
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("Keine Datenpunkte vorhanden.")

    val_size = int(n_samples * val_split)
    train_size = n_samples - val_size
    if train_size <= 0:
        raise ValueError("Trainingssplit ergibt keine Trainingsdaten. val_split anpassen.")

    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]
    return X_train, X_val, y_train, y_val


def train_random_forest(
    sheet: int,
    val_split: float,
    model_out: Path,
):
    param_grid = {
        'max_depth': [4, 5, 6, 7, 8],
     #   'n_estimators': [10, 50, 100, 150, 200],
        'criterion': ['squared_error', 'absolute_error'],
     #   'max_features': ['auto', 'sqrt', 'log2'],
      #  'min_samples_split': [2, 3, 4, 5],
       # 'min_samples_leaf': [2, 3, 4, 5],
      #  'bootstrap': [True, False],
    }


    X, y, feature_names = load_xy(sheet)
    X_train, X_val, y_train, y_val = time_series_split(X, y, val_split)

    reg = RandomForestRegressor(random_state=42)
    regGS = GridSearchCV(estimator=reg,param_grid=param_grid,cv=4,n_jobs=-1)
    regGS.fit(X_train, y_train)


    y_train_pred = regGS.predict(X_train)
    y_val_pred = regGS.predict(X_val)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred) if len(y_val) else float("nan")

    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "val_r2": r2_score(y_val, y_val_pred) if len(y_val) else float("nan"),
        "train_rmse": float(np.sqrt(train_mse)),
        "val_rmse": float(np.sqrt(val_mse)) if not np.isnan(val_mse) else float("nan"),
    }

    print(
        f"RandomForest - Sheet {sheet} | train_r2={metrics['train_r2']:.4f} | "
        f"val_r2={metrics['val_r2']:.4f} | train_rmse={metrics['train_rmse']:.6f} | "
        f"val_rmse={metrics['val_rmse']:.6f}"
    )

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": regGS,
            "feature_names": feature_names,
            "sheet": sheet,
            "metrics": metrics,
            "val_split": val_split,
        },
        model_out,
    )
    print(f"Gespeichert: {model_out}")
    return metrics


def RF(sheet: int | None, report: pd.DataFrame | None = None) -> pd.DataFrame:
    if report is None:
        report = createscore()

    parser = argparse.ArgumentParser(description="RandomForest-Regressor auf Dataprep2-Daten trainen")
    parser.add_argument("--sheet", type=int, default=sheet, help="Sheet-Index (Default: 3)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Anteil Validierung (Default: 0.2)")
    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path("data_output/ramdom_Forest/random_forest_"+str(sheet)+".joblib"),
        help="Speicherpfad für Modell und Metadaten",
    )

    args = parser.parse_args()
    metrics = train_random_forest(
        sheet=args.sheet,
        val_split=args.val_split,
        model_out=args.model_out,
    )

    report.loc[len(report)] = [
        "Random Forest",
        args.sheet,
        metrics["train_r2"],
        metrics["val_r2"],
        "",
        "N/A",
    ]
    return report


def Run_RandomForest() -> pd.DataFrame:
    report = createscore()
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            report = RF(i, report)
    except Exception as e:
        print(f"Ridge run failed: {e}")

    return report

Run_RandomForest()