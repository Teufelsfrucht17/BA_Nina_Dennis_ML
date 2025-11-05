from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import GloablVariableStorage
from Dataprep2 import finalrunner


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
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
    model_out: Path,
):
    X, y, feature_names = load_xy(sheet)
    X_train, X_val, y_train, y_val = time_series_split(X, y, val_split)

    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )
    reg.fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)
    y_val_pred = reg.predict(X_val)

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
            "model": reg,
            "feature_names": feature_names,
            "sheet": sheet,
            "metrics": metrics,
            "val_split": val_split,
        },
        model_out,
    )
    print(f"Gespeichert: {model_out}")
    return metrics


def main(sheet: int | None ):
    parser = argparse.ArgumentParser(description="RandomForest-Regressor auf Dataprep2-Daten trainen")
    parser.add_argument("--sheet", type=int, default=sheet, help="Sheet-Index (Default: 3)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Anteil Validierung (Default: 0.2)")
    parser.add_argument("--n_estimators", type=int, default=500, help="Anzahl Bäume (Default: 500)")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximale Tiefe (Default: None)")
    parser.add_argument("--min_samples_leaf", type=int, default=5, help="Min. Samples pro Blatt (Default: 5)")
    parser.add_argument("--random_state", type=int, default=42, help="Random-State (Default: 42)")
    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path("data_output/random_forest.joblib"),
        help="Speicherpfad für Modell und Metadaten",
    )

    args = parser.parse_args()
    train_random_forest(
        sheet=args.sheet,
        val_split=args.val_split,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        model_out=args.model_out,
    )


if __name__ == "__main__":
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            main(i)
    except Exception as e:
        print(f"Ridge run failed: {e}")

