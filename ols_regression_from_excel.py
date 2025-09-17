"""Compute OLS trends per Ticker based on the exported Excel workbook."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

EXCEL_PATH = Path("./data_output/yahoo_prices.xlsx")
OUTPUT_PATH = EXCEL_PATH.with_name(f"{EXCEL_PATH.stem}_with_ols.xlsx")
COMBINED_SHEET = "combined"
OLS_SHEET = "ols_regression"


def _run_ols_regression(df: pd.DataFrame, value_col: str = "close") -> pd.DataFrame:
    """Compute a simple OLS trend (value ~ time) per ticker."""
    results = []

    for ticker, group in df.groupby("ticker"):
        grp = group.dropna(subset=["date", value_col]).sort_values("date")
        if len(grp) < 2:
            continue  # need at least two points for a regression

        # Convert timestamps to seconds relative to first observation for numerical stability
        time_seconds = (grp["date"] - grp["date"].min()).dt.total_seconds().to_numpy()
        values = grp[value_col].to_numpy(dtype=float)

        X = np.column_stack([np.ones_like(time_seconds), time_seconds])
        beta, *_ = np.linalg.lstsq(X, values, rcond=None)
        intercept, slope_per_second = beta
        fitted = X @ beta

        ss_tot = np.sum((values - values.mean()) ** 2)
        ss_res = np.sum((values - fitted) ** 2)
        r_squared = float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot

        results.append(
            {
                "ticker": ticker,
                "n_obs": len(grp),
                "intercept": intercept,
                "slope_per_sec": slope_per_second,
                "slope_per_day": slope_per_second * 86_400,
                "r_squared": r_squared,
            }
        )

    return pd.DataFrame(results)


def _load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Excel-Datei nicht gefunden: {path}")
    # Load every sheet, we will re-export all sheets later to keep the workbook intact.
    return pd.read_excel(path, sheet_name=None)


def _ensure_datetime(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    if column in df:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def main(path: Path = EXCEL_PATH) -> None:
    try:
        sheets = _load_workbook(path)
    except FileNotFoundError as exc:
        print(f"⚠️ {exc}")
        return

    if COMBINED_SHEET not in sheets:
        print(f"⚠️ Arbeitsblatt '{COMBINED_SHEET}' nicht gefunden in {path}")
        return

    combined = _ensure_datetime(sheets[COMBINED_SHEET].copy())
    if combined.empty:
        print("⚠️ Keine Daten im kombinierten Sheet gefunden.")
        return

    ols_df = _run_ols_regression(combined)
    if ols_df.empty:
        print("⚠️ Konnte keine Regression berechnen (zu wenige Daten?).")
    else:
        print("OLS-Regression (Trend pro Ticker):")
        print(ols_df.to_string(index=False, float_format=lambda v: f"{v:0.6g}"))

    with pd.ExcelWriter(OUTPUT_PATH, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        if not ols_df.empty:
            ols_df.to_excel(writer, index=False, sheet_name=OLS_SHEET[:31])

    print(f"✅ Regressionsergebnisse gespeichert nach: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
