from __future__ import annotations

import pandas as pd


def Excelloader(
    path: str,
    sheet: int | str = 0,
) -> pd.DataFrame:
    """Sehr simpler Excel-Loader: ein Sheet -> DataFrame.

    - `path`: Pfad zur Excel-Datei (.xlsx, .xls, ...)
    - `sheet`: Sheet-Index oder -Name (Default: 0 = erstes Sheet)
    - Rückgabe: DataFrame genau so, wie `pd.read_excel` es liefert
    """
    return pd.read_excel(path, sheet_name=sheet)


def price_plus_minus(df: pd.DataFrame, price_col: str | None = None, out_col: str = "X") -> pd.DataFrame:
    """Erzeugt ein 0/1-Signal aus der Price-Spalte: 1 bei Steigerung, 0 bei Rückgang/gleich.

    - df: DataFrame aus dem Excel-Sheet
    - price_col: Name der Preis-Spalte; wenn None, wird eine übliche Spalte gesucht
    - out_col: Name der Ausgabespalte (Standard: "X")
    Rückgabe: df mit zusätzlicher Spalte `out_col`
    """
    if price_col is None:
        candidates = [
            "Price", "Adj Close", "Close", "CLOSE_PRC", "OPEN_PRC", "PX_LAST", "close",
        ]
        price_col = next((c for c in candidates if c in df.columns), None)
        if price_col is None:
            # Nimm erste numerische Spalte als Preis
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                raise ValueError("Keine Preis-Spalte gefunden und keine numerische Spalte verfügbar.")
            price_col = num_cols[0]

    s = pd.to_numeric(df[price_col], errors="coerce")
    # 1 für positive Veränderung ggü. Vortag/-zeile, sonst 0; erste Zeile wird 0
    x = (s.diff() > 0).fillna(False).astype(int)
    out = df.copy()
    out[out_col] = x
    return out


print(Excelloader("DataStorage/Portfolio.xlsx",sheet=0))
