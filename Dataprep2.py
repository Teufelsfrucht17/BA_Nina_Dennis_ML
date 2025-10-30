from __future__ import annotations

import numpy as np
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
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(how="any")

    return df


def price_plus_minus(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeuge 0/1-Signal je Zeile basierend auf der Preisänderung zur Vorzeile.

    - Positiver Change -> 1
    - Negativer oder unveränderter Change -> 0
    - Nutzt 'Price' (falls vorhanden), sonst erste numerische Spalte.
    - Sortiert nach 'Date', falls vorhanden, bevor die Änderung berechnet wird.
    """
    out = df.copy()

    # Nach Datum sortieren, falls vorhanden
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date").reset_index(drop=True)

    # Preisspalte identifizieren
    if "Price" in out.columns:
        price_col = "Price"
    else:
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        price_col = num_cols[0] if num_cols else None

    if price_col is None:
        raise ValueError("Keine numerische 'Price'-Spalte gefunden.")

    # Numerisch sicherstellen
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    # Änderung zur Vorzeile und 0/1-Signal
    chg = out[price_col].diff()
    out["change"] = (chg > 0).astype(int)

    # Debug: erste Zeilen ausgeben
   # print(out.head())
    return out

def createmomentum(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Fügt eine Momentum-Spalte hinzu (periodenbasiert, nur int-Window).

    Definition:
    momentum_t = Price_t / Price_{t-window} - 1

    Parameter:
    - window: Anzahl Perioden (z. B. 60)
    Voraussetzungen:
    - DataFrame enthält eine Spalte 'Price'.
    """
    out = df.copy()

    # Nach Datum sortieren, falls vorhanden
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date").reset_index(drop=True)

    # Price numerisch sicherstellen
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    # Periodenbasiertes Momentum
    out["momentum"] = out["Price"].pct_change(periods=window)
    out = out.dropna(how="any")
   # print(out.head())
    return out


def runningcycle(sheetstock:int) -> pd.DataFrame:
    try:
        df = Excelloader("DataStorage/Portfolio.xlsx", sheetstock)
        df = price_plus_minus(df)
        df = createmomentum(df, 30)
        df = df.drop(columns=["Price"])
        df1 = Excelloader("DataStorage/INDEX.xlsx", 1)
        df1 = price_plus_minus(df1)
        df1 = df1[["Date", "change"]].copy()
        df1 = df1.rename(columns={"Date": "Date", "change": "change_DAX"})
        df = df.merge(df1, how="inner", on="Date")
        df1 = Excelloader("DataStorage/INDEX.xlsx", 0)
        df1 = price_plus_minus(df1)
        df1 = df1[["Date", "change"]].copy()
        df1 = df1.rename(columns={"Date": "Date", "change": "change_VDAX"})
        df = df.merge(df1, how="inner", on="Date")
      #  print(df.head())
    except Exception as e:
        print(f"Fehler beim Verarbeiten: {e}")
    return df


def splitdataXY(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teilt df in X und Y auf.

    Erwartet die Spalten: 'change', 'momentum', 'change_dax', 'change_vdax'.
    Akzeptiert auch 'change_DAX'/'change_VDAX' und mappt sie auf lowercase.
    """
    df = df.copy()
    # Toleranz für Groß-/Kleinschreibung bei DAX/VDAX
    df.rename(columns={
        "change_DAX": "change_dax",
        "change_VDAX": "change_vdax",
    }, inplace=True)

    required = ["change", "momentum", "change_dax", "change_vdax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}. Vorhanden: {list(df.columns)}")

    X = df[["change"]].copy()
    Y = df[["momentum", "change_dax", "change_vdax"]].copy()
    return X, Y


def finalrunner(sheet:int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
     testvar = runningcycle(sheet)
     X, Y = splitdataXY(testvar)
    # print(testvar.head())
     return X, Y
    except Exception as e:
        print(f"Fehler beim Verarbeiten: {e}")




finalrunner(3)