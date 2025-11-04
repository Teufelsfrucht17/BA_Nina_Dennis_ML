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
    chg = out[price_col].pct_change(periods=1)
    out["change"] = chg
    # out["change"] = (chg > 0).astype(int)  # Hier der Fix für change %

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
    # Nach dem Mergen sicherheitshalber nach Datum sortieren
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    # Zusätzliche abgeleitete Features
    df["change_lag1"] = df["change"].shift(1)
    df["change_lag5"] = df["change"].shift(5)
    df["change_roll_mean5"] = df["change"].rolling(window=5).mean()
    df["change_roll_std5"] = df["change"].rolling(window=5).std()

    df["momentum_lag1"] = df["momentum"].shift(1)

    for col in ("change_DAX", "change_VDAX"):
        base = col.lower() if col in df.columns else col
        if col not in df.columns and base not in df.columns:
            continue
        series_name = col if col in df.columns else base
        df[f"{series_name.lower()}_lag1"] = df[series_name].shift(1)
        df[f"{series_name.lower()}_roll_mean5"] = df[series_name].rolling(window=5).mean()

    df = df.dropna(how="any").reset_index(drop=True)
    return df


def splitdataXY(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teilt df in Featurematrix X und Ziel Y.

    'change' bleibt als Target, alle übrigen Spalten (außer Datum) bilden X.
    Groß-/Kleinschreibung bei DAX/VDAX wird harmonisiert.
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

    Y = df[["change"]].copy()

    # Alle Features außer Ziel und Datum verwenden
    drop_cols = {"change", "Date", "date"}
    X_cols = [c for c in df.columns if c not in drop_cols]
    X = df[X_cols].copy()
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
