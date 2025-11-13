from pathlib import Path
import pandas as pd

import GloablVariableStorage

# Very simple defaults: pick the first existing workbook
_candidates = [
    Path("DataStorage/Mag7.xlsx"),
    Path("DataStorage/Dow30.xlsx"),
]
DEFAULT_EXCEL = next((p for p in _candidates if p.exists()), _candidates[0])


def dataReader(ticker: str, exclesheet: str | Path = DEFAULT_EXCEL) -> pd.DataFrame:
    """Read a single ticker sheet from an Excel file and return a clean DataFrame.

    Keeps only the essential columns [date, adj_close, volume] if present and sorts by date.
    """
    df = pd.read_excel(exclesheet, sheet_name=ticker)
    # Normalize common column names
    df = df.rename(columns={
        "Datetime": "date",
        "Date": "date",
        "Adj Close": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
    })
    if "adj_close" not in df.columns:
        for alt in ["Close", "close"]:
            if alt in df.columns:
                df["adj_close"] = df[alt]
                break
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    cols = []
    if "date" in df.columns:
        cols.append("date")
    for c in ["adj_close", "volume"]:
        if c in df.columns:
            cols.append(c)
    if not cols:
        # nothing recognizable; return the raw frame to avoid KeyErrors
        return df.reset_index(drop=True)

    df = df[cols]
    if "date" in df.columns:
        df = df.dropna(subset=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    return df


def featureEnegnier(ticker: str, exclesheet: str | Path = DEFAULT_EXCEL) -> pd.DataFrame:
    """Minimal feature set for a ticker: returns DataFrame with features and date.

    Features:
    - ret_1: 1-step return on adj_close
    - sma_gap: price relative to 20-period SMA
    - turnover: volume relative to 20-period average volume
    """
    d = dataReader(ticker, exclesheet)
    d = d.copy()
    d["ret_1"] = d["adj_close"].pct_change(1)
    d["sma_20"] = d["adj_close"].rolling(20).mean()
    d["sma_gap"] = d["adj_close"] / d["sma_20"] - 1
    if "volume" in d.columns:
        d["vol_mean_20"] = d["volume"].rolling(20).mean()
        d["turnover"] = d["volume"] / (d["vol_mean_20"] + 1e-9)
    return d


def featuresplit(
    ticker: str,
    exclesheet: str | Path = DEFAULT_EXCEL,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return X, Y for a single ticker using minimal features.

    Y = future percentage change over 'horizon' periods of adj_close.
    """
    df = featureEnegnier(ticker, exclesheet)
    df["Y"] = df["adj_close"].pct_change(horizon).shift(-horizon)
    features = [c for c in ["ret_1", "sma_gap", "turnover"] if c in df.columns]
    out = df.dropna(subset=features + ["Y"])  # simple dropna instead of imputation
    dates = out["date"].reset_index(drop=True) if "date" in out.columns else pd.Series(dtype="datetime64[ns]")
    X = out[features].reset_index(drop=True)
    Y = out["Y"].reset_index(drop=True)
    return X, Y, dates


def combine(
    tickers: list[str],
    exclesheet: str | Path = DEFAULT_EXCEL,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Concatenate X, Y across multiple tickers. Adds one-hot ticker dummies."""
    X_list: list[pd.DataFrame] = []
    Y_list: list[pd.Series] = []
    date_list: list[pd.Series] = []

    for t in tickers:
        X_t, Y_t, dates_t = featuresplit(t, exclesheet, horizon)
        X_t = X_t.copy()
        X_t["ticker"] = t
        X_list.append(X_t)
        Y_list.append(Y_t.rename("Y"))
        if not dates_t.empty:
            date_list.append(dates_t.rename("date"))

    if not X_list:
        return pd.DataFrame(), pd.Series(dtype=float)

    X_all = pd.concat(X_list, axis=0, ignore_index=True)
    Y_all = pd.concat(Y_list, axis=0, ignore_index=True)
    if date_list:
        dates_all = pd.concat(date_list, axis=0, ignore_index=True)
        dates_all = pd.to_datetime(dates_all, errors="coerce")
        if dates_all.notna().any():
            order = dates_all.sort_values(kind="mergesort").index
            X_all = X_all.iloc[order].reset_index(drop=True)
            Y_all = Y_all.iloc[order].reset_index(drop=True)

    # One-hot encode ticker (kept minimal)
    if "ticker" in X_all.columns:
        dummies = pd.get_dummies(X_all.pop("ticker"), prefix="ticker")
        X_all = pd.concat([X_all, dummies], axis=1)

    return X_all, Y_all



combine(GloablVariableStorage.LisofStocks_Dow, DEFAULT_EXCEL, horizon=5)
