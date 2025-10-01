from csv import excel
from pathlib import Path

import pandas as pd
import numpy as np


def load_benchmark() -> pd.DataFrame:
    """Lädt Benchmark (z. B. SPY) und berechnet Tagesrenditen."""
    path = Path("DataStorage/SPY.csv")
    if not path.exists():
        return pd.DataFrame({"date": [], "bench_ret_1d": []})

    benchmark = pd.read_csv(path, parse_dates=["date"])
    benchmark = benchmark[pd.to_numeric(benchmark["adj_close"], errors="coerce").notna()].copy()
    benchmark["adj_close"] = pd.to_numeric(benchmark["adj_close"], errors="coerce")
    if "volume" in benchmark.columns:
        benchmark["volume"] = pd.to_numeric(benchmark["volume"], errors="coerce")
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    benchmark["bench_ret_1d"] = benchmark["adj_close"].pct_change(1)
    return benchmark[["date", "bench_ret_1d"]]


def dataReader(ticker, exclesheet) -> pd.DataFrame:

    dataprep = pd.read_excel(exclesheet, sheet_name=ticker)

    print(dataprep.head())

    dataprep = dataprep.drop(columns=["close", "open", "high", "low", "ticker"], errors="ignore")

    dataprep = dataprep.sort_values("date").reset_index(drop=True)

    dataprep.to_csv(str("DataStorage/" + ticker + ".csv"), index=False)

    return dataprep


def featureEnegnier(ticker) -> pd.DataFrame:

    dataLabel = pd.DataFrame(dataReader(ticker, "DataStorage/testtageperiode.xlsx"))

    dataLabel["ret_1m"] = dataLabel["adj_close"].pct_change(1)
    dataLabel["ret_3m"] = dataLabel["adj_close"].pct_change(3)

    dataLabel["sma_20"] = dataLabel["adj_close"].rolling(20).mean()
    dataLabel["sma_gap"] = dataLabel["adj_close"] / dataLabel["sma_20"] - 1

    dataLabel["vol_mean_20"] = dataLabel["volume"].rolling(20).mean()
    dataLabel["turnover"] = dataLabel["volume"] / (dataLabel["vol_mean_20"] + 1e-9)

    TRADING_DAYS_PER_YEAR = 252
    lookbacks = {
        "mom_3m": int(TRADING_DAYS_PER_YEAR * 3 / 12),
        "mom_1y": TRADING_DAYS_PER_YEAR,
        "mom_5y": TRADING_DAYS_PER_YEAR * 5,
        "mom_10y": TRADING_DAYS_PER_YEAR * 10,
    }
    for name, lb in lookbacks.items():
        if lb > 0:
            dataLabel[name] = dataLabel["adj_close"].pct_change(lb)

    benchmark = load_benchmark()
    dataLabel = dataLabel.merge(benchmark, on="date", how="left")
    dataLabel["bench_ret_1d"] = dataLabel["bench_ret_1d"].fillna(0)

    rolling_cov = dataLabel["ret_1m"].rolling(60).cov(dataLabel["bench_ret_1d"])
    rolling_var = dataLabel["bench_ret_1d"].rolling(60).var()
    dataLabel["beta_60"] = rolling_cov / (rolling_var + 1e-9)
    dataLabel["alpha_60"] = dataLabel["ret_1m"] - dataLabel["beta_60"] * dataLabel["bench_ret_1d"]
    dataLabel["beta_60"] = dataLabel["beta_60"].fillna(0)
    dataLabel["alpha_60"] = dataLabel["alpha_60"].fillna(0)

    return dataLabel


def initilizedayliy(ticker):

    dataLabel = pd.DataFrame(dataReader(ticker, "DataStorage/mag7_1m_last8d.xlsx"))

    dataLabel["ret_1m"] = dataLabel["adj_close"].pct_change(1)
    dataLabel["ret_3m"] = dataLabel["adj_close"].pct_change(3)

    TRADING_DAYS_PER_YEAR = 252
    lookbacks = {
        "mom_3m": int(TRADING_DAYS_PER_YEAR * 3 / 12),  # ~63
        "mom_1y": TRADING_DAYS_PER_YEAR,  # ~252
        "mom_5y": TRADING_DAYS_PER_YEAR * 5,  # ~1260
        "mom_10y": TRADING_DAYS_PER_YEAR * 10,  # ~2520
    }
    for name, lb in lookbacks.items():
        if lb > 0:
            dataLabel[name] = dataLabel["adj_close"].pct_change(lb)
    return dataLabel


def addIntraday(ticker):
    """Kombiniert Daily-Langfrist-Signale mit Intraday-Features (1m)."""

    daily = featureEnegnier(ticker).copy()
    daily.rename(columns={"Datetime": "date", "Date": "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.tz_localize(None)
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    daily_keep = ["date", "mom_3m", "mom_1y", "mom_5y", "mom_10y"]
    daily_feat = daily[daily_keep].dropna().sort_values("date")

    intra = pd.read_excel("DataStorage/mag7_1m_last8d.xlsx", sheet_name=ticker).copy()
    intra.rename(columns={"Datetime": "date", "Date": "date"}, inplace=True)
    intra["date"] = pd.to_datetime(intra["date"], errors="coerce").dt.tz_localize(None)
    if "ticker" not in intra.columns:
        intra["ticker"] = ticker
    intra = intra.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    g = intra.groupby("ticker", group_keys=False)
    intra["ret_1m"] = g["adj_close"].pct_change(1)
    intra["ret_3m"] = g["adj_close"].pct_change(3)
    intra["sma_20"] = g["adj_close"].apply(lambda s: s.rolling(20).mean())
    intra["sma_gap"] = intra["adj_close"] / intra["sma_20"] - 1
    intra["vol_mean_20"] = g["volume"].apply(lambda s: s.rolling(20).mean())
    intra["turnover"] = intra["volume"] / (intra["vol_mean_20"] + 1e-9)

    merged = pd.merge_asof(
        intra.sort_values("date"),
        daily_feat.sort_values("date"),
        on="date",
        direction="backward"
    )

    return merged


def featuresplit(ticker):

    dataSplit = pd.DataFrame(featureEnegnier(ticker))

    dataSplit.to_excel("DataStorage/split.xlsx", index=False)

    dataSplit["Y"] = dataSplit["adj_close"].pct_change(5).shift(-5)

    features = [
        "ret_1m",
        "ret_3m",
        "sma_gap",
        "turnover",
        "mom_3m",
        "mom_1y",
        "mom_5y",
        "mom_10y",
        "bench_ret_1d",
        "beta_60",
        "alpha_60",
    ]

    dataSplit = dataSplit.dropna(subset=features + ["Y"]).reset_index(drop=True)

    X = dataSplit[features]
    Y = dataSplit["Y"]

    return X, Y


def combine(tickers):
    X_list, Y_list = [], []

    for t in tickers:
        Xcom, Ycom = featuresplit(t)   # Xcom: DataFrame, Ycom: Series
        Xcom = Xcom.copy()
        Xcom["ticker"] = t             # optional: Herkunft behalten
        Ycom = Ycom.rename("Y")        # saubere Series-Column

        X_list.append(Xcom)
        Y_list.append(Ycom)

    X_all = pd.concat(X_list, axis=0, ignore_index=True)
    if "ticker" in X_all.columns:
        ticker_dummies = pd.get_dummies(X_all.pop("ticker"), prefix="ticker")
        X_all = pd.concat([X_all, ticker_dummies], axis=1)

    Y_all = pd.concat(Y_list, axis=0, ignore_index=True)

    X_all.to_csv("DataStorage/X.csv", index=False)
    Y_all.to_csv("DataStorage/Y.csv", index=False)
    return X_all, Y_all


print(combine(["AAPL", "MSFT"]))