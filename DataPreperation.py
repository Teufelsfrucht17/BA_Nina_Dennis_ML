import pandas as pd
import numpy as np





def dataReader(ticker,exclesheet) -> pd.DataFrame:



    dataprep = pd.read_excel(exclesheet, sheet_name=ticker)

    print (dataprep.head)

    dataprep = dataprep.drop(columns=["close","open","high","low","ticker"])

    dataprep = dataprep.sort_values("date").reset_index(drop=True)

    dataprep.to_csv(str(ticker+".csv"), index=False)





    return dataprep

dataReader(ticker="AAPL",exclesheet="mag7_1m_last8d.xlsx")



def featureEnegnier(ticker) -> pd.DataFrame:

    dataLabel = pd.DataFrame(dataReader(ticker,"testtageperiode.xlsx"))

    # Returnes 1m und 3m

    dataLabel["ret_1m"] = dataLabel["adj_close"].pct_change(1)
    dataLabel["ret_3m"] = dataLabel["adj_close"].pct_change(3)

    # Moving Average Gap
    dataLabel["sma_20"] = dataLabel["adj_close"].rolling(20).mean()
    dataLabel["sma_gap"] = dataLabel["adj_close"] / dataLabel["sma_20"]-1

    # volume Feature

    dataLabel["vol_mean_20"] = dataLabel["volume"].rolling(20).mean()
    dataLabel["turnover"] = dataLabel["volume"]/ (dataLabel["vol_mean_20"] + 1e-9)

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


def add_intraday_features(ticker) -> pd.DataFrame:
    intra = dataReader(ticker,"mag7_1m_last8d.xlsx")

    intra = intra.sort_values(["ticker","date"]).copy()
    g = intra.groupby("ticker")
    intra["ret_1"] = g["adj_close"].pct_change(1)
    intra["ret_3"] = g["adj_close"].pct_change(3)
    sma = g["adj_close"].transform(lambda s: s.rolling(20).mean())
    intra["sma_gap"] = intra["adj_close"]/sma - 1
    volm = g["volume"].transform(lambda s: s.rolling(20).mean())
    intra["turnover"] = intra["volume"]/(volm + 1e-9)
    # Label: z. B. 5-Min Zukunftsrendite
    intra["y"] = g["adj_close"].pct_change(5).shift(-5)
    return intra

print(add_intraday_features("AAPL").head())





def featuresplit (ticker):

    dataSplit = pd.DataFrame(featureEnegnier(ticker))

    dataSplit["Y"] = dataSplit["adj_close"].pct_change(5).shift(5)

    features = ["ret_1m", "ret_3m", "sma_gap", "turnover","mom_3m", "mom_1y", "mom_5y","mom_10y"]

    dataSplit = dataSplit.dropna(subset=features+["Y"]).reset_index(drop=True)


    X = dataSplit[features]
    Y = dataSplit["Y"]

    return X,Y


def combine(tickers):
    X_list, Y_list = [], []

    for t in tickers:
        Xcom, Ycom = featuresplit(t)   # Xcom: DataFrame, Ycom: Series
        Xcom = Xcom.copy()
      #Xcom["ticker"] = t             # optional: Herkunft behalten
        Ycom = Ycom.rename("Y")        # saubere Series-Column

        X_list.append(Xcom)
        Y_list.append(Ycom)

    X_all = pd.concat(X_list, axis=0, ignore_index=True)
    Y_all = pd.concat(Y_list, axis=0, ignore_index=True)

    X_all.to_csv("X.csv", index=False)
    Y_all.to_csv("Y.csv", index=False)
    return X_all, Y_all

print(combine(["AAPL", "MSFT"]))
