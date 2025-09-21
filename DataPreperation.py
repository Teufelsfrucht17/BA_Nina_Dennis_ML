import pandas as pd
import numpy as np





def dataReader(ticker) -> pd.DataFrame:

    for ticker in ticker:

        dataprep = pd.read_excel("testtageperiode.xlsx", sheet_name=ticker)

        print (dataprep.head)

        dataprep = dataprep.drop(columns=["close","open","high","low","ticker"])

        dataprep = dataprep.sort_values("date").reset_index(drop=True)

        dataprep.to_csv(str(ticker+".csv"), index=False)





    return dataprep





def featureEnegnier(ticker) -> pd.DataFrame:

    dataLabel = pd.DataFrame(dataReader(ticker))

    # Returnes 1m und 3m

    dataLabel["ret_1m"] = dataLabel["adj_close"].pct_change(1)
    dataLabel["ret_3m"] = dataLabel["adj_close"].pct_change(3)

    # Moving Average Gap
    dataLabel["sma_20"] = dataLabel["adj_close"].rolling(20).mean()
    dataLabel["sma_gap"] = dataLabel["adj_close"] / dataLabel["sma_20"]-1

    # volume Feature

    dataLabel["vol_mean_20"] = dataLabel["volume"].rolling(20).mean()
    dataLabel["turnover"] = dataLabel["volume"]/ (dataLabel["vol_mean_20"] + 1e-9)






    return dataLabel

def featuresplit (ticker):

    dataSplit = pd.DataFrame(featureEnegnier(ticker))

    dataSplit["Y"] = dataSplit["adj_close"].pct_change(5).shift(5)

    features = ["ret_1m", "ret_3m", "sma_gap", "turnover"]

    dataSplit = dataSplit.dropna(subset=features+["Y"]).reset_index(drop=True)


    X = dataSplit[features]
    Y = dataSplit["Y"]

    return X,Y




