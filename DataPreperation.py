import pandas as pd
import numpy as np





def dataReader(ticker) -> pd.DataFrame:

    dataprep = pd.read_excel("mag7_1m_last8d.xlsx", sheet_name=ticker)

    print (dataprep.head)

    dataprep = dataprep.drop(columns=["close","open","high","low","ticker"])


    dataprep.to_csv(str(ticker+".csv"), index=False)

    return dataprep


APPLE = dataReader("AAPL")
print(APPLE.head())


def dataLabelEncoder(ticker) -> pd.DataFrame[2]:

   dataLabel = dataReader(ticker)
