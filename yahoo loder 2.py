import yfinance as yf
import pandas as pd

import GloablVariableStorage


def downlaod_ticker(ticker,period,interval,nameofexcle):
    # Dictionary to collect DataFrames for each ticker
    sheets = {}

    # Magnificent 7 Ticker
    TICKERS = ticker
    PERIOD = period
    INTERVAL = interval

    for t in TICKERS:
        df = yf.download(
            t,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
            actions=False,
            group_by="column",
        )
        if df is None or df.empty:
            print(f"⚠️ Keine Daten für {t}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(t, axis=1, level=1)
            except Exception:
                df.columns = ["_".join([str(x) for x in tup if x]).strip("_") for tup in df.columns]
        df = df.reset_index()
        df.rename(columns={"Datetime": "date", "Date": "date"}, inplace=True)
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            except Exception:
                pass
        df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }, inplace=True)
        df["ticker"] = t
        sheets[t] = df

    # Eine Excel-Datei mit je einem Sheet pro Aktie speichern
    out_path = "DataStorage/"+nameofexcle+".xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        for t, d in sheets.items():
            d.to_excel(writer, index=False, sheet_name=t[:31])
    print(f"✅ Gesamtdatei gespeichert: {out_path} (Sheets: {', '.join(sheets.keys())})")

downlaod_ticker(GloablVariableStorage.ListofStock,"10y","1d","Mag7")

def dowload_Index(ticker, period,interval):
    df = yf.download(ticker, period=period, auto_adjust=True,interval=interval)
    df = df.reset_index()[["Date", "Close", "Volume"]]
    df.rename(columns={"Date": "date", "Close": "adj_close", "Volume": "volume"}, inplace=True)
    path = "DataStorage/" + ticker + ".csv"
    df.to_csv(path, index=False)
    print(f"Benchmark gespeichert als {path}")


#dowload_Index("SPY","10y","1d")
#dowload_Index("^VIX","10y","1d")
