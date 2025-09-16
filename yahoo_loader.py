# yahoo_loader.py
import pandas as pd
import numpy as np
import yfinance as yf

_INTRADAY = {"1m","2m","5m","15m","30m","60m","90m"}

def _choose_period(interval: str) -> str:
    if interval == "1m": return "7d"     # Yahoo-Limit
    if interval in {"2m","5m","15m"}: return "60d"
    if interval in {"30m","60m","90m"}: return "60d"
    return "max"

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sorgt dafür, dass df.columns Strings sind (MultiIndex -> flach)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if x is not None]).strip("_")
                      for tup in df.columns.to_list()]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, *cands):
    """Finde erste passende Spalte (case/space-insensitive)."""
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for want in cands:
        w = want.lower().replace(" ", "_")
        if w in norm:
            return norm[w]
    return None

def load_yahoo_data(
    tickers,
    start="2019-01-01",
    end="2024-01-01",
    interval="1d",
    auto_adjust=False,
    quiet=False,
):
    frames, failed = [], []
    use_period = interval in _INTRADAY
    period = _choose_period(interval) if use_period else None

    for t in tickers:
        try:
            if use_period:
                df = yf.download(
                    t, period=period, interval=interval,
                    auto_adjust=auto_adjust, progress=False, threads=False,
                    actions=False, group_by="column"   # <- wichtig: keine MultiIndex-Spalten
                )
            else:
                df = yf.download(
                    t, start=start, end=end, interval=interval,
                    auto_adjust=auto_adjust, progress=False, threads=False,
                    actions=False, group_by="column"
                )

            if df is None or df.empty:
                failed.append((t, "empty dataframe"))
                continue

            df = df.copy()
            df = _flatten_columns(df).reset_index()

            # Spalten robust finden (unterstützt Date/Datetime, Adj Close optional)
            date_col  = _find_col(df, "Date", "Datetime", "date", "datetime")
            open_col  = _find_col(df, "Open")
            high_col  = _find_col(df, "High")
            low_col   = _find_col(df, "Low")
            close_col = _find_col(df, "Close")
            vol_col   = _find_col(df, "Volume")
            adj_col   = _find_col(df, "Adj Close") or close_col  # Fallback, falls auto_adjust=True

            needed = [date_col, open_col, high_col, low_col, close_col, vol_col, adj_col]
            if any(c is None for c in needed):
                failed.append((t, f"missing columns: {needed}"))
                continue

            out = df[[date_col, open_col, high_col, low_col, close_col, adj_col, vol_col]].copy()
            out.columns = ["date","open","high","low","close","adj_close","volume"]
            out["ticker"] = t
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.dropna(subset=["date"])

            frames.append(out[["date","ticker","open","high","low","close","adj_close","volume"]])

        except Exception as e:
            failed.append((t, str(e)))
            if not quiet:
                print(f"[WARN] {t}: {e}")

    if not frames:
        if not quiet:
            print(f"[ERROR] no data returned; failures: {failed}")
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])

    res = pd.concat(frames, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    if failed and not quiet:
        print("Failed tickers:", failed)
    return res
