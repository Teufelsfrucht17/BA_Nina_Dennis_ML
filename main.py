# main.py
# Central entrypoint: defines tickers, calls yahoo_loader, saves to Excel

from pathlib import Path
import pandas as pd
from yahoo_loader import load_yahoo_data

# === User configuration ===
TICKERS  = ['AAPL','MSFT','NVDA','AMZN','GOOGL']   # dein Universum
START    = '2025-09-15'
END      = '2025-09-16'
INTERVAL = '1d'   # '1d' | '1h' | '5m' | '1m'


# Output path
OUT_DIR = Path('./data_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUT_DIR / 'yahoo_prices.xlsx'

def main():
    df = load_yahoo_data( TICKERS,
    interval="5m",         # period=60d intern
    auto_adjust=False)
    if df.empty:
        print('⚠️ Keine Daten geladen – prüfe Ticker/Zeitraum/Interval.')
        return

    with pd.ExcelWriter(EXCEL_PATH, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
        df.to_excel(writer, index=False, sheet_name='combined')
        for t in sorted(df['ticker'].unique()):
            df_t = df[df['ticker'] == t]
            sheet = t[:31]  # Excel-Sheet-Namen max. 31 Zeichen
            df_t.to_excel(writer, index=False, sheet_name=sheet)

    print(f'✅ Excel gespeichert: {EXCEL_PATH}')

if __name__ == '__main__':
    main()
