import datetime
from pathlib import Path

import pandas as pd

import GloablVariableStorage
import LSEG as LS


def exceltextwriter(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        print("⚠️ Keine Daten zurückgegeben – keine Excel erstellt.")
        return

    out_dir = Path("DataStorage")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.xlsx"

    with pd.ExcelWriter(
            out_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss"
    ) as writer:
        for column in df.columns:
            sheet_name = str(column).strip()[:31] or "sheet"
            df[[column]].to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"✅ Excel gespeichert: {out_path}")
    return

def createExcel(
    universe: list[str],
    fields: list[str],
    start: datetime.datetime,
    end: datetime.datetime,
    interval: str,
    name: str,
) -> None:
    """Daten holen und als Excel speichern (kombiniert + je Aktie eigenes Sheet)."""

    df = LS.getHistoryData(
        universe=universe, fields=fields, start=start, end=end, interval=interval )
    exceltextwriter(df, name)
    return




createExcel(
    universe=GloablVariableStorage.Portfolio,
    fields=["OPEN_PRC"],
    start=datetime.datetime(2015, 1, 1),
    end=datetime.datetime(2025, 10, 25),
    interval="30min",
    name="Portfolio",
)
