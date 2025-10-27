import datetime
from pathlib import Path

import pandas as pd

import GloablVariableStorage
import LSEG as LS


def createExcel(
    universe: list[str],
    fields: list[str],
    start: datetime.datetime,
    end: datetime.datetime,
    interval: str,
    name:str
) -> None:
    """Einfach: Daten holen und als eine Excel-Datei speichern."""

    df = LS.getHistoryData(
        universe=universe, fields=fields, start=start, end=end, interval=interval
    )

    if df is None or df.empty:
        print("⚠️ Keine Daten zurückgegeben – keine Excel erstellt.")
        return

    out_dir = Path("DataStorage")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(name+".xlsx")

    # Ein Sheet, Name 'data'
    df.to_excel(out_path, index=False, sheet_name="data")
    print(f"✅ Excel gespeichert: {out_path}")



createExcel(universe= GloablVariableStorage.Portfolio,
                    fields= ["OPEN_PRC"],
                    start= datetime.datetime(2015, 1, 1),
                    end= datetime.datetime(2025, 10, 25) ,
                    interval='30min',
                    name="Portfolio")
