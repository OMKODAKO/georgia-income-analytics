"""
Stage 00 — Inventory of the raw Geostat microdata.

For every downloaded ZIP:
  1. extracts it (read-only copy) to data/raw/<survey>/extracted/<year>/
  2. reads the metadata of every SPSS (.sav) table with pyreadstat
  3. writes a variable inventory: reports/stage00_variable_inventory.csv
     (survey, year, table, n_rows, variable, label, value_labels)

The inventory is what the feasibility report is based on — no variable is
listed in the report unless it appears in this file.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pyreadstat

import data_collection as dc

OUT = dc.PROJECT_ROOT / "reports" / "stage00_variable_inventory.csv"
TABLES = dc.PROJECT_ROOT / "reports" / "stage00_tables.csv"


def extract(src: dc.Source) -> Path:
    target = dc.RAW_DIR / dc.FOLDERS[src.group] / "extracted" / str(src.year)
    if not target.exists():
        with zipfile.ZipFile(dc.local_path(src)) as z:
            for m in z.infolist():
                # guard against path traversal in archive member names
                if m.filename.startswith("/") or ".." in Path(m.filename).parts:
                    continue
                z.extract(m, target)
    return target


def inventory() -> pd.DataFrame:
    rows, tables = [], []
    for src in dc.SOURCES:
        if src.fmt != "zip" or not dc.local_path(src).exists():
            continue
        folder = extract(src)
        for sav in sorted(folder.rglob("*.sav")):
            _, meta = pyreadstat.read_sav(str(sav), metadataonly=True, encoding=None)
            table = sav.stem
            tables.append(dict(survey=src.group, year=src.year, table=table,
                               file=str(sav.relative_to(dc.PROJECT_ROOT)),
                               n_rows=meta.number_rows, n_vars=meta.number_columns))
            for var in meta.column_names:
                vl = meta.variable_value_labels.get(var, {})
                rows.append(dict(survey=src.group, year=src.year, table=table,
                                 n_rows=meta.number_rows, variable=var,
                                 label=meta.column_names_to_labels.get(var) or "",
                                 value_labels=json.dumps(vl, ensure_ascii=False) if vl else ""))
    pd.DataFrame(tables).to_csv(TABLES, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    d = inventory()
    print(f"{len(d)} variables in {d[['survey','year','table']].drop_duplicates().shape[0]} tables → {OUT}")
