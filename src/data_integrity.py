"""
Stage 01 — Reliability checks of the raw data (read-only).

  1. official_source   every file comes from geostat.ge/media and an official listing page
  2. complete_download size == bytes on the wire, SHA-256 == manifest, real format == expected
  3. archive_intact    ZIP CRC test passes; every extracted file equals its archive member
  4. year_identified   year recorded inside the data == year of the source file
                       (tables without a year field are linked by UID to one that has it)
  5. structure         all core tables present each year; child UIDs exist in the
                       household table; ECSTAT persons exist in the demographic table
  6. cross_year        each variable's label and value-label codes compared year to year

Outputs:
  reports/stage01_integrity_checks.csv     one row per check (status PASS / WARN / FAIL)
  reports/stage01_cross_year_consistency.csv  one row per (dataset, table, variable)

Requires data_profile.py to have been run (uses stage01_variable_profile.csv).
"""
from __future__ import annotations

import difflib
import json
import re
import zipfile
import zlib
from pathlib import Path

import pandas as pd
import pyreadstat

import data_collection as dc
from data_profile import canonical_table, VAR_OUT

REPORTS = dc.PROJECT_ROOT / "reports"
CHECKS_OUT = REPORTS / "stage01_integrity_checks.csv"
XYEAR_OUT = REPORTS / "stage01_cross_year_consistency.csv"

CORE_TABLES = {
    "lfs_2017_2019": {"id", "demographic", "ecstat", "personal_weights", "familysize", "deadmen"},
    "lfs_2020_2025": {"demographic", "ecstat"},
    "hies": {"123", "sysschedule", "familysize", "tblshinda02", "tblincomes", "tblexpenditures",
             "tblshinda01", "tblshinda04", "tblconsumption"},
}

checks: list[dict] = []


def add(check, status, detail="", dataset="", year="", table=""):
    checks.append(dict(check=check, dataset=dataset, year=year, table=table, status=status, detail=detail))


def extracted_dir(src: dc.Source) -> Path:
    return dc.RAW_DIR / dc.FOLDERS[src.group] / "extracted" / str(src.year)


def find_table(src: dc.Source, canonical: str) -> Path | None:
    for p in extracted_dir(src).rglob("*.sav"):
        if canonical_table(p.stem) == canonical:
            return p
    return None


def read(path: Path, cols=None) -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(str(path), usecols=cols)
    if "UID" in df.columns:  # 2018 stores UID as text in some tables
        df["UID"] = pd.to_numeric(df["UID"], errors="coerce")
    return df


# 1-2 ---------------------------------------------------------------------------
def check_sources_and_downloads(manifest: dict):
    for src in dc.SOURCES:
        rec = manifest.get(src.key)
        official = src.url.startswith("https://geostat.ge/media/") and \
            src.page.startswith("https://www.geostat.ge/en/")
        add("official_source", "PASS" if official else "FAIL", src.url, src.group, src.year or "", src.key)
        p = dc.local_path(src)
        if not rec or not p.exists():
            add("complete_download", "FAIL", "missing file or manifest entry", src.group, src.year or "", src.key)
            continue
        problems = []
        if dc.sha256(p) != rec["sha256"]:
            problems.append("sha256 differs from manifest")
        if p.stat().st_size != rec["bytes"]:
            problems.append("size differs from manifest")
        if dc.detect_format(p) != src.fmt:
            problems.append(f"format {dc.detect_format(p)} != {src.fmt}")
        add("complete_download", "FAIL" if problems else "PASS",
            "; ".join(problems) or f"{rec['bytes']:,} bytes, {src.fmt}, sha256 ok",
            src.group, src.year or "", src.key)


# 3 -----------------------------------------------------------------------------
def check_archives():
    for src in dc.SOURCES:
        if src.fmt != "zip":
            continue
        with zipfile.ZipFile(dc.local_path(src)) as z:
            bad = z.testzip()
            mism = []
            for info in z.infolist():
                if info.is_dir():
                    continue
                f = extracted_dir(src) / info.filename
                if not f.exists() or f.stat().st_size != info.file_size or \
                        zlib.crc32(f.read_bytes()) & 0xFFFFFFFF != info.CRC:
                    mism.append(info.filename)
            n = sum(1 for i in z.infolist() if not i.is_dir())
        status = "FAIL" if bad or mism else "PASS"
        add("archive_intact", status,
            f"{n} members, CRC ok, extracted copy identical" if status == "PASS"
            else f"corrupt={bad} mismatched={mism[:5]}", src.group, src.year, src.key)


# 4-5 ---------------------------------------------------------------------------
def check_years_and_structure():
    for src in dc.SOURCES:
        if src.fmt != "zip":
            continue
        y = src.year
        present = {canonical_table(p.stem) for p in extracted_dir(src).rglob("*.sav")}
        need = CORE_TABLES["hies"] if src.group == "hies" else \
            CORE_TABLES["lfs_2017_2019" if y <= 2019 else "lfs_2020_2025"]
        missing = need - present
        add("structure_core_tables", "FAIL" if missing else "PASS",
            f"missing {sorted(missing)}" if missing else f"{len(present)} tables, all core tables present",
            src.group, y)

        if src.group == "lfs":
            hh_tab = "id" if y <= 2019 else "demographic"
            hh = read(find_table(src, hh_tab))
            years = set(hh["Year"].dropna().astype(int))
            add("year_identified", "PASS" if years == {y} else "FAIL",
                f"Year field in {hh_tab} = {sorted(years)}", "lfs", y, hh_tab)
            hh_uids = set(hh["UID"].dropna())
            for p in extracted_dir(src).rglob("*.sav"):
                ct = canonical_table(p.stem)
                if ct == hh_tab:
                    continue
                d = read(p, ["UID"])
                orphan = int((~d["UID"].isin(hh_uids)).sum())
                add("structure_uid_linkage", "PASS" if orphan == 0 else "WARN",
                    f"{orphan} of {len(d)} rows have a UID not found in {hh_tab}"
                    + (" (incl. missing UID)" if d["UID"].isna().any() else ""), "lfs", y, p.stem)
            demo = read(find_table(src, "demographic"), ["UID", "MemberNo"])
            ec = read(find_table(src, "ecstat"), ["UID", "MemberNo", "Age"])
            k = demo.set_index(["UID", "MemberNo"]).index
            miss = int((~ec.set_index(["UID", "MemberNo"]).index.isin(k)).sum())
            add("structure_person_linkage", "PASS" if miss == 0 else "WARN",
                f"{miss} ECSTAT persons not in demographic; ECSTAT min age = {ec.Age.min():g}",
                "lfs", y, "ecstat")
        else:
            sched = read(find_table(src, "sysschedule"))
            q = read(find_table(src, "123"))
            qy = dict(zip(q.QuartNo.astype(int), q.YarNo.astype(int)))
            years = {qy.get(int(v), -1) for v in sched.QuartNo.dropna().unique()}
            add("year_identified", "PASS" if years == {y} else "FAIL",
                f"sysschedule quarters map to years {sorted(years)}", "hies", y, "sysschedule")
            uids = set(sched["UID"].dropna())
            for p in extracted_dir(src).rglob("*.sav"):
                ct = canonical_table(p.stem)
                if ct in ("sysschedule", "123", "tblavgprices"):
                    continue
                d = read(p, ["UID"])
                orphan = int((~d["UID"].isin(uids)).sum())
                add("structure_uid_linkage", "PASS" if orphan == 0 else "WARN",
                    f"{orphan} of {len(d)} rows have a UID not in sysschedule", "hies", y, p.stem)


# 5b ----------------------------------------------------------------------------
def lfs_codebook(path: Path) -> dict[str, str]:
    """{canonical_table: text of its section} from a Geostat LFS codebook (.docx)."""
    import docx
    d = docx.Document(str(path))
    names = [p.text.split(":", 1)[1].strip() for p in d.paragraphs if p.text.strip().startswith("Table:")]
    return {canonical_table(n): "\n".join(c.text for r in t.rows for c in r.cells)
            for n, t in zip(names, d.tables)}


def hies_codebook(path: Path) -> dict[str, str]:
    """{canonical_table: text of its section} from the IHS/HIES codebook (.pdf)."""
    import pypdf
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages)
    parts = re.split(r"(?m)^\s*Table\s+([A-Za-z0-9_]+)\s*$", text)
    return {parts[i].lower(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _documented(var: str, section: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(var)}(?![A-Za-z0-9_])", section, re.I) is not None


def check_metadata_vs_data():
    """Codebook tables/variables vs. what the data files actually contain.
    A variable is 'documented' if its exact name appears in its table's codebook section."""
    v = pd.read_csv(VAR_OUT, dtype=str, keep_default_na=False)
    v["year"] = v.year.astype(int)
    books = [("lfs", range(2017, 2020), "lfs_vars_2017_2019", lfs_codebook),
             ("lfs", range(2020, 2026), "lfs_vars_2020plus", lfs_codebook),
             ("hies", range(2017, 2026), "hies_vars", hies_codebook)]
    for ds, years, key, parser in books:
        src = next(s for s in dc.SOURCES if s.key == key)
        book = parser(dc.local_path(src))
        for y in years:
            data = v[(v.dataset == ds) & (v.year == y)]
            data_tabs = {t: list(g.variable) for t, g in data.groupby("canonical_table")}
            absent_tabs = sorted(set(book) - set(data_tabs))
            if absent_tabs:
                add("metadata_matches_data", "INFO",
                    f"{key}: codebook describes tables absent from {y} data: {absent_tabs}", ds, y)
            for tab, dvars in data_tabs.items():
                if tab not in book:
                    add("metadata_matches_data", "WARN", f"{key}: table not described in codebook",
                        ds, y, tab)
                    continue
                undoc = [x for x in dvars if not _documented(x, book[tab])]
                share = 1 - len(undoc) / max(len(dvars), 1)
                add("metadata_matches_data", "PASS" if not undoc else "WARN",
                    f"{key}: {share:.0%} of {len(dvars)} data vars documented"
                    + (f"; undocumented={undoc}" if undoc else ""), ds, y, tab)


# 6 -----------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(women|woman|female)\b", "female", s)
    s = re.sub(r"\b(men|man|male)\b", "male", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def cross_year() -> pd.DataFrame:
    v = pd.read_csv(VAR_OUT, dtype=str, keep_default_na=False)
    rows = []
    for (ds, ct, var), g in v.groupby(["dataset", "canonical_table", "variable"]):
        g = g.sort_values("year")
        years = g.year.tolist()
        base = g.iloc[-1]                                  # compare every year to the latest
        lab_sims = [_sim(r.label, base.label) for r in g.itertuples()]
        codes = [frozenset(json.loads(r.value_labels).keys()) if r.value_labels else frozenset()
                 for r in g.itertuples()]
        val_sims = []
        for r in g.itertuples():
            a = json.loads(r.value_labels) if r.value_labels else {}
            b = json.loads(base.value_labels) if base.value_labels else {}
            common = set(a) & set(b)
            val_sims += [_sim(a[c], b[c]) for c in common]
        types = set(g.spss_type)
        if len(years) == 1:
            status = "single_year"
        elif len(set(codes)) > 1:
            status = "CODES_CHANGED"
        elif len(types) > 1:
            status = "TYPE_CHANGED"
        elif min(lab_sims) < 0.6 or (val_sims and min(val_sims) < 0.6):
            status = "LABEL_CHANGED"
        elif min(lab_sims) < 1 or (val_sims and min(val_sims) < 1):
            status = "wording_only"
        else:
            status = "identical"
        rows.append(dict(dataset=ds, canonical_table=ct, variable=var,
                         years=",".join(years), n_years=len(years), status=status,
                         spss_types="|".join(sorted(types)),
                         min_label_similarity=round(min(lab_sims), 2),
                         min_value_label_similarity=round(min(val_sims), 2) if val_sims else "",
                         labels_by_year=" || ".join(f"{r.year}: {r.label}" for r in g.itertuples()
                                                     if r.label != base.label) or base.label,
                         code_sets_by_year=" || ".join(f"{r.year}: {sorted(c)}" for r, c in
                                                        zip(g.itertuples(), codes))
                         if status == "CODES_CHANGED" else ""))
    x = pd.DataFrame(rows)
    x.to_csv(XYEAR_OUT, index=False)
    for (ds, st), n in x.groupby(["dataset", "status"]).size().items():
        add("cross_year_definitions", "WARN" if st.isupper() else "PASS", f"{n} variables: {st}", ds)
    return x


def run():
    checks.clear()
    manifest = dc.load_manifest()
    check_sources_and_downloads(manifest)
    check_archives()
    check_years_and_structure()
    check_metadata_vs_data()
    cross_year()
    c = pd.DataFrame(checks)
    c.to_csv(CHECKS_OUT, index=False)
    return c


if __name__ == "__main__":
    c = run()
    print(c.groupby(["check", "status"]).size().unstack(fill_value=0).to_string())
    print()
    print(c[c.status != "PASS"].to_string(index=False, max_colwidth=110))
