"""
Stage 01 — Data collection from official Geostat sources.

Downloads the public, anonymised microdata and metadata published by the
National Statistics Office of Georgia (Geostat) and records provenance.

For every file it:
  * checks the HTTP response (status 200, not an HTML error page),
  * checks the real file format from its content (magic bytes / ZIP structure),
  * checks the size against Content-Length,
  * skips files already downloaded whose SHA-256 matches the manifest,
  * never overwrites an existing raw file whose checksum differs (raises instead),
  * appends every action to data/raw/download_log.csv,
  * stores URL, listing page, size, SHA-256 and download time in data/raw/manifest.json.

Nothing here creates or modifies data: files are saved byte-for-byte as served.

Usage:
    python src/data_collection.py                 # download everything missing
    python src/data_collection.py --only lfs      # one group: lfs | hies | meta | agg
    python src/data_collection.py --check-links   # verify every official URL still works
    python src/data_collection.py --list          # print the registry
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"
LOG = RAW_DIR / "download_log.csv"
LOG_FIELDS = ["timestamp_utc", "key", "action", "http_status", "bytes", "remote_bytes",
              "detected_format", "sha256", "url", "message"]

# group code -> raw sub-folder
FOLDERS = {
    "lfs": "labour_force_survey",
    "hies": "household_survey",
    "meta": "metadata",
    "agg": "official_aggregates",
}

# Listing pages where the links were found (verified 2026-11-27).
LFS_DB_PAGE = "https://www.geostat.ge/en/modules/categories/130/labour-force-survey-databases"
HIES_DB_PAGE = ("https://www.geostat.ge/en/modules/categories/128/databases-of-2009-2016-"
                "integrated-household-survey-and-2017-households-income-and-expenditure-survey")
WAGES_PAGE = "https://www.geostat.ge/en/modules/categories/39/wages"
CPI_PAGE = "https://www.geostat.ge/en/modules/categories/26/consumer-price-index-inflation"
METHOD_PAGE = "https://www.geostat.ge/en/modules/categories/552"
LFS_Q_PAGE = "https://www.geostat.ge/en/modules/categories/562"
HIES_Q_PAGE = "https://www.geostat.ge/en/modules/categories/560"


@dataclass(frozen=True)
class Source:
    key: str          # unique id, also the local file stem
    group: str        # lfs | hies | meta | agg
    year: int | None
    url: str
    page: str         # listing page the link was taken from
    fmt: str          # expected real format: zip | docx | xlsx | pdf | html
    description: str


def _lfs(year, url):
    return Source(f"lfs_{year}", "lfs", year, url, LFS_DB_PAGE, "zip",
                  f"Labour Force Survey database {year} (SPSS, English)")


def _hies(year, url):
    return Source(f"hies_{year}", "hies", year, url, HIES_DB_PAGE, "zip",
                  f"Household Incomes and Expenditures Survey database {year} (SPSS, English)")


def _meta(key, url, page, fmt, desc):
    return Source(key, "meta", None, url, page, fmt, desc)


def _agg(key, url, page, desc):
    return Source(key, "agg", None, url, page, "xlsx", desc)


SOURCES: list[Source] = [
    # --- A. Labour Force Survey microdata -------------------------------------
    _lfs(2017, "https://geostat.ge/media/23299/2017_LFS_DB_ENG.zip"),
    _lfs(2018, "https://geostat.ge/media/30751/SPSS---Eng-2018.zip"),
    _lfs(2019, "https://geostat.ge/media/31881/SPSS---Eng-2019.zip"),
    _lfs(2020, "https://geostat.ge/media/48123/Labour-Force-Survey-Database-2020-Year.zip"),
    _lfs(2021, "https://geostat.ge/media/48124/Labour-Force-Survey-Database-2021-Year.zip"),
    _lfs(2022, "https://geostat.ge/media/54646/Labour-Force-Survey-Database-2022-Year.zip"),
    _lfs(2023, "https://geostat.ge/media/63553/Labour-Force-Survey-2023-eng.zip"),
    _lfs(2024, "https://geostat.ge/media/71337/Labour-Force-Survey-2024.zip"),
    _lfs(2025, "https://geostat.ge/media/79807/Labour-Force-Survey-2025.zip"),
    # --- B. Household Incomes and Expenditures Survey microdata ---------------
    # (2009-2016 Integrated Household Survey files also exist on the same page;
    #  out of scope: the project period is 2017-2025.)
    _hies(2017, "https://geostat.ge/media/62840/2017.zip"),
    _hies(2018, "https://geostat.ge/media/62841/2018.zip"),
    _hies(2019, "https://geostat.ge/media/62842/2019.zip"),
    _hies(2020, "https://geostat.ge/media/62843/2020.zip"),
    _hies(2021, "https://geostat.ge/media/62844/2021.zip"),
    _hies(2022, "https://geostat.ge/media/62845/2022.zip"),
    _hies(2023, "https://geostat.ge/media/62846/2023.zip"),
    _hies(2024, "https://geostat.ge/media/71164/2024.zip"),
    _hies(2025, "https://geostat.ge/media/79814/2025.zip"),
    # --- C. Supporting metadata ----------------------------------------------
    _meta("lfs_vars_2017_2019",
          "https://geostat.ge/media/27443/Description-of-Labour-Force-Survey-Database-Variables.docx",
          LFS_DB_PAGE, "docx", "Codebook: LFS database variables 2017-2019"),
    _meta("lfs_vars_2020plus",
          "https://geostat.ge/media/79812/Description-of-Labour-Force-Survey-Database-Variables.docx",
          LFS_DB_PAGE, "docx", "Codebook: LFS database variables 2020+"),
    _meta("hies_vars", "https://geostat.ge/media/62847/Description-of-IHS-database-variables---Eng.pdf",
          HIES_DB_PAGE, "pdf", "Codebook: IHS/HIES database variables"),
    _meta("lfs_methodology", "https://geostat.ge/media/80099/geostat_labour_force_survey_en.html",
          METHOD_PAGE, "html", "Methodology: Labour Force Statistics (Geostat metadata)"),
    _meta("wages_methodology", "https://geostat.ge/media/80100/geostat_wages_methodology_en.html",
          METHOD_PAGE, "html", "Methodology: Earnings Statistics (Geostat metadata)"),
    _meta("lfs_main_questionnaire_new", "https://geostat.ge/media/68082/LFS_MainQ_ENG.pdf",
          LFS_Q_PAGE, "pdf", "LFS Main Questionnaire (new methodology, 2020+)"),
    _meta("lfs_main_questionnaire_old",
          "https://geostat.ge/media/37157/2019-02-19---LFS-Main-Questionnaire-%28old-methodology%29.pdf",
          LFS_Q_PAGE, "pdf", "LFS Main Questionnaire (old methodology, to 2019)"),
    _meta("lfs_demographic_questionnaire", "https://geostat.ge/media/68081/LFS-DemographicQ_ENG.pdf",
          LFS_Q_PAGE, "pdf", "LFS Demographic Questionnaire"),
    _meta("hies_q_shinda05_1", "https://geostat.ge/media/57998/Shinda05-1_2019_Eng.pdf",
          HIES_Q_PAGE, "pdf", "HIES Shinda 05-1: employment and income questionnaire (per member)"),
    _meta("hies_q_shinda05", "https://geostat.ge/media/57997/Shinda05_2019_Eng.pdf",
          HIES_Q_PAGE, "pdf", "HIES Shinda 05: transfers, property income, savings"),
    # --- Official aggregate tables (benchmarks / deflator) --------------------
    # Geostat replaces /media/<id>/ when a table is updated; if a link 404s,
    # take the new one from the listing page.
    _agg("wages_gpg", "https://geostat.ge/media/83390/GPG_Eng.xlsx", WAGES_PAGE,
         "Official adjusted gender pay gap (source: LFS)"),
    _agg("wages_by_sex_annual", "https://geostat.ge/media/83306/03_Earnings-by-sex_annual.xlsx", WAGES_PAGE,
         "Average monthly nominal earnings of employees by activity and sex, annual"),
    _agg("wages_median", "https://geostat.ge/media/83297/Median-Monthly-Earnings.xlsx", WAGES_PAGE,
         "Median monthly earnings of employees"),
    _agg("wages_isco", "https://geostat.ge/media/83726/11-Earnings-%28ISCO%29.xlsx", WAGES_PAGE,
         "Average monthly nominal earnings by occupation (ISCO)"),
    _agg("wages_regions", "https://geostat.ge/media/83309/13_Earnings-by-regions_annual.xlsx", WAGES_PAGE,
         "Average monthly nominal earnings by region, annual"),
    _agg("cpi_2010", "https://geostat.ge/media/83813/consumer-price-index-2010%3D100.xlsx", CPI_PAGE,
         "Consumer Price Index, 2010=100, monthly (deflator)"),
]

# Items that cannot be collected automatically — documented, not faked.
NOT_AUTOMATABLE = [
    dict(item="HIES individual (per-member) income — Shinda 05-1",
         reason="Questionnaire is published, but the public database contains no person-level "
                "income table (only household-level tblincomes / tblshinda04).",
         alternative="Use LFS earnings for individual analysis; request microdata access from Geostat "
                     "(info@geostat.ge) if needed."),
    dict(item="LFS exact earnings amount (question B17 / B24)",
         reason="Public files contain only the banded variable (B17_B18 / B24_B25).",
         alternative="Interval methods; exact amounts only via Geostat confidential-access request."),
    dict(item="Sampling design variables (strata, PSU)",
         reason="Not included in public LFS/HIES files.",
         alternative="Household-clustered SEs (UID/DiaryID) as approximation."),
    dict(item="Geostat Terms of Use",
         reason="Web page, not a dataset; must be read and accepted by a person.",
         alternative="https://www.geostat.ge/en → Services → Terms of Use (manual review)."),
    dict(item="Dedicated HIES methodology document",
         reason="Not found on geostat.ge (methodology page lists only general UN/ILO manuals).",
         alternative="Codebook hies_vars.pdf + questionnaires inside each HIES ZIP."),
]


# ------------------------------------------------------------------ helpers --
def local_path(src: Source) -> Path:
    return RAW_DIR / FOLDERS[src.group] / f"{src.key}.{src.fmt}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_format(path: Path) -> str:
    """Identify the real file type from its content, not its extension."""
    with path.open("rb") as fh:
        head = fh.read(512)
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04") and zipfile.is_zipfile(path):
        names = set(zipfile.ZipFile(path).namelist())
        if "word/document.xml" in names:
            return "docx"
        if "xl/workbook.xml" in names:
            return "xlsx"
        return "zip"
    if head.startswith(b"$FL2") or head.startswith(b"$FL3"):
        return "sav"
    if b"<html" in head.lower() or b"<!doctype html" in head.lower():
        return "html"
    return "unknown"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False))


def log(row: dict, path: Path | None = None) -> None:
    path = path or LOG
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS} |
                   {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (research data download)"
    return s


# ----------------------------------------------------------------- download --
def download(src: Source, dest: Path, session: requests.Session, retries: int = 6,
             log_path: Path | None = None) -> dict:
    """Download one file with exponential back-off (Geostat resets bursty clients).

    Validates HTTP status, content-type, Content-Length and real format before the
    file is moved into place. Returns the manifest record."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = ""
    for attempt in range(1, retries + 1):
        status = None
        try:
            with session.get(src.url, stream=True, timeout=120) as r:
                status = r.status_code
                if r.status_code != 200:
                    raise IOError(f"HTTP {r.status_code}")
                ctype = r.headers.get("Content-Type", "")
                if src.fmt != "html" and "text/html" in ctype:
                    raise IOError(f"got HTML page instead of {src.fmt}")
                remote = int(r.headers.get("Content-Length", 0))
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                # With gzip transfer-encoding Content-Length is the *compressed* size,
                # so compare it with the bytes actually received on the wire.
                wire = r.raw.tell()
            size = tmp.stat().st_size
            if size == 0:
                raise IOError("empty file")
            if remote and wire != remote:
                raise IOError(f"incomplete: received {wire} of {remote} bytes")
            fmt = detect_format(tmp)
            if fmt != src.fmt:
                raise IOError(f"format mismatch: expected {src.fmt}, content is {fmt}")
            if fmt == "zip":
                bad = zipfile.ZipFile(tmp).testzip()
                if bad:
                    raise IOError(f"corrupt ZIP member {bad}")
            tmp.replace(dest)
            rec = {**asdict(src), "local_file": str(dest.relative_to(PROJECT_ROOT))
                   if dest.is_relative_to(PROJECT_ROOT) else str(dest),
                   "bytes": size, "remote_bytes": remote, "detected_format": fmt,
                   "sha256": sha256(dest),
                   "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            log(dict(key=src.key, action="downloaded", http_status=status, bytes=size,
                     remote_bytes=remote, detected_format=fmt, sha256=rec["sha256"], url=src.url),
                log_path)
            return rec
        except Exception as exc:  # network resets are common; retry
            last = str(exc)
            tmp.unlink(missing_ok=True)
            log(dict(key=src.key, action=f"attempt_{attempt}_failed", http_status=status or "",
                     url=src.url, message=last[:200]), log_path)
            if attempt < retries:
                time.sleep(min(5 * 2 ** (attempt - 1), 120))
    log(dict(key=src.key, action="failed", url=src.url, message=last[:200]), log_path)
    raise RuntimeError(f"Could not download {src.url}: {last}")


def run(only: str | None = None, pause: float = 8.0) -> dict:
    manifest = load_manifest()
    session = _session()
    failed = []
    for src in SOURCES:
        if only and src.group != only:
            continue
        dest = local_path(src)
        if dest.exists():
            rec = manifest.get(src.key)
            digest = sha256(dest)
            if rec and rec["sha256"] == digest:
                log(dict(key=src.key, action="skipped_present", bytes=dest.stat().st_size,
                         sha256=digest, url=src.url, detected_format=detect_format(dest)))
                print(f"✓ {src.key} already present (checksum ok)")
                continue
            # never silently overwrite raw data
            raise RuntimeError(f"{dest} exists but does not match the manifest — inspect manually")
        print(f"↓ {src.key}  {src.url}")
        try:
            manifest[src.key] = download(src, dest, session)
            save_manifest(manifest)
        except RuntimeError as e:
            failed.append(src.key)
            print(f"✗ {e}")
        time.sleep(pause)  # be polite to the server
    if failed:
        print("FAILED:", failed)
    return manifest


def check_links(pause: float = 4.0) -> list[dict]:
    """Verify each official URL still answers 200 with the same size as our copy.
    Uses a 1-byte range GET (HEAD requests are reset by the server)."""
    session, out = _session(), []
    manifest = load_manifest()
    for src in SOURCES:
        res = dict(key=src.key, url=src.url, http_status="", remote_bytes="", local_bytes="", ok=False)
        for attempt in range(4):
            try:
                with session.get(src.url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60) as r:
                    res["http_status"] = r.status_code
                    cr = r.headers.get("Content-Range", "")
                    res["remote_bytes"] = int(cr.split("/")[-1]) if "/" in cr else \
                        int(r.headers.get("Content-Length", 0))
                break
            except Exception as exc:
                res["message"] = str(exc)[:120]
                time.sleep(10 * (attempt + 1))
        rec = manifest.get(src.key)
        res["local_bytes"] = rec["bytes"] if rec else ""
        # compare with the size *on the wire* at download time (gzip-served HTML differs from file size)
        expected = (rec.get("remote_bytes") or rec["bytes"]) if rec else None
        res["ok"] = res["http_status"] in (200, 206) and (expected is None or res["remote_bytes"] == expected)
        log(dict(key=src.key, action="link_check_ok" if res["ok"] else "link_check_FAIL",
                 http_status=res["http_status"], remote_bytes=res["remote_bytes"],
                 bytes=res["local_bytes"], url=src.url, message=res.get("message", "")))
        print(f"{'OK ' if res['ok'] else 'BAD'} {res['http_status']} {src.key}")
        out.append(res)
        time.sleep(pause)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(FOLDERS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check-links", action="store_true")
    a = ap.parse_args()
    if a.list:
        for s in SOURCES:
            print(f"{s.group:5} {s.key:30} {s.url}")
    elif a.check_links:
        check_links()
    else:
        run(a.only)
