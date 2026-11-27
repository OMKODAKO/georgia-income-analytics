"""Stage 01 tests — raw data collection, readability, metadata and download log.

Run:  .venv/bin/python -m pytest -q tests
(The download-log tests use a local HTTP server and small throw-away files in a
temporary folder; they never touch data/raw.)"""
import csv
import http.server
import io
import json
import sys
import threading
import zipfile
from pathlib import Path

import pandas as pd
import pyreadstat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_collection as dc  # noqa: E402

REPORTS = ROOT / "reports"
TABLES = pd.read_csv(REPORTS / "stage01_table_profile.csv")
VARS = pd.read_csv(REPORTS / "stage01_variable_profile.csv", dtype=str, keep_default_na=False)
CHECKS = pd.read_csv(REPORTS / "stage01_integrity_checks.csv")
YEARS = range(2017, 2026)


def tables_of(ds, year, canonical):
    return TABLES[(TABLES.dataset == ds) & (TABLES.year == year) & (TABLES.canonical_table == canonical)]


# 1. all files exist --------------------------------------------------------------
@pytest.mark.parametrize("src", dc.SOURCES, ids=lambda s: s.key)
def test_1_file_exists_with_manifest_entry(src):
    p = dc.local_path(src)
    assert p.exists() and p.stat().st_size > 0
    rec = dc.load_manifest()[src.key]
    assert rec["url"] == src.url and rec["sha256"] == dc.sha256(p)
    assert rec["downloaded_utc"]


def test_1_every_year_extracted():
    for group in ("lfs", "hies"):
        for y in YEARS:
            assert any((dc.RAW_DIR / dc.FOLDERS[group] / "extracted" / str(y)).rglob("*.sav")), (group, y)


# 2. files are readable -----------------------------------------------------------
def test_2_every_sav_readable():
    for f in TABLES.file:
        _, meta = pyreadstat.read_sav(str(ROOT / f), metadataonly=True)
        assert meta.number_columns > 0


@pytest.mark.parametrize("src", [s for s in dc.SOURCES if s.fmt != "zip"], ids=lambda s: s.key)
def test_2_metadata_and_aggregates_readable(src):
    p = dc.local_path(src)
    assert dc.detect_format(p) == src.fmt
    if src.fmt == "xlsx":
        assert len(pd.read_excel(p, sheet_name=None, header=None)) > 0
    elif src.fmt == "docx":
        import docx
        assert docx.Document(str(p)).paragraphs
    elif src.fmt == "pdf":
        import pypdf
        assert len(pypdf.PdfReader(str(p)).pages) > 0
    elif src.fmt == "html":
        text = p.read_text(encoding="utf-8", errors="strict")   # strict: fails on bad bytes
        assert "<html" in text.lower()
        if src.key == "lfs_methodology":                       # Georgian script decodes correctly
            assert "საქართველოს სტატისტიკის ეროვნული სამსახური" in text


def test_2_labels_decode_cleanly():
    assert TABLES.n_label_encoding_issues.sum() == 0
    assert set(TABLES.encoding) <= {"UTF-8", "windows-1252"}


# 3. datasets are not empty -------------------------------------------------------
def test_3_no_empty_table():
    assert (TABLES.n_rows > 0).all() and (TABLES.n_cols > 0).all()
    assert len(TABLES) == 30 + 19 * 9


# 4. gender variable where expected -----------------------------------------------
@pytest.mark.parametrize("year", YEARS)
def test_4_gender_present_and_coded(year):
    expect = [("lfs", "demographic", "Sex"), ("lfs", "ecstat", "Sex"), ("hies", "tblshinda02", "Gender")]
    for ds, tab, var in expect:
        r = VARS[(VARS.dataset == ds) & (VARS.year == str(year)) & (VARS.canonical_table == tab)
                 & (VARS.variable == var)]
        assert len(r) == 1, (ds, tab, var, year)
        labels = json.loads(r.value_labels.iloc[0])
        assert set(labels) == {"1.0", "2.0"}
        assert labels["1.0"].lower().startswith(("female", "wom")), labels
        assert float(r.pct_missing.iloc[0]) == 0.0


# 5. income variable where expected (and NOT where it should not be) ---------------
@pytest.mark.parametrize("year", YEARS)
def test_5_individual_earnings_in_lfs(year):
    var = "B17_B18_Net_earnings" if year <= 2019 else "B24_B25_Net_earnings"
    r = VARS[(VARS.dataset == "lfs") & (VARS.year == str(year)) & (VARS.variable == var)]
    assert len(r) == 1 and r.canonical_table.iloc[0] == "ecstat"
    assert "GEL" in r.value_labels.iloc[0]


@pytest.mark.parametrize("year", YEARS)
def test_5_household_income_in_hies(year):
    r = VARS[(VARS.dataset == "hies") & (VARS.year == str(year)) & (VARS.canonical_table == "tblincomes")]
    assert {"UID", "ShemDaq", "Shemosavalisul"} <= set(r.variable)
    assert "MemberNo" not in set(r.variable)        # household level: no person id


def test_5_no_person_level_income_in_hies():
    person_tabs = VARS[(VARS.dataset == "hies") & (VARS.variable == "MemberNo")].canonical_table.unique()
    p = VARS[(VARS.dataset == "hies") & VARS.canonical_table.isin(person_tabs)]
    assert not p.label.str.contains(r"\b(?:incomes?|wages?|salar(?:y|ies)|earnings?)\b", case=False).any()


# 6. metadata matches dataset -----------------------------------------------------
def test_6_no_failed_integrity_check():
    assert (CHECKS.status == "FAIL").sum() == 0, CHECKS[CHECKS.status == "FAIL"]


def test_6_years_identified_correctly():
    y = CHECKS[CHECKS.check == "year_identified"]
    assert len(y) == 18 and (y.status == "PASS").all()


def test_6_key_variables_documented_in_codebooks():
    from data_integrity import lfs_codebook, hies_codebook, _documented
    get = lambda k: dc.local_path(next(s for s in dc.SOURCES if s.key == k))
    old, new, hies = lfs_codebook(get("lfs_vars_2017_2019")), lfs_codebook(get("lfs_vars_2020plus")), \
        hies_codebook(get("hies_vars"))
    for var in ["Sex", "Age", "B7_Status", "B17_B18_Net_earnings", "C2_M_Usually_hours", "I1_Education"]:
        assert _documented(var, old["ecstat"]), var
    assert _documented("P_Weights", old["personal_weights"])
    for var in ["Sex", "Age", "Status", "B24_B25_Net_earnings", "M_Usually_hours", "Education",
                "P_Weights", "Region"]:
        assert _documented(var, new["ecstat"]), var
    for var in ["ShemDaq", "Shemosavalisul"]:
        assert _documented(var, hies["tblincomes"]), var
    assert _documented("Weights", hies["sysschedule"])


def test_6_data_dictionaries_cover_every_variable():
    dd = ROOT / "reports" / "data_dictionary"
    for ds in ("lfs", "hies"):
        d = pd.read_csv(dd / f"{ds}_data_dictionary.csv")
        expected = VARS[VARS.dataset == ds][["canonical_table", "variable"]].drop_duplicates()
        assert len(d) == len(expected)
        for col in ["Variable_Name", "Meaning", "Data_Type", "Possible_Values", "Missing_Values", "Source"]:
            assert d[col].notna().all(), (ds, col)
    agg = pd.read_csv(dd / "official_aggregates_data_dictionary.csv")
    assert set(agg.Variable_Name) == {s.key for s in dc.SOURCES if s.group == "agg"}


# 7. the download log works -------------------------------------------------------
def test_7_real_log_covers_every_source():
    with dc.LOG.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and set(rows[0]) == set(dc.LOG_FIELDS)
    done = {r["key"] for r in rows if r["action"] in ("downloaded", "migrated_stage00_layout")}
    assert {s.key for s in dc.SOURCES} <= done
    checked = {r["key"] for r in rows if r["action"] == "link_check_ok"}
    assert {s.key for s in dc.SOURCES} <= checked


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "test fixture")
    return buf.getvalue()


@pytest.fixture
def server(tmp_path):
    """Local HTTP server: /ok.zip (valid), /fake.zip (HTML page), /short.zip (truncated)."""
    payload = _zip_bytes()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ok.zip":
                body, ctype, length = payload, "application/zip", len(payload)
            elif self.path == "/fake.zip":
                body, ctype = b"<!doctype html><html>error</html>", "application/zip"
                length = len(body)
            elif self.path == "/short.zip":
                body, ctype, length = payload[:10], "application/zip", len(payload)
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _src(base, name):
    return dc.Source(f"t_{name}", "lfs", 2099, f"{base}/{name}.zip", "local-test", "zip", "fixture")


def _log_rows(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_7_download_logs_and_validates(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "RAW_DIR", tmp_path)
    monkeypatch.setattr(dc, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(dc, "LOG", tmp_path / "log.csv")
    ok = _src(server, "ok")
    monkeypatch.setattr(dc, "SOURCES", [ok])
    m = dc.run(pause=0)
    rec = m[ok.key]
    assert rec["detected_format"] == "zip" and rec["bytes"] == len(_zip_bytes())
    assert rec["url"] == ok.url and rec["downloaded_utc"]
    rows = _log_rows(dc.LOG)
    assert rows[-1]["action"] == "downloaded" and rows[-1]["http_status"] == "200"
    # second run: no re-download, logged as skipped
    dc.run(pause=0)
    assert _log_rows(dc.LOG)[-1]["action"] == "skipped_present"


@pytest.mark.parametrize("name,reason", [("fake", "format mismatch"), ("short", ""), ("missing", "HTTP 404")])
def test_7_bad_downloads_rejected_and_logged(server, tmp_path, monkeypatch, name, reason):
    monkeypatch.setattr(dc, "LOG", tmp_path / "log.csv")
    src = _src(server, name)
    dest = tmp_path / "out.zip"
    with pytest.raises(RuntimeError):
        dc.download(src, dest, dc._session(), retries=1)
    assert not dest.exists() and not (tmp_path / "out.zip.part").exists()
    rows = _log_rows(dc.LOG)
    assert rows[-1]["action"] == "failed" and reason in rows[-1]["message"]


def test_7_existing_raw_file_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "RAW_DIR", tmp_path)
    monkeypatch.setattr(dc, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(dc, "LOG", tmp_path / "log.csv")
    src = _src("http://127.0.0.1:9", "ok")
    p = dc.local_path(src)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"something else")
    dc.save_manifest({src.key: {"sha256": "0" * 64}})
    monkeypatch.setattr(dc, "SOURCES", [src])
    with pytest.raises(RuntimeError, match="does not match"):
        dc.run(pause=0)
    assert p.read_bytes() == b"something else"
