"""Stage 00 checks: raw files exist, are untouched (checksum), are readable,
and contain the variables the feasibility report relies on."""
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_collection as dc  # noqa: E402

MANIFEST = dc.load_manifest()


def test_registry_only_official_hosts():
    for s in dc.SOURCES:
        assert s.url.startswith("https://geostat.ge/media/"), s.url


def test_registry_keys_unique():
    keys = [s.key for s in dc.SOURCES]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("src", dc.SOURCES, ids=lambda s: s.key)
def test_file_downloaded_and_checksum_matches(src):
    p = dc.local_path(src)
    assert p.exists(), f"missing {p} — run src/data_collection.py"
    assert src.key in MANIFEST
    assert MANIFEST[src.key]["sha256"] == dc.sha256(p)
    assert MANIFEST[src.key]["bytes"] == p.stat().st_size


@pytest.mark.parametrize("src", [s for s in dc.SOURCES if s.url.endswith(".zip")], ids=lambda s: s.key)
def test_zip_is_valid(src):
    with zipfile.ZipFile(dc.local_path(src)) as z:
        assert z.testzip() is None
        assert any(n.lower().endswith(".sav") for n in z.namelist()), "no SPSS file inside"
