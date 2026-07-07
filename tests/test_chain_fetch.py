"""Bundle transport: packaging roundtrip, hash verification, hostile archives."""

from __future__ import annotations

import io
import tarfile

import pytest

from optima.bundle_hash import content_hash
from optima.chain.fetch import FetchError, fetch_bundle, package_bundle


def _make_bundle(root, name="bundle"):
    b = root / name
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text('bundle_id = "t"\n')
    (b / "kernels" / "k.py").write_text("def f():\n    return 1\n")
    return b


def test_package_roundtrip(tmp_path):
    bundle = _make_bundle(tmp_path)
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    assert ch == content_hash(bundle)
    fetched = fetch_bundle(archive.as_uri(), ch, tmp_path / "cache")
    assert content_hash(fetched) == ch
    assert (fetched / "manifest.toml").exists()


def test_package_excludes_junk(tmp_path):
    bundle = _make_bundle(tmp_path)
    (bundle / "__pycache__").mkdir()
    (bundle / "__pycache__" / "k.cpython-311.pyc").write_bytes(b"junk")
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert not any("__pycache__" in n for n in names)
    # junk does not perturb identity either
    assert ch == content_hash(bundle)


def test_fetch_is_idempotent_and_detects_corrupted_cache(tmp_path):
    bundle = _make_bundle(tmp_path)
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    cache = tmp_path / "cache"
    first = fetch_bundle(archive.as_uri(), ch, cache)
    again = fetch_bundle(archive.as_uri(), ch, cache)
    assert first == again
    (first / "kernels" / "k.py").write_text("tampered = True\n")
    with pytest.raises(FetchError, match="re-hashes"):
        fetch_bundle(archive.as_uri(), ch, cache)


def test_fetch_rejects_hash_mismatch(tmp_path):
    bundle = _make_bundle(tmp_path)
    archive, _ = package_bundle(bundle, tmp_path / "out.tar.gz")
    with pytest.raises(FetchError, match="mismatch"):
        fetch_bundle(archive.as_uri(), "b" * 64, tmp_path / "cache")
    # nothing cached under the bogus hash
    assert not (tmp_path / "cache" / ("b" * 64)).exists()


def _write_tar(path, members):
    """members: list of (TarInfo, bytes|None)"""
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)


def _reg(name, data=b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def test_extract_rejects_symlink_member(tmp_path):
    evil = tarfile.TarInfo("bundle/link")
    evil.type = tarfile.SYMTYPE
    evil.linkname = "/etc/passwd"
    path = tmp_path / "evil.tar.gz"
    _write_tar(path, [_reg("bundle/manifest.toml"), (evil, None)])
    with pytest.raises(FetchError, match="not a regular file"):
        fetch_bundle(path.as_uri(), "a" * 64, tmp_path / "cache")


def test_extract_rejects_path_escape(tmp_path):
    for name in ("../outside.py", "/abs.py", "bundle/../../outside.py"):
        path = tmp_path / "evil.tar.gz"
        _write_tar(path, [_reg(name)])
        with pytest.raises(FetchError, match="escapes"):
            fetch_bundle(path.as_uri(), "a" * 64, tmp_path / "cache")
        assert not (tmp_path / "outside.py").exists()
        assert not (tmp_path / "cache" / "outside.py").exists()


def test_extract_rejects_hardlink_member(tmp_path):
    evil = tarfile.TarInfo("bundle/hard")
    evil.type = tarfile.LNKTYPE
    evil.linkname = "manifest.toml"
    path = tmp_path / "evil.tar.gz"
    _write_tar(path, [_reg("bundle/manifest.toml"), (evil, None)])
    with pytest.raises(FetchError, match="not a regular file"):
        fetch_bundle(path.as_uri(), "a" * 64, tmp_path / "cache")


def test_download_size_cap(tmp_path, monkeypatch):
    import optima.chain.fetch as fetch_mod

    bundle = _make_bundle(tmp_path)
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    monkeypatch.setattr(fetch_mod, "MAX_ARCHIVE_BYTES", 10)
    with pytest.raises(FetchError, match="exceeds"):
        fetch_bundle(archive.as_uri(), ch, tmp_path / "cache")


def test_fetch_rejects_unknown_scheme(tmp_path):
    with pytest.raises(FetchError, match="scheme"):
        fetch_bundle("ftp://example.com/x.tar.gz", "a" * 64, tmp_path / "cache")
