"""Hardened transport for content-addressed proposal archives.

Production follows HTTPS only and pins each request to DNS answers reviewed as
globally routable while retaining TLS hostname verification.  Archive extraction
accepts only members covered by :func:`optima.bundle_hash.content_hash`; ignored
Python/editor/VCS noise and unneeded empty directories are rejected rather than
becoming executable, unhashed bytes beside the submitted delta.
"""

from __future__ import annotations

import http.client
import ipaddress
import logging
import os
import queue
import re
import socket
import ssl
import stat
import tarfile
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse

from optima.bundle_hash import (
    _SKIP_DIRS,
    _SKIP_SUFFIXES,
    _iter_files,
    content_hash,
)

logger = logging.getLogger("optima.chain.fetch")

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 4096
FETCH_TIMEOUT_S = 60.0
MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_TRANSFER_CHUNK_BYTES = 1024 * 1024
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


class FetchError(RuntimeError):
    """A proposal could not be fetched, extracted, or authenticated."""


class FetchTransientError(FetchError):
    """The remote transport may recover without changing proposal identity."""

    retryable = True


def package_bundle(
    bundle_dir: str | Path, out_path: str | Path | None = None
) -> tuple[Path, str]:
    """Package exactly the regular files covered by bundle identity."""

    root = Path(bundle_dir).resolve(strict=True)
    ch = content_hash(root)
    out = Path(out_path) if out_path else Path(f"{root.name}.tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for path in _iter_files(root):
            relative = path.relative_to(root).as_posix()
            tar.add(path, arcname=f"{root.name}/{relative}", recursive=False)
    return out, ch


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FetchTransientError("bundle transfer exceeded its absolute deadline")
    return remaining


def _public_ip(
    value: str, *, context: str
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        raise FetchError(f"{context} is not a canonical IP address") from None
    if not address.is_global:
        raise FetchError(
            f"{context} resolves to a non-public destination ({address.compressed})"
        )
    return address


def _resolve_addresses(hostname: str, port: int, *, deadline: float) -> tuple[str, ...]:
    """Bound libc/NSS resolution by the transfer's absolute deadline."""

    result: queue.Queue[object] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result.put(
                socket.getaddrinfo(
                    hostname,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                )
            )
        except BaseException as exc:  # passed back to the trusted caller
            result.put(exc)

    threading.Thread(target=resolve, name="optima-bundle-dns", daemon=True).start()
    try:
        resolved = result.get(timeout=_remaining(deadline))
    except queue.Empty:
        raise FetchTransientError(
            "bundle host DNS resolution exceeded the transfer deadline"
        ) from None
    if isinstance(resolved, BaseException):
        raise FetchTransientError(
            f"bundle host DNS resolution failed: {resolved}"
        ) from None
    addresses = tuple(sorted({str(answer[4][0]) for answer in resolved}))
    if not addresses:
        raise FetchError("bundle host DNS resolution returned no addresses")
    for address in addresses:
        _public_ip(address, context="bundle host")
    return addresses


def _validated_https_url(url: str, *, deadline: float):
    if (
        not isinstance(url, str)
        or not url
        or len(url) > 8_192
        or not url.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
    ):
        raise FetchError("bundle URL is empty, oversized, or non-canonical")
    try:
        parsed = urlparse(url)
        port = parsed.port or 443
    except ValueError as exc:
        raise FetchError(f"bundle URL is malformed: {exc}") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not 1 <= port <= 65_535
    ):
        raise FetchError(
            "production bundle URL scheme must be HTTPS without credentials or fragments"
        )
    return parsed, _resolve_addresses(parsed.hostname, port, deadline=deadline)


def _open_pinned_https(
    hostname: str,
    port: int,
    addresses: tuple[str, ...],
    *,
    deadline: float,
) -> http.client.HTTPSConnection:
    """Connect to reviewed IPs while retaining TLS SNI/hostname checks."""

    context = ssl.create_default_context()
    failures: list[str] = []
    for raw_address in addresses:
        address = _public_ip(raw_address, context="bundle host")
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        wrapped = None
        try:
            sock.settimeout(_remaining(deadline))
            destination = (
                (address.compressed, port, 0, 0)
                if family == socket.AF_INET6
                else (address.compressed, port)
            )
            sock.connect(destination)
            sock.settimeout(_remaining(deadline))
            wrapped = context.wrap_socket(sock, server_hostname=hostname)
            peer = _public_ip(str(wrapped.getpeername()[0]), context="connected bundle peer")
            if peer != address:
                raise FetchError("connected bundle peer differs from the reviewed address")
            wrapped.settimeout(_remaining(deadline))
            connection = http.client.HTTPSConnection(
                hostname,
                port,
                timeout=_remaining(deadline),
                context=context,
            )
            connection.sock = wrapped
            return connection
        except FetchError:
            if wrapped is not None:
                wrapped.close()
            else:
                sock.close()
            raise
        except (OSError, ssl.SSLError) as exc:
            failures.append(f"{address.compressed}:{type(exc).__name__}")
            if wrapped is not None:
                wrapped.close()
            else:
                sock.close()
    raise FetchTransientError(
        "could not establish pinned HTTPS connection: " + ",".join(failures)[:1024]
    )


def _response_read_socket(response: http.client.HTTPResponse):
    stream = getattr(response, "fp", None)
    raw = getattr(stream, "raw", stream)
    sock = getattr(raw, "_sock", None)
    if sock is None or not callable(getattr(sock, "settimeout", None)):
        raise FetchTransientError(
            "bundle response does not expose a deadline-controlled body socket"
        )
    return sock


def _download_https(url: str, destination: Path, max_bytes: int, *, deadline: float) -> None:
    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        parsed, addresses = _validated_https_url(current, deadline=deadline)
        port = parsed.port or 443
        connection = _open_pinned_https(
            parsed.hostname, port, addresses, deadline=deadline
        )
        try:
            if connection.sock is None:
                raise FetchError("HTTPS connection exposed no peer socket")
            wire_socket = connection.sock
            wire_socket.settimeout(_remaining(deadline))
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            connection.request(
                "GET",
                path,
                headers={
                    "Accept": "application/octet-stream, application/gzip",
                    "Connection": "close",
                    "User-Agent": "optima-validator-bundle-fetch/1",
                },
            )
            wire_socket.settimeout(_remaining(deadline))
            response = connection.getresponse()
            if response.status in _REDIRECT_STATUSES:
                location = response.getheader("Location")
                response.close()
                if not location:
                    raise FetchError("HTTPS redirect omitted its Location header")
                if redirect_count >= MAX_REDIRECTS:
                    raise FetchError("bundle URL exceeded the redirect limit")
                current = urljoin(current, location)
                _validated_https_url(current, deadline=deadline)
                continue
            if response.status != 200:
                error = (
                    FetchTransientError
                    if response.status in {408, 425, 429} or response.status >= 500
                    else FetchError
                )
                raise error(f"bundle host returned HTTP status {response.status}")
            body_socket = _response_read_socket(response)
            raw_length = response.getheader("Content-Length")
            declared = None
            if raw_length is not None:
                try:
                    declared = int(raw_length)
                except (TypeError, ValueError):
                    raise FetchError("bundle response Content-Length is invalid") from None
                if declared < 0 or declared > max_bytes:
                    raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(destination, flags, 0o600)
            total = 0
            try:
                while not response.isclosed():
                    body_socket.settimeout(_remaining(deadline))
                    chunk = response.read(_TRANSFER_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise FetchError("bundle archive write made no progress")
                        view = view[written:]
                if declared is not None and total != declared:
                    raise FetchTransientError(
                        "bundle response closed before its declared Content-Length"
                    )
                _remaining(deadline)
                os.fsync(fd)
            finally:
                os.close(fd)
            return
        except FetchError:
            raise
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise FetchTransientError(
                f"download failed for {current}: {type(exc).__name__}: {exc}"
            ) from None
        finally:
            connection.close()
    raise FetchError("bundle URL exceeded the redirect limit")


def _copy_local_archive_for_testing(
    url: str, destination: Path, max_bytes: int, *, deadline: float
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or parsed.query:
        raise FetchError("test-only local bundle URL must be file:// on this host")
    source = Path(unquote(parsed.path))
    try:
        before = source.stat()
    except OSError as exc:
        raise FetchError(f"test-only file URL is unreadable: {exc}") from None
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise FetchError("test-only file URL is not one regular single-linked file")
    if before.st_size > max_bytes:
        raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as stream:
            remaining = before.st_size
            while remaining:
                _remaining(deadline)
                chunk = stream.read(min(_TRANSFER_CHUNK_BYTES, remaining))
                if not chunk:
                    raise FetchError("test-only local archive was truncated")
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise FetchError("test-only archive copy made no progress")
                    view = view[written:]
                remaining -= len(chunk)
            if stream.read(1):
                raise FetchError("test-only local archive grew during copy")
        after = source.stat()
        stable_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
            "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise FetchError("test-only local archive changed during copy")
        _remaining(deadline)
        os.fsync(fd)
    finally:
        os.close(fd)


def _member_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or any(ord(char) < 32 or ord(char) == 127 for char in raw)
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw.rstrip("/")
    ):
        raise FetchError(f"archive member escapes destination: {raw!r}")
    if any(part in _SKIP_DIRS for part in path.parts):
        raise FetchError(f"archive member is excluded from bundle identity: {raw!r}")
    if path.suffix in _SKIP_SUFFIXES or path.name.startswith("._"):
        raise FetchError(f"archive member is excluded from bundle identity: {raw!r}")
    return path


def _safe_extract(archive: Path, destination: Path) -> None:
    budget = MAX_EXTRACTED_BYTES
    seen: dict[PurePosixPath, str] = {}
    directories: set[PurePosixPath] = set()
    file_paths: set[PurePosixPath] = set()
    try:
        with tarfile.open(archive, "r:*") as tar:
            for count, member in enumerate(tar, start=1):
                if count > MAX_MEMBERS:
                    raise FetchError(f"archive has more than {MAX_MEMBERS} members")
                name = _member_path(member.name)
                if name in seen:
                    raise FetchError(f"archive contains duplicate member path: {member.name!r}")
                ancestors = tuple(name.parents)[:-1]
                blocking = next((parent for parent in ancestors if seen.get(parent) == "file"), None)
                if blocking is not None:
                    raise FetchError(
                        f"archive path conflicts with earlier file {blocking.as_posix()!r}"
                    )
                target = destination.joinpath(*name.parts)
                if member.isdir():
                    seen[name] = "dir"
                    directories.add(name)
                    target.mkdir(mode=0o700, parents=True, exist_ok=True)
                    continue
                if not member.isreg():
                    raise FetchError(f"archive member is not a regular file: {member.name!r}")
                if any(prior != name and name in prior.parents for prior in seen):
                    raise FetchError(
                        f"archive file conflicts with an earlier child: {member.name!r}"
                    )
                if type(member.size) is not int or member.size < 0:
                    raise FetchError(f"archive member has an invalid size: {member.name!r}")
                budget -= member.size
                if budget < 0:
                    raise FetchError(f"extracted size exceeds {MAX_EXTRACTED_BYTES} bytes")
                seen[name] = "file"
                file_paths.add(name)
                source = tar.extractfile(member)
                if source is None:
                    raise FetchError(f"unreadable archive member: {member.name!r}")
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(target, flags, 0o600)
                try:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(_TRANSFER_CHUNK_BYTES, remaining))
                        if not chunk:
                            raise FetchError(f"archive member was truncated: {member.name!r}")
                        view = memoryview(chunk)
                        while view:
                            written = os.write(fd, view)
                            if written <= 0:
                                raise FetchError("archive extraction write made no progress")
                            view = view[written:]
                        remaining -= len(chunk)
                    if source.read(1):
                        raise FetchError(
                            f"archive member exceeded its declared size: {member.name!r}"
                        )
                    os.fsync(fd)
                finally:
                    source.close()
                    os.close(fd)
        if not file_paths:
            raise FetchError("archive contains no identity-bearing files")
        required_directories = {
            parent
            for path in file_paths
            for parent in tuple(path.parents)[:-1]
        }
        extra = directories - required_directories
        if extra:
            raise FetchError(
                "archive contains identity-excluded empty directories: "
                + ", ".join(sorted(path.as_posix() for path in extra)[:16])
            )
    except FetchError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise FetchError(f"corrupt archive: {exc}") from None


def _bundle_root(extract_dir: Path) -> Path:
    entries = list(extract_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def _private_destination(root: str | Path) -> Path:
    path = Path(root)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise FetchError(
            "bundle destination root must be owner-private and validator-owned"
        )
    return path


def _validate_private_tree(root: Path) -> None:
    root_info = root.lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.geteuid()
        or stat.S_IMODE(root_info.st_mode) != 0o700
    ):
        raise FetchError("cached bundle root is not owner-private")
    directories: set[PurePosixPath] = set()
    files: set[PurePosixPath] = set()
    for current, child_directories, child_files in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        child_directories.sort()
        child_files.sort()
        for name in child_directories:
            path = current_path / name
            info = path.lstat()
            relative = PurePosixPath(path.relative_to(root).as_posix())
            _member_path(relative.as_posix())
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise FetchError("cached bundle contains an unsafe directory")
            directories.add(relative)
        for name in child_files:
            path = current_path / name
            info = path.lstat()
            relative = PurePosixPath(path.relative_to(root).as_posix())
            _member_path(relative.as_posix())
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise FetchError("cached bundle contains an unsafe regular file")
            files.add(relative)
    required = {
        parent for path in files for parent in tuple(path.parents)[:-1]
    }
    if not files or directories != required:
        raise FetchError("cached bundle contains identity-excluded directory state")


def _fetch_bundle(
    url: str,
    expected_hash: str,
    dest_root: str | Path,
    *,
    test_only_local_file: bool,
    transfer_timeout_s: float,
) -> Path:
    if not isinstance(expected_hash, str) or _HASH_RE.fullmatch(expected_hash) is None:
        raise FetchError("expected bundle hash must be 64 lowercase hex characters")
    if (
        isinstance(transfer_timeout_s, bool)
        or not isinstance(transfer_timeout_s, (int, float))
        or not 0 < float(transfer_timeout_s) <= 600
    ):
        raise FetchError("bundle transfer timeout must be in (0, 600] seconds")
    deadline = time.monotonic() + float(transfer_timeout_s)
    root = _private_destination(dest_root)
    final = root / expected_hash
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_dir():
            raise FetchError("cached bundle has an unsafe shape")
        _validate_private_tree(final)
        actual = content_hash(final)
        if actual == expected_hash:
            return final
        raise FetchError(
            f"cached bundle at {final} re-hashes to {actual[:16]}…; remove it manually"
        )

    with tempfile.TemporaryDirectory(dir=root, prefix=".fetch.") as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "bundle.tar.gz"
        if test_only_local_file:
            _copy_local_archive_for_testing(
                url, archive, MAX_ARCHIVE_BYTES, deadline=deadline
            )
        else:
            _download_https(url, archive, MAX_ARCHIVE_BYTES, deadline=deadline)
        extract_dir = temporary_path / "extract"
        extract_dir.mkdir(mode=0o700)
        _safe_extract(archive, extract_dir)
        proposal = _bundle_root(extract_dir)
        # ``mkdir(parents=True)`` may create the archive's single wrapper
        # directory with the process umask rather than the leaf's explicit
        # mode. Modes are not bundle identity, so normalize the trusted cache
        # root before validation/publication.
        proposal.chmod(0o700)
        try:
            actual = content_hash(proposal)
        except (OSError, ValueError, NotADirectoryError) as exc:
            raise FetchError(f"extracted archive is not a bundle: {exc}") from None
        if actual != expected_hash:
            raise FetchError(
                f"content hash mismatch: committed {expected_hash[:16]}…, "
                f"fetched {actual[:16]}…"
            )
        try:
            proposal.rename(final)
        except OSError as exc:
            raise FetchError(f"bundle cache publication failed: {exc}") from None
        _validate_private_tree(final)
    logger.info("fetched bundle %s… from %s", expected_hash[:16], url)
    return final


def fetch_bundle(
    url: str,
    expected_hash: str,
    dest_root: str | Path,
    *,
    transfer_timeout_s: float = FETCH_TIMEOUT_S,
) -> Path:
    """Fetch one production proposal over pinned HTTPS."""

    return _fetch_bundle(
        url,
        expected_hash,
        dest_root,
        test_only_local_file=False,
        transfer_timeout_s=transfer_timeout_s,
    )


def fetch_bundle_from_local_file_for_testing(
    url: str,
    expected_hash: str,
    dest_root: str | Path,
    *,
    transfer_timeout_s: float = FETCH_TIMEOUT_S,
) -> Path:
    """Fetch through an explicit, hermetic-only local-file transport."""

    return _fetch_bundle(
        url,
        expected_hash,
        dest_root,
        test_only_local_file=True,
        transfer_timeout_s=transfer_timeout_s,
    )
