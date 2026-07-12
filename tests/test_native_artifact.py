from __future__ import annotations

import ctypes
import dataclasses
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from optima.eval import native_artifact as module
from optima.eval.native_artifact import (
    NativeArtifactCollisionError,
    NativeArtifactError,
    NativeArtifactFile,
    NativeArtifactLimits,
    NativeArtifactPublication,
    NativeArtifactRaceError,
    publish_native_artifact,
    reopen_native_artifact,
)
from optima.stack_identity import canonical_json_bytes


BUILD = "ab" * 32
OTHER_BUILD = "cd" * 32


def _source(tmp_path: Path, data: bytes = b"native-bytes") -> Path:
    root = tmp_path / "build"
    (root / "cuda" / "deps").mkdir(parents=True)
    (root / "cuda" / "kernel.so").write_bytes(data)
    (root / "cuda" / "deps" / "kernel.d").write_text("kernel.so: kernel.cu\n")
    return root


def _publish(tmp_path: Path, data: bytes = b"native-bytes"):
    source = _source(tmp_path, data)
    return source, publish_native_artifact(
        source, tmp_path / "published", build_spec_digest=BUILD
    )


def _manifest(publication: NativeArtifactPublication) -> Path:
    return publication.root / ".optima-native-artifact.json"


def _make_writable(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts)):
        if child.is_dir() and not child.is_symlink():
            child.chmod(0o755)
        elif child.is_file() and not child.is_symlink():
            child.chmod(0o644)
    path.chmod(0o755)


def test_publish_is_build_addressed_canonical_immutable_and_reopenable(tmp_path):
    source, published = _publish(tmp_path)

    assert published.root == (tmp_path / "published").resolve() / BUILD[:2] / BUILD
    assert published.path == published.root
    assert published.tree_digest == published.publication_digest
    assert published.build_spec_digest == BUILD
    assert not published.reused
    assert published.directories == ("cuda", "cuda/deps")
    assert [row.path for row in published.files] == [
        "cuda/deps/kernel.d",
        "cuda/kernel.so",
    ]
    assert stat.S_IMODE(published.root.stat().st_mode) == 0o555
    for path in published.root.rglob("*"):
        expected = 0o555 if path.is_dir() else 0o444
        assert stat.S_IMODE(path.stat().st_mode) == expected

    raw = _manifest(published).read_bytes()
    decoded = json.loads(raw)
    assert raw == canonical_json_bytes(decoded) + b"\n"
    assert decoded == published.identity_dict()
    assert str(source) not in raw.decode()
    assert str(tmp_path / "published") not in raw.decode()

    reopened = reopen_native_artifact(
        published.root,
        expected_build_spec_digest=BUILD,
        expected_publication_digest=published.publication_digest,
    )
    assert reopened == published


def test_real_sglang_components_are_canonical_and_reopenable(tmp_path):
    source = tmp_path / "build"
    package = source / "sglang" / "srt"
    package.mkdir(parents=True)
    (source / "sglang" / "__init__.py").write_text("# package\n")
    (package / "__init__.py").write_text("# subpackage\n")
    (package / "__main__.py").write_text("# module entry\n")
    (package / "_version.py").write_text("VERSION = 1\n")
    (package / "_core.cpython-312-x86_64-linux-gnu.so").write_bytes(b"elf")
    tuning = "E=128,N=384,device_name=NVIDIA_B200,block_shape=[128, 128].json"
    (package / tuning).write_text("{}\n")
    (package / "scheduler.py").write_text("VALUE = 1\n")

    published = publish_native_artifact(
        source, tmp_path / "published", build_spec_digest=BUILD
    )
    assert tuple(row.path for row in published.files) == (
        "sglang/__init__.py",
        f"sglang/srt/{tuning}",
        "sglang/srt/__init__.py",
        "sglang/srt/__main__.py",
        "sglang/srt/_core.cpython-312-x86_64-linux-gnu.so",
        "sglang/srt/_version.py",
        "sglang/srt/scheduler.py",
    )
    assert reopen_native_artifact(
        published.root,
        expected_build_spec_digest=BUILD,
        expected_publication_digest=published.publication_digest,
    ) == published


@pytest.mark.parametrize(
    "component",
    (
        "_private",
        "_.py",
        "__init__.pyc",
        "__pycache__",
        ".hidden.py",
        "bad$name.json",
        "bad;name.json",
        "bad(name).json",
    ),
)
def test_non_module_private_components_remain_forbidden(tmp_path, component):
    source = tmp_path / "build"
    source.mkdir()
    (source / component).write_bytes(b"x")
    with pytest.raises(NativeArtifactError, match="component is unsafe"):
        publish_native_artifact(
            source, tmp_path / "published", build_spec_digest=BUILD
        )


def test_exact_existing_publication_is_reused_but_different_bytes_collide(tmp_path):
    source, first = _publish(tmp_path)
    second = publish_native_artifact(
        source, tmp_path / "published", build_spec_digest=BUILD
    )
    assert second.reused
    assert dataclasses.replace(second, reused=False) == first
    assert not list(first.root.parent.glob(".stage-*"))

    (source / "cuda" / "kernel.so").write_bytes(b"different")
    with pytest.raises(NativeArtifactCollisionError, match="differs from expected"):
        publish_native_artifact(
            source, tmp_path / "published", build_spec_digest=BUILD
        )
    assert not list(first.root.parent.glob(".stage-*"))


@pytest.mark.skipif(sys.platform != "linux", reason="production publication uses Linux renameat2")
def test_lease_owned_work_root_holds_copy_stage_and_is_empty_after_publish(
    tmp_path, monkeypatch
):
    source = _source(tmp_path)
    work = tmp_path / "lease" / "publication-work"
    work.mkdir(parents=True, mode=0o700)
    observed_parent = []
    original = module._create_stage

    def observe(parent_fd, build_spec_digest):
        observed_parent.append(os.fstat(parent_fd))
        return original(parent_fd, build_spec_digest)

    monkeypatch.setattr(module, "_create_stage", observe)
    published = publish_native_artifact(
        source,
        tmp_path / "published",
        build_spec_digest=BUILD,
        work_root=work,
    )

    assert observed_parent and observed_parent[0].st_ino == work.stat().st_ino
    assert not list(work.iterdir())
    assert not list(published.root.parent.glob(".stage-*"))


def test_lease_owned_work_root_must_share_publication_filesystem(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module.sys, "platform", "linux")
    source = _source(tmp_path)
    work = tmp_path / "lease-work"
    work.mkdir(mode=0o700)
    original = module._canonical_existing_directory

    def different_device(path, *, name, writable, expected_owner_uid=module._CURRENT_OWNER):
        canonical, fd, info = original(
            path,
            name=name,
            writable=writable,
            expected_owner_uid=expected_owner_uid,
        )
        if name == "work_root":
            values = list(info)
            values[stat.ST_DEV] += 1
            info = os.stat_result(values)
        return canonical, fd, info

    monkeypatch.setattr(module, "_canonical_existing_directory", different_device)
    with pytest.raises(NativeArtifactError, match="share a filesystem"):
        publish_native_artifact(
            source,
            tmp_path / "published",
            build_spec_digest=BUILD,
            work_root=work,
        )
    assert not list(work.iterdir())


def test_host_publication_and_reopen_do_not_execute_constructor_bearing_elf(
    tmp_path,
):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("constructor nonexecution fixture needs a host C compiler")
    sentinel = tmp_path / "constructor-fired"
    source = tmp_path / "constructor.c"
    source.write_text(
        "#include <stdio.h>\n"
        "__attribute__((constructor)) static void fire(void) {\n"
        f"  FILE *f = fopen({json.dumps(str(sentinel))}, \"wb\");\n"
        "  if (f != NULL) { fputs(\"fired\", f); fclose(f); }\n"
        "}\n"
    )
    stage = tmp_path / "native-stage"
    stage.mkdir()
    library = stage / "constructor.so"
    command = [compiler]
    if sys.platform == "darwin":
        command.append("-dynamiclib")
    else:
        command.extend(("-shared", "-fPIC"))
    subprocess.run(
        [*command, str(source), "-o", str(library)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    publication = publish_native_artifact(
        stage, tmp_path / "published", build_spec_digest=BUILD
    )
    assert not sentinel.exists()
    reopened = reopen_native_artifact(
        publication.root,
        expected_build_spec_digest=BUILD,
        expected_publication_digest=publication.publication_digest,
    )
    assert not sentinel.exists()

    ctypes.CDLL(str(reopened.root / "constructor.so"))
    assert sentinel.read_text() == "fired", "fixture must prove constructor liveness"


def test_publication_digest_is_path_and_source_metadata_independent(tmp_path):
    _, left = _publish(tmp_path / "left")
    right_source = _source(tmp_path / "right")
    for path in right_source.rglob("*"):
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    right = publish_native_artifact(
        right_source, tmp_path / "other-publication", build_spec_digest=BUILD
    )
    assert left.publication_digest == right.publication_digest
    assert left.directories == right.directories
    assert left.files == right.files

def test_reopen_requires_build_address_and_can_require_external_tree_digest(tmp_path):
    _, published = _publish(tmp_path)
    with pytest.raises(NativeArtifactError, match="path is not derived"):
        reopen_native_artifact(
            published.root,
            expected_build_spec_digest=OTHER_BUILD,
        )
    with pytest.raises(NativeArtifactCollisionError, match="differs from expected"):
        reopen_native_artifact(
            published.root,
            expected_build_spec_digest=BUILD,
            expected_publication_digest="00" * 32,
        )
    assert reopen_native_artifact(
        published.root, expected_build_spec_digest=BUILD
    ).publication_digest == published.publication_digest


def test_nonroot_readonly_runtime_can_reopen_root_owned_publication(
    tmp_path, monkeypatch
):
    _, published = _publish(tmp_path)
    owner = published.root.stat().st_uid
    monkeypatch.setattr(module.os, "geteuid", lambda: owner + 10_000)
    with pytest.raises(NativeArtifactError, match="owned"):
        reopen_native_artifact(
            published.root,
            expected_build_spec_digest=BUILD,
            expected_publication_digest=published.publication_digest,
        )
    reopened = reopen_native_artifact(
        published.root,
        expected_build_spec_digest=BUILD,
        expected_publication_digest=published.publication_digest,
        expected_owner_uid=None,
    )
    assert reopened.publication_digest == published.publication_digest


@pytest.mark.parametrize("value", ["", "A" * 64, "0" * 63, "g" * 64, True, None])
def test_build_digest_is_strict(value, tmp_path):
    source = _source(tmp_path)
    with pytest.raises(NativeArtifactError, match="lowercase 64-hex"):
        publish_native_artifact(source, tmp_path / "published", build_spec_digest=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_files": 0},
        {"max_files": True},
        {"max_file_bytes": 2, "max_total_bytes": 1},
        {"max_depth": -1},
    ],
)
def test_limits_are_strict(kwargs):
    with pytest.raises(NativeArtifactError):
        NativeArtifactLimits(**kwargs)


def test_public_result_types_are_frozen_and_validate_identity(tmp_path):
    _, published = _publish(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        published.reused = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        published.files[0].size = 4
    with pytest.raises(NativeArtifactError, match="digest is inconsistent"):
        dataclasses.replace(published, publication_digest="00" * 32)
    with pytest.raises(NativeArtifactError, match="nonnegative"):
        NativeArtifactFile("x.so", "00" * 32, -1)


def test_publish_rejects_source_and_publication_root_overlap(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(NativeArtifactError, match="must not overlap"):
        publish_native_artifact(source, source, build_spec_digest=BUILD)
    with pytest.raises(NativeArtifactError, match="must not overlap"):
        publish_native_artifact(
            source, source / "published", build_spec_digest=BUILD
        )
    assert not (source / "published").exists()

    publication = tmp_path / "outer"
    nested_source = publication / "source"
    nested_source.mkdir(parents=True)
    (nested_source / "x.so").write_bytes(b"x")
    publication.chmod(0o700)
    with pytest.raises(NativeArtifactError, match="must not overlap"):
        publish_native_artifact(
            nested_source, publication, build_spec_digest=BUILD
        )


def test_root_symlinks_and_insecure_publication_root_reject(tmp_path):
    source = _source(tmp_path)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(NativeArtifactError, match="must not be a symlink"):
        publish_native_artifact(
            source_link, tmp_path / "published", build_spec_digest=BUILD
        )

    actual_publication = tmp_path / "actual-publication"
    actual_publication.mkdir(mode=0o700)
    publication_link = tmp_path / "publication-link"
    publication_link.symlink_to(actual_publication, target_is_directory=True)
    with pytest.raises(NativeArtifactError, match="must not be a symlink"):
        publish_native_artifact(source, publication_link, build_spec_digest=BUILD)

    insecure = tmp_path / "insecure"
    insecure.mkdir(mode=0o777)
    insecure.chmod(0o777)
    with pytest.raises(NativeArtifactError, match="group/world writable"):
        publish_native_artifact(source, insecure, build_spec_digest=BUILD)


def test_reopen_rejects_symlinked_publication_root(tmp_path):
    _, published = _publish(tmp_path)
    link = published.root.parent / OTHER_BUILD
    link.symlink_to(published.root, target_is_directory=True)
    # The name is intentionally not the expected build address.
    with pytest.raises(NativeArtifactError):
        reopen_native_artifact(link, expected_build_spec_digest=OTHER_BUILD)


def test_source_symlink_hardlink_fifo_and_socket_reject(tmp_path):
    factories = []

    def symlink(root):
        target = tmp_path / "target"
        target.write_bytes(b"x")
        (root / "x.so").symlink_to(target)

    factories.append(symlink)

    def hardlink(root):
        path = root / "x.so"
        path.write_bytes(b"x")
        os.link(path, tmp_path / "outside-link")

    factories.append(hardlink)

    if hasattr(os, "mkfifo"):
        factories.append(lambda root: os.mkfifo(root / "x.so"))

    for index, factory in enumerate(factories):
        case = tmp_path / f"case-{index}"
        case.mkdir()
        factory(case)
        with pytest.raises(NativeArtifactError):
            publish_native_artifact(
                case, tmp_path / f"publication-{index}", build_spec_digest=BUILD
            )

    socket_root = Path(tempfile.mkdtemp(prefix="optima-na-", dir="/tmp"))
    handle = socket.socket(socket.AF_UNIX)
    try:
        handle.bind(str(socket_root / "x.so"))
        with pytest.raises(NativeArtifactError, match="non-regular"):
            publish_native_artifact(
                socket_root, tmp_path / "socket-publication", build_spec_digest=BUILD
            )
    finally:
        handle.close()
        shutil.rmtree(socket_root)


def test_source_sparse_file_rejects_when_filesystem_reports_holes(tmp_path):
    source = tmp_path / "build"
    source.mkdir()
    sparse = source / "kernel.so"
    with sparse.open("wb") as handle:
        handle.seek(4 << 20)
        handle.write(b"x")
    info = sparse.stat()
    if not hasattr(info, "st_blocks") or info.st_blocks * 512 >= info.st_size:
        pytest.skip("filesystem does not expose sparse allocation")
    with pytest.raises(NativeArtifactError, match="sparse"):
        publish_native_artifact(
            source, tmp_path / "published", build_spec_digest=BUILD
        )


@pytest.mark.parametrize(
    ("limits", "match"),
    [
        (NativeArtifactLimits(max_file_bytes=4, max_total_bytes=4), "file exceeds"),
        (NativeArtifactLimits(max_files=1), "file-count"),
        (NativeArtifactLimits(max_file_bytes=24, max_total_bytes=24), "total-byte"),
        (NativeArtifactLimits(max_directories=1), "directory-count"),
        (NativeArtifactLimits(max_depth=1), "depth"),
        (NativeArtifactLimits(max_path_bytes=8), "byte bound"),
    ],
)
def test_source_resource_bounds_reject(tmp_path, limits, match):
    source = _source(tmp_path, b"12345")
    with pytest.raises(NativeArtifactError, match=match):
        publish_native_artifact(
            source,
            tmp_path / "published",
            build_spec_digest=BUILD,
            limits=limits,
        )


def test_empty_tree_empty_directory_hidden_and_reserved_names_reject(tmp_path):
    cases: list[tuple[str, callable]] = [
        ("empty", lambda root: None),
        ("empty-dir", lambda root: (root / "unused").mkdir()),
        ("hidden", lambda root: (root / ".cache").write_bytes(b"x")),
        (
            "reserved",
            lambda root: (root / ".optima-native-artifact.json").write_bytes(b"x"),
        ),
    ]
    for name, populate in cases:
        source = tmp_path / name
        source.mkdir()
        populate(source)
        with pytest.raises(NativeArtifactError):
            publish_native_artifact(
                source, tmp_path / f"pub-{name}", build_spec_digest=BUILD
            )


def test_staged_name_collision_fails_closed_and_preserves_peer(tmp_path, monkeypatch):
    source = _source(tmp_path)
    publication = tmp_path / "published"
    shard = publication / BUILD[:2]
    shard.mkdir(parents=True, mode=0o700)
    token = "11" * 16
    peer = shard / f".stage-{BUILD[:16]}-{token}"
    peer.mkdir()
    marker = peer / "peer"
    marker.write_bytes(b"untouched")
    monkeypatch.setattr(module.secrets, "token_hex", lambda size: token)

    with pytest.raises(NativeArtifactRaceError, match="stage name"):
        publish_native_artifact(source, publication, build_spec_digest=BUILD)
    assert marker.read_bytes() == b"untouched"


def test_destination_appearance_during_staging_fails_without_replace(tmp_path, monkeypatch):
    source = _source(tmp_path)
    publication = tmp_path / "published"
    original = module._entry_exists
    calls = 0

    def race(parent_fd, name):
        nonlocal calls
        calls += 1
        if calls == 2:
            os.mkdir(name, 0o555, dir_fd=parent_fd)
        return original(parent_fd, name)

    monkeypatch.setattr(module, "_entry_exists", race)
    with pytest.raises(NativeArtifactRaceError, match="appeared during staging"):
        publish_native_artifact(source, publication, build_spec_digest=BUILD)
    assert (publication / BUILD[:2] / BUILD).is_dir()
    assert not list((publication / BUILD[:2]).glob(".stage-*"))


def test_existing_publication_replacement_race_rejects(tmp_path, monkeypatch):
    source, published = _publish(tmp_path)
    original = module._copy_stage

    def race(*args, **kwargs):
        result = original(*args, **kwargs)
        published.root.chmod(0o755)
        published.root.chmod(0o555)
        return result

    monkeypatch.setattr(module, "_copy_stage", race)
    with pytest.raises(NativeArtifactRaceError, match="existing native artifact changed"):
        publish_native_artifact(
            source, tmp_path / "published", build_spec_digest=BUILD
        )
    assert not list(published.root.parent.glob(".stage-*"))


def test_source_mutation_before_file_open_is_detected_and_no_publication_remains(
    tmp_path, monkeypatch
):
    source = _source(tmp_path)
    victim = source / "cuda" / "kernel.so"
    original = module._copy_regular
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        relative = args[3]
        if relative == "cuda/kernel.so" and not changed:
            changed = True
            victim.write_bytes(b"mutated")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_copy_regular", mutate)
    with pytest.raises(NativeArtifactRaceError, match="changed while opening"):
        publish_native_artifact(
            source, tmp_path / "published", build_spec_digest=BUILD
        )
    shard = tmp_path / "published" / BUILD[:2]
    assert not (shard / BUILD).exists()
    assert not list(shard.glob(".stage-*"))


def test_file_content_tamper_is_detected(tmp_path):
    _, published = _publish(tmp_path)
    victim = published.root / "cuda" / "kernel.so"
    victim.chmod(0o644)
    victim.write_bytes(b"tampered!!!")
    victim.chmod(0o444)
    with pytest.raises(NativeArtifactError, match="inventory differs"):
        reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)


def test_manifest_noncanonical_duplicate_extra_and_wrong_identity_reject(tmp_path):
    mutators = []

    def noncanonical(raw):
        return json.dumps(json.loads(raw), indent=2).encode() + b"\n"

    mutators.append(noncanonical)
    mutators.append(lambda raw: raw.replace(b'{"build_spec_digest"', b'{"extra":1,"build_spec_digest"', 1))
    mutators.append(
        lambda raw: raw.replace(
            b'{"build_spec_digest"',
            b'{"build_spec_digest":"' + BUILD.encode() + b'","build_spec_digest"',
            1,
        )
    )
    mutators.append(lambda raw: raw.replace(BUILD.encode(), OTHER_BUILD.encode(), 1))

    for index, mutate in enumerate(mutators):
        case = tmp_path / f"case-{index}"
        _, published = _publish(case)
        manifest = _manifest(published)
        raw = manifest.read_bytes()
        manifest.chmod(0o644)
        manifest.write_bytes(mutate(raw))
        manifest.chmod(0o444)
        with pytest.raises(NativeArtifactError):
            reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)


def test_extra_file_directory_symlink_and_writable_modes_reject(tmp_path):
    # Extra regular file.
    _, published = _publish(tmp_path / "extra-file")
    published.root.chmod(0o755)
    extra = published.root / "extra.so"
    extra.write_bytes(b"x")
    extra.chmod(0o444)
    published.root.chmod(0o555)
    with pytest.raises(NativeArtifactError, match="inventory differs"):
        reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)

    # Extra empty directory.
    _, published = _publish(tmp_path / "extra-directory")
    published.root.chmod(0o755)
    extra_dir = published.root / "extra"
    extra_dir.mkdir(mode=0o555)
    published.root.chmod(0o555)
    with pytest.raises(NativeArtifactError):
        reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)

    # Symlink entry.
    _, published = _publish(tmp_path / "symlink")
    published.root.chmod(0o755)
    (published.root / "escape").symlink_to("cuda/kernel.so")
    published.root.chmod(0o555)
    with pytest.raises(NativeArtifactError):
        reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)

    # Writable payload and root independently reject.
    _, published = _publish(tmp_path / "writable-file")
    (published.root / "cuda" / "kernel.so").chmod(0o644)
    with pytest.raises(NativeArtifactError, match="unsafe shape"):
        reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)

    _, published = _publish(tmp_path / "writable-root")
    published.root.chmod(0o755)
    with pytest.raises(NativeArtifactError, match="mode 0555"):
        reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)


def test_hardlinked_and_sparse_published_payload_reject(tmp_path):
    _, published = _publish(tmp_path / "hardlink")
    victim = published.root / "cuda" / "kernel.so"
    victim.parent.chmod(0o755)
    os.link(victim, tmp_path / "outside-hardlink")
    victim.parent.chmod(0o555)
    with pytest.raises(NativeArtifactError, match="unsafe shape"):
        reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)

    _, published = _publish(tmp_path / "sparse")
    victim = published.root / "cuda" / "kernel.so"
    victim.chmod(0o644)
    with victim.open("wb") as handle:
        handle.seek(4 << 20)
        handle.write(b"x")
    victim.chmod(0o444)
    info = victim.stat()
    if hasattr(info, "st_blocks") and info.st_blocks * 512 < info.st_size:
        with pytest.raises(NativeArtifactError, match="unsafe shape"):
            reopen_native_artifact(published.root, expected_build_spec_digest=BUILD)


def test_manifest_and_inventory_oversize_reject_on_reopen(tmp_path):
    _, published = _publish(tmp_path)
    with pytest.raises(NativeArtifactError):
        reopen_native_artifact(
            published.root,
            expected_build_spec_digest=BUILD,
            limits=NativeArtifactLimits(max_manifest_bytes=8),
        )
    with pytest.raises(NativeArtifactError):
        reopen_native_artifact(
            published.root,
            expected_build_spec_digest=BUILD,
            limits=NativeArtifactLimits(max_file_bytes=4, max_total_bytes=4),
        )


def test_manifest_bound_rejects_before_publication(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(NativeArtifactError, match="manifest exceeds"):
        publish_native_artifact(
            source,
            tmp_path / "published",
            build_spec_digest=BUILD,
            limits=NativeArtifactLimits(max_manifest_bytes=8),
        )
    shard = tmp_path / "published" / BUILD[:2]
    assert not (shard / BUILD).exists()
    assert not list(shard.glob(".stage-*"))


def test_module_has_no_loader_or_framework_dependency():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import sglang" not in source
    assert "dlopen" not in source
    assert "ctypes.CDLL" in source  # used only for renameat2/renameatx_np
