"""CPU tests for resident backend pieces: staging helper + argv/mount policy."""

from __future__ import annotations

import pytest

from optima.bundle_hash import content_hash
from optima.eval.oci_backend import OCIBackendError, stage_swap_bundle


class TestStageSwapBundle:
    def _source(self, tmp_path):
        source = tmp_path / "worker-tree"
        source.mkdir()
        (source / "manifest.toml").write_text("[bundle]\nname='x'\n")
        kernels = source / "kernels"
        kernels.mkdir()
        (kernels / "k.py").write_text("def k(): pass\n")
        return source

    def test_stage_publishes_content_addressed_tree(self, tmp_path) -> None:
        root = tmp_path / "intake"
        root.mkdir()
        source = self._source(tmp_path)
        digest = stage_swap_bundle(root, source)
        assert digest == content_hash(source)
        staged = root / digest
        assert staged.is_dir()
        assert content_hash(staged) == digest

    def test_stage_is_idempotent(self, tmp_path) -> None:
        root = tmp_path / "intake"
        root.mkdir()
        source = self._source(tmp_path)
        first = stage_swap_bundle(root, source)
        second = stage_swap_bundle(root, source)
        assert first == second

    def test_stage_rejects_expected_digest_mismatch(self, tmp_path) -> None:
        root = tmp_path / "intake"
        root.mkdir()
        source = self._source(tmp_path)
        with pytest.raises(OCIBackendError, match="committed digest"):
            stage_swap_bundle(root, source, expected_digest="9" * 64)

    def test_stage_detects_tampered_destination(self, tmp_path) -> None:
        root = tmp_path / "intake"
        root.mkdir()
        source = self._source(tmp_path)
        digest = stage_swap_bundle(root, source)
        (root / digest / "kernels" / "k.py").write_text("def k(): return 1\n")
        with pytest.raises(OCIBackendError, match="different bytes"):
            stage_swap_bundle(root, source)

    def test_no_partial_publication_on_failure(self, tmp_path) -> None:
        root = tmp_path / "intake"
        root.mkdir()
        source = self._source(tmp_path)
        with pytest.raises(OCIBackendError):
            stage_swap_bundle(root, source, expected_digest="8" * 64)
        leftovers = [p.name for p in root.iterdir()]
        assert leftovers == []


class TestResidentArgvPolicy:
    def test_swap_intake_requires_resident_protocol(self) -> None:
        # The full argv builder needs a resolved launch; the protocol/mount
        # pairing rule is testable through its guard clause alone.
        from optima.eval.oci_backend import build_runtime_argv

        with pytest.raises(OCIBackendError, match="not registered"):
            build_runtime_argv(
                lease=None,  # type: ignore[arg-type]
                resolved=None,  # type: ignore[arg-type]
                preflight=None,  # type: ignore[arg-type]
                model_root=None,  # type: ignore[arg-type]
                publication=None,  # type: ignore[arg-type]
                cache_root=None,  # type: ignore[arg-type]
                seccomp_path=None,  # type: ignore[arg-type]
                runtime=None,  # type: ignore[arg-type]
                session_protocol="bogus",
            )

    def test_ordinary_protocol_rejects_swap_root(self, tmp_path) -> None:
        from optima.eval.oci_backend import build_runtime_argv

        with pytest.raises(OCIBackendError, match="exactly for resident"):
            build_runtime_argv(
                lease=None,  # type: ignore[arg-type]
                resolved=None,  # type: ignore[arg-type]
                preflight=None,  # type: ignore[arg-type]
                model_root=None,  # type: ignore[arg-type]
                publication=None,  # type: ignore[arg-type]
                cache_root=None,  # type: ignore[arg-type]
                seccomp_path=None,  # type: ignore[arg-type]
                runtime=None,  # type: ignore[arg-type]
                session_protocol="ordinary",
                swap_intake_root=tmp_path,
            )

    def test_resident_protocol_requires_swap_root(self) -> None:
        from optima.eval.oci_backend import build_runtime_argv

        with pytest.raises(OCIBackendError, match="exactly for resident"):
            build_runtime_argv(
                lease=None,  # type: ignore[arg-type]
                resolved=None,  # type: ignore[arg-type]
                preflight=None,  # type: ignore[arg-type]
                model_root=None,  # type: ignore[arg-type]
                publication=None,  # type: ignore[arg-type]
                cache_root=None,  # type: ignore[arg-type]
                seccomp_path=None,  # type: ignore[arg-type]
                runtime=None,  # type: ignore[arg-type]
                session_protocol="resident",
            )


class TestEngineKwargAdditions:
    def test_watchdog_timeout_accepted(self) -> None:
        from optima.eval.oci_session_protocol import _validate_engine_kwargs

        result = _validate_engine_kwargs({"watchdog_timeout": 1800})
        assert result == {"watchdog_timeout": 1800}

    def test_cuda_graph_bs_accepted_sorted(self) -> None:
        from optima.eval.oci_session_protocol import _validate_engine_kwargs

        result = _validate_engine_kwargs({"cuda_graph_bs": [1, 8, 256]})
        assert result == {"cuda_graph_bs": [1, 8, 256]}

    def test_cuda_graph_bs_rejects_unsorted_or_duplicates(self) -> None:
        from optima.eval.oci_session_protocol import (
            SessionProtocolError,
            _validate_engine_kwargs,
        )

        for bad in ([256, 8], [8, 8], [0], [], "256"):
            with pytest.raises(SessionProtocolError):
                _validate_engine_kwargs({"cuda_graph_bs": bad})
