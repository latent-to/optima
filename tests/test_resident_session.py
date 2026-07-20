"""CPU tests for the resident hot-swap protocol, worker verbs, and host session."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from optima.eval.oci_session_protocol import (
    SessionProtocolError,
    SwapRequest,
    swap_evidence_message,
    swap_request,
    validate_swap_evidence,
    validate_swap_request,
)

SESSION = "1" * 32
LAUNCH = "a" * 64
DIGEST = "b" * 64


def _swap(index: int = 0, generation: int = 1, digest: str | None = DIGEST) -> SwapRequest:
    return SwapRequest(
        SESSION,
        LAUNCH,
        f"{index + 2:032x}",
        f"{index + 3:032x}".replace("0", "9", 1),
        index,
        generation,
        digest,
    )


class TestSwapProtocol:
    def test_round_trip(self) -> None:
        message = swap_request(
            session_id=SESSION,
            launch_digest=LAUNCH,
            request_id="2" * 32,
            nonce="3" * 32,
            swap_index=0,
            generation=1,
            bundle_digest=DIGEST,
        )
        parsed = validate_swap_request(message)
        assert parsed.bundle_digest == DIGEST
        assert parsed.generation == 1

    def test_stock_swap_round_trip(self) -> None:
        request = _swap(digest=None)
        parsed = validate_swap_request(request.to_dict())
        assert parsed.bundle_digest is None

    def test_rejects_zero_generation(self) -> None:
        with pytest.raises(SessionProtocolError):
            _swap(generation=0)

    def test_rejects_malformed_digest(self) -> None:
        with pytest.raises(SessionProtocolError):
            _swap(digest="zz" * 32)

    def test_rejects_extra_fields(self) -> None:
        message = _swap().to_dict()
        message["extra"] = 1
        with pytest.raises(SessionProtocolError):
            validate_swap_request(message)

    def test_evidence_round_trip(self) -> None:
        request = _swap()
        message = swap_evidence_message(
            request=request, slots=("moe.fused_experts",), rank_count=4
        )
        slots = validate_swap_evidence(message, request=request, expected_rank_count=4)
        assert slots == ("moe.fused_experts",)

    def test_evidence_binds_rank_count(self) -> None:
        request = _swap()
        message = swap_evidence_message(
            request=request, slots=("moe.fused_experts",), rank_count=4
        )
        with pytest.raises(SessionProtocolError):
            validate_swap_evidence(message, request=request, expected_rank_count=8)

    def test_evidence_binds_generation(self) -> None:
        request = _swap()
        message = swap_evidence_message(
            request=request, slots=("moe.fused_experts",), rank_count=4
        )
        other = _swap(index=1, generation=2)
        with pytest.raises(SessionProtocolError):
            validate_swap_evidence(message, request=other, expected_rank_count=4)

    def test_stock_evidence_must_be_slotless(self) -> None:
        request = _swap(digest=None)
        with pytest.raises(SessionProtocolError):
            swap_evidence_message(
                request=request, slots=("moe.fused_experts",), rank_count=4
            )
        message = swap_evidence_message(request=request, slots=(), rank_count=4)
        assert validate_swap_evidence(
            message, request=request, expected_rank_count=4
        ) == ()

    def test_bundle_evidence_must_register_slots(self) -> None:
        request = _swap()
        with pytest.raises(SessionProtocolError):
            swap_evidence_message(request=request, slots=(), rank_count=4)

    def test_evidence_rejects_unsorted_slots(self) -> None:
        request = _swap()
        message = swap_evidence_message(
            request=request,
            slots=("a.one", "b.two"),
            rank_count=4,
        )
        message["slots"] = ["b.two", "a.one"]
        with pytest.raises(SessionProtocolError):
            validate_swap_evidence(message, request=request, expected_rank_count=4)


class TestWorkerSwapApplication:
    def _control(self, tmp_path: Path) -> str:
        control = tmp_path / "ctl"
        control.mkdir()
        return str(control)

    def _ack_all(
        self, control: str, generation: int, *, tp: int = 2, slots=("s.one",)
    ) -> None:
        for rank in range(tp):
            (Path(control) / f"ack.rank{rank}.json").write_text(
                json.dumps(
                    {"generation": generation, "ok": True, "slots": list(slots)}
                )
            )

    def test_stock_swap_writes_command_and_collects_acks(self, tmp_path) -> None:
        from optima.eval import oci_session_worker as worker

        control = self._control(tmp_path)
        request = _swap(digest=None)

        class Engine:
            def flush_cache(self_inner) -> bool:
                self._ack_all(control, request.generation, slots=())
                return True

        slots = worker._apply_resident_swap(
            Engine(), request, control_dir=control, tp_size=2
        )
        assert slots == ()
        command = json.loads((Path(control) / "command.json").read_text())
        assert command == {"bundle": None, "generation": 1}

    def test_bundle_swap_rehashes_and_rejects_mismatch(self, tmp_path, monkeypatch) -> None:
        from optima.eval import oci_session_worker as worker

        control = self._control(tmp_path)
        staged_root = tmp_path / "intake"
        staged = staged_root / DIGEST
        staged.mkdir(parents=True)
        (staged / "kernel.py").write_text("def k(): pass\n")
        monkeypatch.setattr(
            worker, "CONTAINER_SWAP_INTAKE_PATH", str(staged_root)
        )
        monkeypatch.setattr(worker, "_read_only_directory", lambda path: True)

        class Engine:
            def flush_cache(self_inner) -> bool:
                return True

        request = _swap()
        with pytest.raises(SessionProtocolError, match="differs from its committed"):
            worker._apply_resident_swap(
                Engine(), request, control_dir=control, tp_size=2
            )
        assert not (Path(control) / "command.json").exists()

    def test_bundle_swap_accepts_exact_hash(self, tmp_path, monkeypatch) -> None:
        from optima.bundle_hash import content_hash
        from optima.eval import oci_session_worker as worker

        staged_root = tmp_path / "intake"
        seed = staged_root / "seed"
        seed.mkdir(parents=True)
        (seed / "kernel.py").write_text("def k(): pass\n")
        digest = content_hash(seed)
        seed.rename(staged_root / digest)
        control = self._control(tmp_path)
        monkeypatch.setattr(
            worker, "CONTAINER_SWAP_INTAKE_PATH", str(staged_root)
        )
        monkeypatch.setattr(worker, "_read_only_directory", lambda path: True)
        request = _swap(digest=digest)

        class Engine:
            def flush_cache(self_inner) -> bool:
                self._ack_all(control, request.generation, slots=("s.one",))
                return True

        slots = worker._apply_resident_swap(
            Engine(), request, control_dir=control, tp_size=2
        )
        assert slots == ("s.one",)
        command = json.loads((Path(control) / "command.json").read_text())
        assert command["bundle"].endswith(digest)

    def test_rank_failure_fails_the_swap(self, tmp_path) -> None:
        from optima.eval import oci_session_worker as worker

        control = self._control(tmp_path)
        request = _swap(digest=None)

        class Engine:
            def flush_cache(self_inner) -> bool:
                (Path(control) / "ack.rank0.json").write_text(
                    json.dumps(
                        {"generation": 1, "ok": True, "slots": []}
                    )
                )
                (Path(control) / "ack.rank1.json").write_text(
                    json.dumps(
                        {"generation": 1, "ok": False, "error": "boom", "slots": []}
                    )
                )
                return True

        with pytest.raises(SessionProtocolError, match="failed on rank 1"):
            worker._apply_resident_swap(
                Engine(), request, control_dir=control, tp_size=2
            )

    def test_rank_slot_disagreement_fails_the_swap(self, tmp_path) -> None:
        from optima.eval import oci_session_worker as worker

        control = self._control(tmp_path)
        request = _swap(digest=None)

        class Engine:
            def flush_cache(self_inner) -> bool:
                (Path(control) / "ack.rank0.json").write_text(
                    json.dumps({"generation": 1, "ok": True, "slots": ["a"]})
                )
                (Path(control) / "ack.rank1.json").write_text(
                    json.dumps({"generation": 1, "ok": True, "slots": ["b"]})
                )
                return True

        with pytest.raises(SessionProtocolError, match="different slot sets"):
            worker._apply_resident_swap(
                Engine(), request, control_dir=control, tp_size=2
            )

    def test_missing_flush_api_fails_closed(self, tmp_path) -> None:
        from optima.eval import oci_session_worker as worker

        control = self._control(tmp_path)
        with pytest.raises(SessionProtocolError, match="flush_cache"):
            worker._apply_resident_swap(
                object(), _swap(digest=None), control_dir=control, tp_size=2
            )

    def test_stale_generation_acks_are_ignored(self, tmp_path, monkeypatch) -> None:
        from optima.eval import oci_session_worker as worker

        control = self._control(tmp_path)
        request = _swap(index=1, generation=2, digest=None)
        calls = {"n": 0}

        class Engine:
            def flush_cache(self_inner) -> bool:
                calls["n"] += 1
                if calls["n"] == 1:
                    self._ack_all(control, 1, slots=())  # stale generation
                else:
                    self._ack_all(control, 2, slots=())
                return True

        monkeypatch.setattr(worker, "_RESIDENT_FLUSH_RETRY_SECONDS", 0.0)
        monkeypatch.setattr(worker, "_RESIDENT_ACK_POLL_SECONDS", 0.0)
        slots = worker._apply_resident_swap(
            Engine(), request, control_dir=control, tp_size=2
        )
        assert slots == ()
        assert calls["n"] >= 2


class TestServeResidentLoop:
    def _frames(self, messages) -> bytes:
        from optima.eval.oci_session_protocol import (
            MAX_BATCH_REQUEST_BYTES,
            frame_message,
        )

        return b"".join(
            frame_message(message, max_bytes=MAX_BATCH_REQUEST_BYTES)
            for message in messages
        )

    def test_swap_then_batch_stream(self, tmp_path, monkeypatch) -> None:
        from optima.eval import oci_session_worker as worker
        from optima.eval.oci_session_protocol import (
            batch_request,
            parse_frame_bytes,
        )

        control = tmp_path / "ctl"
        control.mkdir()

        class Engine:
            def flush_cache(self_inner) -> bool:
                for rank in range(2):
                    (control / f"ack.rank{rank}.json").write_text(
                        json.dumps(
                            {"generation": 1, "ok": True, "slots": []}
                        )
                    )
                return True

            def generate(self_inner, **kwargs):
                prompts = kwargs["prompt"]
                tokens = kwargs["sampling_params"]["max_new_tokens"]
                width = kwargs["top_logprobs_num"]
                return [
                    {
                        "meta_info": {
                            "output_ids": list(range(tokens)),
                            "output_top_logprobs": [
                                [(-0.5 - column, column) for column in range(width)]
                                for _ in range(tokens)
                            ],
                        }
                    }
                    for _ in prompts
                ]

        swap = _swap(digest=None).to_dict()
        batch = batch_request(
            session_id=SESSION,
            launch_digest=LAUNCH,
            request_id="7" * 32,
            nonce="8" * 32,
            batch_index=0,
            prompts=("hello",),
            max_new_tokens=2,
            top_logprobs_num=2,
            temperature=1.0,
        )
        payload = self._frames([swap, batch])
        read_fd, write_fd = os.pipe()
        out_read_fd, out_write_fd = os.pipe()
        os.write(write_fd, payload)
        os.close(write_fd)
        try:
            with pytest.raises(SessionProtocolError):
                worker._serve_resident(
                    Engine(),
                    read_fd,
                    out_write_fd,
                    session_id=SESSION,
                    launch_digest=LAUNCH,
                    control_dir=str(control),
                    tp_size=2,
                )
            os.close(out_write_fd)
            produced = b""
            while True:
                chunk = os.read(out_read_fd, 1 << 20)
                if not chunk:
                    break
                produced += chunk
        finally:
            os.close(read_fd)
            os.close(out_read_fd)
        # First frame: swap evidence control frame.
        assert produced[:4] == b"OES1"
        size = int.from_bytes(produced[4:8], "big")
        evidence = parse_frame_bytes(
            produced[: 8 + size], max_bytes=1 << 16
        )
        assert evidence["type"] == "swap_evidence"
        assert evidence["generation"] == 1
        # Second frame: binary batch evidence.
        assert produced[8 + size : 12 + size] == b"OEE1"

    def test_replayed_swap_nonce_fails(self, tmp_path) -> None:
        from optima.eval import oci_session_worker as worker

        control = tmp_path / "ctl"
        control.mkdir()

        class Engine:
            def flush_cache(self_inner) -> bool:
                for rank in range(2):
                    (control / f"ack.rank{rank}.json").write_text(
                        json.dumps({"generation": 1, "ok": True, "slots": []})
                    )
                return True

        first = _swap(digest=None)
        replay = SwapRequest(
            SESSION,
            LAUNCH,
            first.request_id,
            first.nonce,
            1,
            2,
            None,
        )
        payload = self._frames([first.to_dict(), replay.to_dict()])
        read_fd, write_fd = os.pipe()
        out_read_fd, out_write_fd = os.pipe()
        os.write(write_fd, payload)
        os.close(write_fd)
        try:
            with pytest.raises(SessionProtocolError, match="replay"):
                worker._serve_resident(
                    Engine(),
                    read_fd,
                    out_write_fd,
                    session_id=SESSION,
                    launch_digest=LAUNCH,
                    control_dir=str(control),
                    tp_size=2,
                )
        finally:
            for fd in (read_fd, out_read_fd, out_write_fd):
                os.close(fd)


class TestResidentOuterSession:
    def _plan(self):
        from optima.eval.oci_resident_session import ResidentSessionPlan
        from optima.eval.oci_session_protocol import (
            EngineSessionConfig,
            RuntimePreflightFacts,
        )

        config = EngineSessionConfig(
            model_path="/optima/input/model",
            dtype="bfloat16",
            deterministic=False,
            attention_backend="fa4",
            disable_cuda_graph=False,
            mem_fraction_static=0.8,
            log_level="info",
            max_running_requests=64,
            tp_size=2,
            moe_runner_backend=None,
            disable_custom_all_reduce=False,
        )
        facts = RuntimePreflightFacts(
            launch_digest=LAUNCH,
            runtime_digest="c" * 64,
            stack_digest="d" * 64,
            tree_digest="e" * 64,
            engine_config_digest=config.digest,
            worker_distribution_digest="f" * 64,
            model_revision_digest="1" * 64,
            model_manifest_digest="2" * 64,
            model_content_digest="3" * 64,
            sglang_version="0.5.0",
            gpu_architectures=("sm103", "sm103"),
            topology_digest="4" * 64,
            loopback_only=True,
            read_only_inputs=True,
            private_writable_cache=True,
        )
        return ResidentSessionPlan(
            launch_digest=LAUNCH,
            expected_engine_config_digest=config.digest,
            engine_config=config,
            expected_preflight=facts,
            max_swaps=10,
            max_batches=10,
            max_new_tokens=2,
            top_logprobs_num=2,
            temperature=1.0,
        )

    def _transport(self, plan):
        """A scripted in-memory transport implementing the worker side."""
        from optima.eval.oci_session_protocol import (
            BatchEvidence,
            PromptEvidence,
            evidence_frame,
            frame_message,
            preflight_message,
            ready_message,
            swap_evidence_message,
            validate_batch_request,
            validate_init,
            validate_preflight_accept,
            validate_swap_request,
        )

        class Transport:
            def __init__(self) -> None:
                self.outbox: list[bytes] = []
                self.state = "init"
                self.session_id: str | None = None
                self.aborted = False
                self.finalized = False

            def start(self) -> None:
                pass

            def has_pending_output(self) -> bool:
                return False

            def write_frame(self, frame: bytes, *, deadline: float) -> None:
                from optima.eval.oci_session_protocol import parse_frame_bytes

                message = parse_frame_bytes(frame, max_bytes=1 << 27)
                kind = message.get("type")
                if kind == "init":
                    self.session_id, _, _ = validate_init(message)
                    self.outbox.append(
                        frame_message(
                            preflight_message(
                                session_id=self.session_id,
                                launch_digest=LAUNCH,
                                facts=plan.expected_preflight,
                            ),
                            max_bytes=1 << 16,
                        )
                    )
                elif kind == "preflight_accept":
                    validate_preflight_accept(
                        message,
                        session_id=self.session_id,
                        launch_digest=LAUNCH,
                        expected_facts_digest=plan.expected_preflight.digest,
                    )
                    self.outbox.append(
                        frame_message(
                            ready_message(
                                session_id=self.session_id, launch_digest=LAUNCH
                            ),
                            max_bytes=1 << 16,
                        )
                    )
                elif kind == "swap_request":
                    request = validate_swap_request(message)
                    slots = (
                        ()
                        if request.bundle_digest is None
                        else ("moe.fused_experts",)
                    )
                    self.outbox.append(
                        frame_message(
                            swap_evidence_message(
                                request=request, slots=slots, rank_count=2
                            ),
                            max_bytes=1 << 16,
                        )
                    )
                elif kind == "batch_request":
                    request = validate_batch_request(message)
                    prompts = tuple(
                        PromptEvidence(
                            tuple(range(request.max_new_tokens)),
                            tuple(
                                tuple(
                                    (-0.5 - column, column)
                                    for column in range(request.top_logprobs_num)
                                )
                                for _ in range(request.max_new_tokens)
                            ),
                        )
                        for _ in request.prompts
                    )
                    self.outbox.append(
                        evidence_frame(BatchEvidence(prompts), request=request)
                    )
                    self._last_batch = request
                else:  # pragma: no cover - scripted transport
                    raise AssertionError(f"unexpected frame {kind!r}")

            def read_control(self, *, max_bytes: int, deadline: float) -> dict:
                from optima.eval.oci_session_protocol import parse_frame_bytes

                return parse_frame_bytes(self.outbox.pop(0), max_bytes=max_bytes)

            def read_evidence(self, request, *, deadline: float):
                from optima.eval.oci_session_protocol import (
                    parse_evidence_frame_bytes,
                )

                return parse_evidence_frame_bytes(
                    self.outbox.pop(0), request=request
                )

            def finalize(self) -> None:
                self.finalized = True

            def abort(self) -> None:
                self.aborted = True

        return Transport()

    def _open(self):
        from optima.eval.oci_resident_session import ResidentOuterSession

        plan = self._plan()
        transport = self._transport(plan)
        session = ResidentOuterSession(
            plan,
            transport=transport,
            deadline=1_000_000.0,
            init_timeout_s=100.0,
            batch_timeout_s=100.0,
            swap_timeout_s=100.0,
            clock=iter(range(1, 100_000)).__next__,
        )
        session.start()
        return session

    def test_swap_and_read_stream(self) -> None:
        session = self._open()
        receipt = session.swap(DIGEST)
        assert receipt.generation == 1
        assert receipt.slots == ("moe.fused_experts",)
        row = session.execute_batch(("hello",))
        assert row.generation == 1
        assert row.active_slots == ("moe.fused_experts",)
        back = session.swap(None)
        assert back.generation == 2 and back.slots == ()
        canary = session.execute_batch(("hello",), canary=True)
        assert canary.canary and canary.generation == 2
        evidence = session.finish()
        assert len(evidence.batches) == 2
        assert len(evidence.swaps) == 2

    def test_canary_requires_stock(self) -> None:
        from optima.eval.oci_outer_session import OuterSessionInfrastructureError

        session = self._open()
        session.swap(DIGEST)
        with pytest.raises(OuterSessionInfrastructureError, match="canary"):
            session.execute_batch(("hello",), canary=True)

    def test_swap_budget_is_enforced(self) -> None:
        from optima.eval.oci_outer_session import OuterSessionInfrastructureError

        session = self._open()
        for _ in range(10):
            session.swap(None if _ % 2 else DIGEST)
        with pytest.raises(OuterSessionInfrastructureError, match="swap budget"):
            session.swap(DIGEST)

    def test_finish_requires_a_batch(self) -> None:
        from optima.eval.oci_outer_session import OuterSessionInfrastructureError

        session = self._open()
        with pytest.raises(OuterSessionInfrastructureError, match="no batches"):
            session.finish()
