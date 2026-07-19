"""Canonical operator authority for the one-campaign incentive cutover.

Activation is deliberately a two-step operational act: an operator first writes
and reviews three canonical manifests, then supplies their independently recorded
approval digest to the activation command.  This module only parses and binds
those bytes; it owns neither a wallet nor chain-finality authority.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from optima._strict import duplicate_key_pairs, require_digest
from optima.chain.incentive_composition_store import (
    SelectedIncentiveActivationApproval,
)
from optima.finite_debt import FiniteDebtPolicyManifest
from optima.incentive_composition import IncentiveCompositionPolicyManifest
from optima.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex


MAX_ACTIVATION_INPUT_BYTES = 1 << 20


class IncentiveActivationError(RuntimeError):
    """An operator activation manifest is unsafe or differs from reviewed bytes."""


def _reject_number(_value: str) -> None:
    raise IncentiveActivationError(
        "incentive activation JSON permits integers only, not floats or constants"
    )


def _stable_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_canonical_json(path: str | os.PathLike[str], *, label: str) -> tuple[Any, str]:
    """Read one bounded, stable, single-link canonical JSON authority file."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise IncentiveActivationError("activation inputs require O_NOFOLLOW support")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise IncentiveActivationError(f"cannot open {label}: {exc}") from None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= MAX_ACTIVATION_INPUT_BYTES
        ):
            raise IncentiveActivationError(
                f"{label} must be a nonempty single-link regular file no larger than "
                f"{MAX_ACTIVATION_INPUT_BYTES} bytes"
            )
        chunks: list[bytes] = []
        observed = 0
        while observed <= MAX_ACTIVATION_INPUT_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_ACTIVATION_INPUT_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) != before.st_size
            or len(raw) > MAX_ACTIVATION_INPUT_BYTES
            or _stable_file_identity(after) != _stable_file_identity(before)
        ):
            raise IncentiveActivationError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: duplicate_key_pairs(
                pairs,
                label=label,
                error=IncentiveActivationError,
            ),
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
        encoded = canonical_json_bytes(value)
    except IncentiveActivationError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IncentiveActivationError(f"{label} is not strict UTF-8 JSON: {exc}") from None
    if raw not in {encoded, encoded + b"\n"}:
        raise IncentiveActivationError(f"{label} is not canonically encoded JSON")
    return value, sha256_hex(raw)


@dataclass(frozen=True)
class SelectedIncentiveActivationBundle:
    """Exact reviewed inputs consumed by one atomic local cutover."""

    core_policy: FiniteDebtPolicyManifest
    composition_policy: IncentiveCompositionPolicyManifest
    approval: SelectedIncentiveActivationApproval
    expected_approval_digest: str
    core_policy_file_sha256: str
    composition_policy_file_sha256: str
    approval_file_sha256: str

    def __post_init__(self) -> None:
        if type(self.core_policy) is not FiniteDebtPolicyManifest:
            raise IncentiveActivationError("core policy is not exactly typed")
        if type(self.composition_policy) is not IncentiveCompositionPolicyManifest:
            raise IncentiveActivationError("composition policy is not exactly typed")
        if type(self.approval) is not SelectedIncentiveActivationApproval:
            raise IncentiveActivationError("activation approval is not exactly typed")
        expected = require_digest(
            self.expected_approval_digest,
            field="expected_approval_digest",
            error=IncentiveActivationError,
        )
        object.__setattr__(self, "expected_approval_digest", expected)
        for field in (
            "core_policy_file_sha256",
            "composition_policy_file_sha256",
            "approval_file_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_digest(
                    getattr(self, field), field=field, error=IncentiveActivationError
                ),
            )
        if (
            self.approval.digest != expected
            or self.approval.core_policy_digest != self.core_policy.digest
            or self.approval.composition_policy_digest
            != self.composition_policy.digest
            or self.composition_policy.innovation_policy_digest
            != self.core_policy.digest
        ):
            raise IncentiveActivationError(
                "activation manifests differ from their pinned approval"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "approval": self.approval.to_dict(),
            "approval_file_sha256": self.approval_file_sha256,
            "composition_policy": self.composition_policy.to_dict(),
            "composition_policy_file_sha256": self.composition_policy_file_sha256,
            "core_policy": self.core_policy.to_dict(),
            "core_policy_file_sha256": self.core_policy_file_sha256,
            "expected_approval_digest": self.expected_approval_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.selected-incentive-activation-bundle", self.to_dict()
        )


def load_selected_incentive_activation_bundle(
    *,
    core_policy_path: str | os.PathLike[str],
    composition_policy_path: str | os.PathLike[str],
    approval_path: str | os.PathLike[str],
    expected_approval_digest: str,
) -> SelectedIncentiveActivationBundle:
    """Reopen the three reviewed manifest files without accepting loose JSON."""

    core_value, core_sha = _read_canonical_json(
        Path(core_policy_path), label="core policy"
    )
    composition_value, composition_sha = _read_canonical_json(
        Path(composition_policy_path), label="composition policy"
    )
    approval_value, approval_sha = _read_canonical_json(
        Path(approval_path), label="activation approval"
    )
    try:
        core = FiniteDebtPolicyManifest.from_dict(core_value)
        composition = IncentiveCompositionPolicyManifest.from_dict(composition_value)
        approval = SelectedIncentiveActivationApproval.from_dict(approval_value)
    except (TypeError, ValueError) as exc:
        raise IncentiveActivationError(f"activation manifest is invalid: {exc}") from None
    return SelectedIncentiveActivationBundle(
        core,
        composition,
        approval,
        expected_approval_digest,
        core_sha,
        composition_sha,
        approval_sha,
    )


@dataclass(frozen=True)
class SelectedIncentiveActivationResult:
    """Minimal public receipt for one durable, wallet-free local cutover."""

    chain_scope_digest: str
    bundle_digest: str
    approval_digest: str
    activation_digest: str
    arena_digest: str
    evaluation_stack_digest: str
    catalog_digest: str
    membership_digest: str
    audit_control_manifest_digest: str
    audit_canary_receipt_digest: str
    audit_residual_risk_acceptance_digest: str
    campaign_id: str
    reward_family_ids: tuple[str, ...]
    reserve_hotkey: str
    activation_block: int
    activation_block_hash: str

    def __post_init__(self) -> None:
        for field in (
            "chain_scope_digest",
            "bundle_digest",
            "approval_digest",
            "activation_digest",
            "arena_digest",
            "evaluation_stack_digest",
            "catalog_digest",
            "membership_digest",
            "audit_control_manifest_digest",
            "audit_canary_receipt_digest",
            "audit_residual_risk_acceptance_digest",
            "campaign_id",
        ):
            object.__setattr__(
                self,
                field,
                require_digest(
                    getattr(self, field),
                    field=field,
                    error=IncentiveActivationError,
                ),
            )
        families = tuple(self.reward_family_ids)
        if (
            not families
            or families != tuple(sorted(set(families)))
            or any(
                require_digest(
                    value,
                    field="reward_family_id",
                    error=IncentiveActivationError,
                )
                != value
                for value in families
            )
        ):
            raise IncentiveActivationError(
                "activation result reward-family roster is malformed"
            )
        object.__setattr__(self, "reward_family_ids", families)
        if (
            not isinstance(self.reserve_hotkey, str)
            or not self.reserve_hotkey
            or len(self.reserve_hotkey) > 256
            or any(char.isspace() for char in self.reserve_hotkey)
        ):
            raise IncentiveActivationError("activation result reserve hotkey is malformed")
        if type(self.activation_block) is not int or self.activation_block < 0:
            raise IncentiveActivationError("activation result block is malformed")
        if (
            not isinstance(self.activation_block_hash, str)
            or len(self.activation_block_hash) != 66
            or not self.activation_block_hash.startswith("0x")
        ):
            raise IncentiveActivationError("activation result block hash is malformed")
        try:
            int(self.activation_block_hash[2:], 16)
        except ValueError:
            raise IncentiveActivationError(
                "activation result block hash is malformed"
            ) from None

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_block": self.activation_block,
            "activation_block_hash": self.activation_block_hash,
            "activation_digest": self.activation_digest,
            "approval_digest": self.approval_digest,
            "arena_digest": self.arena_digest,
            "audit_canary_receipt_digest": self.audit_canary_receipt_digest,
            "audit_control_manifest_digest": self.audit_control_manifest_digest,
            "audit_residual_risk_acceptance_digest": (
                self.audit_residual_risk_acceptance_digest
            ),
            "bundle_digest": self.bundle_digest,
            "campaign_id": self.campaign_id,
            "catalog_digest": self.catalog_digest,
            "chain_scope_digest": self.chain_scope_digest,
            "evaluation_stack_digest": self.evaluation_stack_digest,
            "membership_digest": self.membership_digest,
            "reserve_hotkey": self.reserve_hotkey,
            "reward_family_ids": list(self.reward_family_ids),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.selected-incentive-activation-result", self.to_dict()
        )


def selected_model_campaign_id(
    *,
    arena_digest: str,
    catalog_digest: str,
    reward_family_ids: tuple[str, ...],
) -> str:
    """Derive one campaign identity from its exact arena/catalog/family roster."""

    arena = require_digest(
        arena_digest, field="arena_digest", error=IncentiveActivationError
    )
    catalog = require_digest(
        catalog_digest, field="catalog_digest", error=IncentiveActivationError
    )
    families = tuple(reward_family_ids)
    if (
        not families
        or families != tuple(sorted(set(families)))
        or any(
            require_digest(
                value,
                field="reward_family_id",
                error=IncentiveActivationError,
            )
            != value
            for value in families
        )
    ):
        raise IncentiveActivationError("campaign reward-family roster is malformed")
    return canonical_digest(
        "optima.economics.model-campaign.v1",
        {
            "arena_digest": arena,
            "catalog_digest": catalog,
            "reward_family_ids": list(families),
        },
    )


def _preflight_selected_campaign(
    *,
    store: object,
    approval: SelectedIncentiveActivationApproval,
    metagraph: object,
    netuid: int,
) -> dict[str, str]:
    """Cross-check reviewed launch bytes against retained arena and chain facts."""

    from optima.chain.finite_debt_store import reward_family_id

    try:
        states = tuple(getattr(store, "evaluation_stacks")())
        metagraph_netuid = getattr(metagraph, "netuid")
        metagraph_block = getattr(metagraph, "block")
        metagraph_hash = str(getattr(metagraph, "block_hash")).lower()
        hotkeys = tuple(getattr(metagraph, "hotkeys"))
        uids = tuple(getattr(metagraph, "uids"))
    except Exception as exc:
        raise IncentiveActivationError(
            f"cannot reopen campaign preflight authority: {exc}"
        ) from None
    if (
        metagraph_netuid != netuid
        or metagraph_block != approval.activation_block
        or metagraph_hash != approval.activation_block_hash
        or len(hotkeys) != len(uids)
        or len(set(hotkeys)) != len(hotkeys)
        or len(set(uids)) != len(uids)
        or approval.reserve_hotkey not in hotkeys
    ):
        raise IncentiveActivationError(
            "activation reserve or metagraph differs from the approved boundary"
        )
    matches = tuple(
        state for state in states if state.arena_digest == approval.arena_digest
    )
    if len(matches) != 1:
        raise IncentiveActivationError(
            "retained arena differs from the pinned activation approval"
        )
    state = matches[0]
    if state.manifest.catalog_digest != approval.catalog_digest:
        raise IncentiveActivationError(
            "retained catalog differs from the pinned activation approval"
        )
    if state.digest != approval.evaluation_stack_digest:
        raise IncentiveActivationError(
            "retained evaluation stack differs from the pinned activation approval"
        )
    try:
        snapshot = state.manifest.catalog_snapshot
        targets = snapshot["targets"]
        if type(targets) is not list or not targets:
            raise ValueError("catalog targets are missing")
        families = tuple(
            sorted(
                reward_family_id(
                    state.arena_digest,
                    row["target_id"],
                    canonical_digest("optima.target-spec", row),
                )
                for row in targets
            )
        )
    except Exception as exc:
        raise IncentiveActivationError(
            f"retained evaluation catalog cannot define reward families: {exc}"
        ) from None
    if families != approval.reward_family_ids:
        raise IncentiveActivationError(
            "retained reward-family roster differs from the pinned activation approval"
        )
    campaign_id = selected_model_campaign_id(
        arena_digest=state.arena_digest,
        catalog_digest=state.manifest.catalog_digest,
        reward_family_ids=families,
    )
    if campaign_id != approval.campaign_id:
        raise IncentiveActivationError(
            "approved campaign ID differs from its retained arena/catalog roster"
        )
    members = [
        {"hotkey": hotkey, "uid": uid}
        for uid, hotkey in sorted(
            zip(uids, hotkeys, strict=True), key=lambda item: (item[0], item[1])
        )
    ]
    preflight = {
        "arena_digest": state.arena_digest,
        "catalog_digest": state.manifest.catalog_digest,
        "evaluation_stack_digest": state.digest,
        "membership_digest": canonical_digest(
            "optima.economics.metagraph-membership",
            {
                "block": metagraph_block,
                "block_hash": metagraph_hash,
                "chain_scope_digest": approval.chain_scope_digest,
                "members": members,
            },
        ),
    }
    for field in (
        "arena_digest",
        "evaluation_stack_digest",
        "catalog_digest",
        "membership_digest",
    ):
        if preflight[field] != getattr(approval, field):
            raise IncentiveActivationError(
                f"retained {field.replace('_', ' ')} differs from the pinned "
                "activation approval"
            )
    return preflight


def execute_selected_incentive_activation(
    *,
    network: str,
    netuid: int,
    intake_db: str | os.PathLike[str],
    core_policy_path: str | os.PathLike[str],
    composition_policy_path: str | os.PathLike[str],
    approval_path: str | os.PathLike[str],
    expected_approval_digest: str,
    connect: Callable[[str], object],
    read_finalized_head: Callable[[object], tuple[int, str]],
    fetch_metagraph: Callable[..., object],
) -> SelectedIncentiveActivationResult:
    """Activate reviewed bytes only at the intake database's exact chain cursor.

    This path intentionally constructs no wallet and signs nothing.  The RPC is
    used solely to reopen genesis scope, finalized ancestry, and the approved
    historical block hash before the SQLite transaction is allowed to commit.
    """

    from optima.chain.intake import FinalizedIntakeStore, IntakeError, IntakeScope

    if not isinstance(network, str) or not network:
        raise IncentiveActivationError("activation network is malformed")
    if type(netuid) is not int or netuid < 0:
        raise IncentiveActivationError("activation netuid is malformed")
    bundle = load_selected_incentive_activation_bundle(
        core_policy_path=core_policy_path,
        composition_policy_path=composition_policy_path,
        approval_path=approval_path,
        expected_approval_digest=expected_approval_digest,
    )
    try:
        subtensor = connect(network)
        genesis_hash = str(getattr(subtensor, "get_block_hash")(0)).lower()
        finalized_block, finalized_hash = read_finalized_head(subtensor)
    except Exception as exc:
        raise IncentiveActivationError(
            f"cannot reopen activation chain authority: {exc}"
        ) from None
    try:
        scope = IntakeScope(genesis_hash, netuid)
    except (TypeError, ValueError, IntakeError) as exc:
        raise IncentiveActivationError(f"activation chain scope is invalid: {exc}") from None
    approval = bundle.approval
    if approval.chain_scope_digest != scope.digest:
        raise IncentiveActivationError(
            "activation approval differs from the requested chain scope"
        )
    if (
        type(finalized_block) is not int
        or finalized_block < approval.activation_block
        or not isinstance(finalized_hash, str)
    ):
        raise IncentiveActivationError(
            "activation approval is newer than the finalized chain head"
        )
    try:
        historical_hash = str(
            getattr(subtensor, "get_block_hash")(approval.activation_block)
        ).lower()
        activation_metagraph = fetch_metagraph(
            subtensor,
            netuid,
            block=approval.activation_block,
        )
    except Exception as exc:
        raise IncentiveActivationError(
            f"cannot reopen approved activation block: {exc}"
        ) from None
    if historical_hash != approval.activation_block_hash:
        raise IncentiveActivationError(
            "approved activation block is not on the finalized chain"
        )
    if (
        finalized_block == approval.activation_block
        and str(finalized_hash).lower() != approval.activation_block_hash
    ):
        raise IncentiveActivationError(
            "finalized head differs from the approved activation block"
        )
    with FinalizedIntakeStore(intake_db, scope=scope) as store:
        if store.finalized_cursor() != (
            approval.activation_block,
            approval.activation_block_hash,
        ):
            raise IncentiveActivationError(
                "intake cursor differs from the approved activation boundary"
            )
        preflight = _preflight_selected_campaign(
            store=store,
            approval=approval,
            metagraph=activation_metagraph,
            netuid=netuid,
        )
        activation = store.activate_selected_incentives(
            bundle.core_policy,
            bundle.composition_policy,
            approval,
            expected_approval_digest=bundle.expected_approval_digest,
        )
    return SelectedIncentiveActivationResult(
        scope.digest,
        bundle.digest,
        approval.digest,
        activation.digest,
        preflight["arena_digest"],
        preflight["evaluation_stack_digest"],
        preflight["catalog_digest"],
        preflight["membership_digest"],
        approval.audit_control_manifest_digest,
        approval.audit_canary_receipt_digest,
        approval.audit_residual_risk_acceptance_digest,
        approval.campaign_id,
        approval.reward_family_ids,
        approval.reserve_hotkey,
        approval.activation_block,
        approval.activation_block_hash,
    )


__all__ = [
    "IncentiveActivationError",
    "MAX_ACTIVATION_INPUT_BYTES",
    "SelectedIncentiveActivationBundle",
    "SelectedIncentiveActivationResult",
    "execute_selected_incentive_activation",
    "load_selected_incentive_activation_bundle",
    "selected_model_campaign_id",
]
