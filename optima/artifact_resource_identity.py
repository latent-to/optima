"""Provider-neutral identity for validator-allocated artifact storage.

The allocation language is shared by every direct artifact provider.  Its
canonical ordering, validation, and digest therefore live outside any CuTe,
CUTLASS, CUDA, or runtime loader module.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from optima.stack_identity import canonical_json_bytes

if TYPE_CHECKING:
    from optima.artifact_abi import ArtifactResourcePlan
    from optima.manifest import ArtifactTargetAuthority


ARTIFACT_RESOURCE_PLAN_SCHEMA = "optima.artifact-resource-plan.v1"


class ArtifactResourceIdentityError(RuntimeError):
    """An artifact resource declaration cannot form canonical identity."""


def artifact_resource_plan_identity(
    *,
    slot: object | None = None,
    authority: ArtifactTargetAuthority | None = None,
    call_frame: object | None = None,
    resources: object,
) -> tuple[ArtifactResourcePlan, dict[str, object], str]:
    """Canonicalize one artifact storage namespace and bind its exact digest.

    Resource declaration order is not execution authority, so the canonical plan
    is name-sorted before hashing. Shape/factor order remains semantic and is
    preserved. The schema tag is part of the digest preimage so allocation-language
    changes cannot collide with this version.
    """

    from optima.artifact_abi import (
        ArtifactABIError,
        ArtifactResource,
        ArtifactResourcePlan,
        SlotCallABI,
    )
    from optima.manifest import (
        ArtifactTargetAuthority,
        ManifestError,
        static_artifact_target_authority,
    )

    try:
        if authority is not None and call_frame is not None:
            raise ArtifactABIError(
                "artifact resource identity may not mix executable authority "
                "with a structural call frame"
            )
        if call_frame is not None:
            if type(call_frame) is not SlotCallABI:
                raise ArtifactABIError(
                    "artifact structural call frame has the wrong type"
                )
            if slot is not None:
                raise ArtifactABIError(
                    "artifact structural call frame already owns target identity"
                )
            call_abi = call_frame
            target_id = call_frame.slot
        elif authority is None:
            if not isinstance(slot, str):
                raise ArtifactABIError(
                    "artifact resource plan requires target authority, structural "
                    "call frame, or static slot"
                )
            authority = static_artifact_target_authority(slot)
            call_abi = authority.call_abi
            target_id = authority.target_id
        else:
            if not isinstance(authority, ArtifactTargetAuthority):
                raise ArtifactABIError("artifact target authority has the wrong type")
            if slot is not None and slot != authority.dispatch_slot:
                raise ArtifactABIError(
                    "artifact target authority dispatch slot mismatch"
                )
            call_abi = authority.call_abi
            target_id = authority.target_id
        if isinstance(resources, ArtifactResourcePlan):
            if resources.slot != target_id:
                raise ArtifactABIError("artifact resource plan slot mismatch")
            declarations = resources.resources
        elif type(resources) is tuple and all(
            isinstance(resource, ArtifactResource) for resource in resources
        ):
            declarations = resources
        else:
            raise ArtifactABIError(
                "artifact resources must be an exact validated resource tuple"
            )
        plan = ArtifactResourcePlan(
            slot=target_id,
            resources=tuple(
                sorted(declarations, key=lambda resource: resource.name)
            ),
        )
        plan.validate_for(call_abi)
        data = plan.to_dict()
        normalized = json.loads(canonical_json_bytes(data).decode("utf-8"))
        if normalized != data:
            raise ArtifactABIError(
                "artifact resource plan is not canonical JSON data"
            )
    except (ArtifactABIError, ManifestError) as exc:
        raise ArtifactResourceIdentityError(
            f"artifact resource plan is invalid: {exc}"
        ) from None
    digest = hashlib.sha256(
        canonical_json_bytes(
            {"plan": normalized, "schema": ARTIFACT_RESOURCE_PLAN_SCHEMA}
        )
    ).hexdigest()
    return plan, normalized, digest


__all__ = [
    "ARTIFACT_RESOURCE_PLAN_SCHEMA",
    "ArtifactResourceIdentityError",
    "artifact_resource_plan_identity",
]
