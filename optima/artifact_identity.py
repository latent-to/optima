"""Canonical submission identity for direct native-artifact declarations.

Direct artifacts execute validator-constructed launch plans, not ``ops.entry``.
Their identity must therefore ignore that legacy, unused Python-callable field and
cover every declaration that can change the generated artifact or its launch:
factory/profile inputs, ordered exports, bindings, specialization, prelaunch,
provider capabilities, and validator-allocated resource/lifecycle storage.

This module is provider-neutral and import-light.  It turns already-parsed manifest
objects into JSON-shaped data; it never imports candidate code or a GPU runtime.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optima.manifest import Manifest, OpEntry


DIRECT_ARTIFACT_ENTRY = "_optima_direct_artifact"
DIRECT_ARTIFACT_IDENTITY_SCHEMA = "optima.direct-artifact-execution.v1"


class ArtifactIdentityError(ValueError):
    """A parsed artifact declaration no longer matches validator reconstruction."""


def _direct_artifact_projection(
    manifest: "Manifest",
    op: "OpEntry",
) -> tuple[dict[str, object], str, list[dict[str, object]]]:
    """Reconstruct the resource plan and executable export rows.

    Manifest parsing already validates the pipeline.  This projection repeats the
    capability derivation because provider and specialization mappings are validator
    policy: changing either mapping must fail reopen, while unrelated allowlist growth
    must not rotate an existing artifact.
    """

    from optima.artifact_abi import ArtifactABIError
    from optima.artifact_device_launch import DeviceLaunchError
    from optima.artifact_provider import ARTIFACT_PROVIDERS, ArtifactBindingABI
    from optima.artifact_resource_identity import (
        ArtifactResourceIdentityError,
        artifact_resource_plan_identity,
    )
    from optima.manifest import Manifest, OpEntry

    if type(manifest) is not Manifest or type(op) is not OpEntry:
        raise ArtifactIdentityError(
            "direct artifact identity requires exact manifest and op rows"
        )
    if not op.aot_exports:
        raise ArtifactIdentityError(
            "direct artifact identity requires at least one artifact export"
        )

    try:
        target_authority = manifest.artifact_target_authority(op)
        resource_plan, resource_plan_data, resource_plan_sha256 = (
            artifact_resource_plan_identity(
                authority=target_authority,
                resources=op.artifact_resources,
            )
        )
        call_abi = target_authority.call_abi

        exports: list[dict[str, object]] = []
        for export in sorted(
            op.aot_exports,
            key=lambda row: (
                row.provider,
                row.plan,
                row.step,
                row.name,
            ),
        ):
            provider = ARTIFACT_PROVIDERS.get(export.provider)
            if provider is not None:
                is_device = (
                    provider.binding_abi is ArtifactBindingABI.CUDA_DRIVER_PARAMS_V1
                )
                if is_device != (export.device_plan is not None):
                    raise ArtifactABIError(
                        f"AOT export {export.name!r} device-plan presence differs "
                        "from its provider binding ABI"
                    )
                if export.device_plan is not None:
                    export.device_plan.validate_bindings(
                        export.bindings,
                        provider_capabilities=provider.provider_capabilities,
                    )
            provider_requirements = call_abi.provider_capability_requirements(
                export.bindings,
                artifact_resources=resource_plan,
            )
            specialization_requirements = (
                call_abi.specialization_capability_requirements(
                    export.specializes,
                    artifact_resources=resource_plan,
                )
            )
            if (
                export.provider_capability_requirements != provider_requirements
                or export.specialization_capability_requirements
                != specialization_requirements
            ):
                raise ArtifactABIError(
                    f"AOT export {export.name!r} capability requirements differ "
                    "from validator reconstruction"
                )
            export_row: dict[str, object] = {
                "bindings": [
                    binding.to_dict() for binding in export.bindings
                ],
                "factory": export.factory,
                "name": export.name,
                "plan": export.plan,
                "prelaunch": [
                    operation.to_dict() for operation in export.prelaunch
                ],
                "profile_inputs": sorted(export.profile_inputs),
                "provider": export.provider,
                "provider_capability_requirements": [
                    requirement.to_dict()
                    for requirement in export.provider_capability_requirements
                ],
                "role": export.role,
                "specialization_capability_requirements": [
                    requirement.to_dict()
                    for requirement in (
                        export.specialization_capability_requirements
                    )
                ],
                "specializes": dict(export.specializes),
                "step": export.step,
            }
            if export.device_plan is not None:
                export_row["device_plan"] = export.device_plan.to_dict()
            exports.append(export_row)
    except (
        ArtifactABIError,
        ArtifactResourceIdentityError,
        DeviceLaunchError,
        ValueError,
    ) as exc:
        raise ArtifactIdentityError(
            f"direct artifact declaration is not canonical: {exc}"
        ) from None

    return resource_plan_data, resource_plan_sha256, exports


def _identity_scalar(value: object) -> object:
    """Encode a bounded scalar in strict, exact JSON identity form.

    Stack identities deliberately reject JSON floats.  The artifact language does
    allow finite float specializations and fill values, so preserve their exact
    binary value with Python's canonical hexadecimal spelling.  The type tag keeps
    ``1.0`` distinct from the separately-valid integer ``1``.
    """

    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactIdentityError("artifact identity contains a non-finite float")
        return {"hex": value.hex(), "type": "float"}
    return value


def _identity_export(row: dict[str, object]) -> dict[str, object]:
    specializes = row.get("specializes")
    prelaunch = row.get("prelaunch")
    if not isinstance(specializes, dict) or not isinstance(prelaunch, list):
        raise ArtifactIdentityError("artifact export identity is malformed")
    identity = dict(row)
    identity["specializes"] = {
        source: _identity_scalar(value)
        for source, value in specializes.items()
    }
    encoded_prelaunch: list[dict[str, object]] = []
    for operation in prelaunch:
        if not isinstance(operation, dict) or "value" not in operation:
            raise ArtifactIdentityError("artifact prelaunch identity is malformed")
        encoded_prelaunch.append(
            {
                **operation,
                "value": _identity_scalar(operation["value"]),
            }
        )
    identity["prelaunch"] = encoded_prelaunch
    return identity


def direct_artifact_execution_identity(
    manifest: "Manifest",
    op: "OpEntry",
) -> dict[str, object]:
    """Return the strict-JSON canonical identity for one direct artifact."""

    resource_plan, resource_plan_sha256, exports = _direct_artifact_projection(
        manifest, op
    )
    return {
        "artifact_resource_plan": resource_plan,
        "artifact_resource_plan_sha256": resource_plan_sha256,
        "exports": [_identity_export(export) for export in exports],
        "schema": DIRECT_ARTIFACT_IDENTITY_SCHEMA,
    }


def direct_artifact_runtime_exports(
    manifest: "Manifest",
    op: "OpEntry",
) -> list[dict[str, object]]:
    """Return executable rows, retaining native scalar types for runtime TOML."""

    _resource_plan, _resource_plan_sha256, exports = _direct_artifact_projection(
        manifest, op
    )
    return exports


__all__ = [
    "ArtifactIdentityError",
    "DIRECT_ARTIFACT_ENTRY",
    "DIRECT_ARTIFACT_IDENTITY_SCHEMA",
    "direct_artifact_execution_identity",
    "direct_artifact_runtime_exports",
]
