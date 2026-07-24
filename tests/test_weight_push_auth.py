"""Rotatable HMAC credentials for eval → serve-weights push."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from optima.chain.weight_push_auth import (
    PushCredential,
    PushCredentialSet,
    WeightPushAuthError,
    load_push_credentials,
    mint_push_credential,
    sign_push_request,
    verify_push_request,
    write_push_credentials,
)


def test_mint_write_load_and_rotate(tmp_path: Path) -> None:
    path = tmp_path / "push.json"
    first = mint_push_credential(credential_id="alpha")
    write_push_credentials(path, PushCredentialSet((first,)))
    loaded = load_push_credentials(path)
    assert loaded.active()[0].credential_id == "alpha"
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    second = mint_push_credential(credential_id="beta")
    retired = PushCredential(first.credential_id, first.secret, "retired")
    write_push_credentials(path, PushCredentialSet((retired, second)))
    rotated = load_push_credentials(path)
    assert [row.credential_id for row in rotated.active()] == ["beta"]


def test_sign_verify_and_reject_retired_or_tampered() -> None:
    credential = mint_push_credential(credential_id="alpha")
    credentials = PushCredentialSet((credential,))
    body = b'{"lane":"legacy_v1"}\n'
    headers = sign_push_request(credential, timestamp=1_700_000_000, body=body)
    assert (
        verify_push_request(
            credentials,
            headers=headers,
            body=body,
            now=1_700_000_010,
        )
        == "alpha"
    )

    retired = PushCredentialSet(
        (PushCredential(credential.credential_id, credential.secret, "retired"),
         mint_push_credential(credential_id="beta"))
    )
    with pytest.raises(WeightPushAuthError, match="unknown or retired"):
        verify_push_request(
            retired,
            headers=headers,
            body=body,
            now=1_700_000_010,
        )

    bad = dict(headers)
    bad["X-Optima-Push-Body-Digest"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(WeightPushAuthError, match="digest mismatch"):
        verify_push_request(
            credentials,
            headers=bad,
            body=body,
            now=1_700_000_010,
        )


def test_resolve_push_credentials_from_env_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from optima.chain.weight_push_auth import (
        DEFAULT_ENV_CREDENTIAL_ID,
        ENV_PUSH_CREDENTIAL_ID,
        ENV_PUSH_CREDENTIALS,
        ENV_PUSH_KEY,
        resolve_push_credentials,
    )

    monkeypatch.delenv(ENV_PUSH_CREDENTIALS, raising=False)
    monkeypatch.delenv(ENV_PUSH_KEY, raising=False)
    monkeypatch.delenv(ENV_PUSH_CREDENTIAL_ID, raising=False)
    assert resolve_push_credentials(None) is None
    with pytest.raises(WeightPushAuthError, match="required"):
        resolve_push_credentials(None, required=True)

    monkeypatch.setenv(ENV_PUSH_KEY, "x" * 48)
    resolved = resolve_push_credentials(None, required=True)
    assert resolved is not None
    active = resolved.active()
    assert len(active) == 1
    assert active[0].credential_id == DEFAULT_ENV_CREDENTIAL_ID
    assert active[0].secret == "x" * 48

    monkeypatch.setenv(ENV_PUSH_CREDENTIAL_ID, "eval-prod")
    assert resolve_push_credentials(None).active()[0].credential_id == "eval-prod"

    file_cred = mint_push_credential(credential_id="from-file")
    path = tmp_path / "creds.json"
    write_push_credentials(path, PushCredentialSet((file_cred,)))
    monkeypatch.setenv(ENV_PUSH_CREDENTIALS, str(path))
    # File env wins over inline key when no CLI path is given.
    assert resolve_push_credentials(None).active()[0].credential_id == "from-file"

    cli_cred = mint_push_credential(credential_id="from-cli")
    cli_path = tmp_path / "cli.json"
    write_push_credentials(cli_path, PushCredentialSet((cli_cred,)))
    assert (
        resolve_push_credentials(cli_path).active()[0].credential_id == "from-cli"
    )
