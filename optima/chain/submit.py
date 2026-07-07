"""Miner-side submission: bind a bundle to a URL and commit it on chain.

The miner's whole on-chain footprint is one timelock commitment carrying
``{"v":1,"h":content_hash,"u":url}``. The artifact at ``url`` must extract to a
directory whose ``content_hash`` equals ``h`` (use ``package_bundle`` to produce
it); the validator rejects anything else. Hotkey-signed — no coldkey involved.
"""

from __future__ import annotations

from pathlib import Path

from optima.bundle_hash import content_hash
from optima.chain import post_reveal_commitment
from optima.chain.payload import encode_payload


def submit_bundle(subtensor, wallet, netuid: int, bundle_dir: str | Path, url: str, *,
                  blocks_until_reveal: int = 10, dry_run: bool = False) -> dict:
    """Compute the bundle's identity hash, build the payload, and commit it.
    Raises PayloadError before touching the chain if the payload would be rejected."""
    ch = content_hash(bundle_dir)
    data = encode_payload(ch, url)
    result = post_reveal_commitment(subtensor, wallet, netuid, data,
                                    blocks_until_reveal=blocks_until_reveal,
                                    dry_run=dry_run)
    return {**result, "content_hash": ch, "payload": data}
