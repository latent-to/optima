"""Wire the Optima dispatcher into the MSA PREFILL indexer waist.

The defining function lives at
``sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx``.
The production caller, however, imports that function *by value* into
``minimax_sparse``.  Rebinding only the defining module therefore misses an already
imported caller and silently leaves the candidate cold.  This adapter owns both
bindings:

* patch the defining module before a future ``from ... import`` reads it; and
* patch the consumer's local binding when that module already finished importing.

The bootstrap watches the defining module.  When it is imported as part of the
consumer's ``from`` statement, the post-import hook runs while the consumer is still
initialising; patching the source is sufficient because the pending import then reads
the dispatcher.  All other loaded-consumer states are checked by object identity and
unexpected drift fails closed instead of reporting an installed but unreachable seam.

``OPTIMA_MSA_PREFILL_SEAM=1`` still gates dispatch at call time.  The stock top-k
selection and attend remain validator-owned; the candidate fills only the causal
block-score sheet for ``attention.msa_prefill_block_score``.
"""

from __future__ import annotations

from types import ModuleType

from optima.dispatch import make_msa_prefill_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_msa_prefill_patched"
_SOURCE_MODULE = (
    "sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx"
)
_CONSUMER_MODULE = "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse"
_FUNC = "flash_prefill_with_topk_index"
_ORIG_ATTR = "_optima_orig_flash_prefill_with_topk_index"
_DISPATCH_ATTR = "_optima_dispatch_flash_prefill_with_topk_index"


def _is_initializing(module: ModuleType) -> bool:
    """Return whether importlib is still executing ``module``'s body."""

    spec = getattr(module, "__spec__", None)
    return bool(spec is not None and getattr(spec, "_initializing", False))


def _consumer_binding(
    consumer: ModuleType | None,
    *,
    original: object,
    dispatcher: object,
) -> str:
    """Classify the consumer without mutating it.

    A missing binding is legitimate only during the narrow importlib window in which
    the consumer has entered ``sys.modules`` but its ``from`` statement has not yet
    assigned the name.  Once import completes, missing or foreign bindings mean the
    pinned engine call site drifted and must not be papered over.
    """

    if consumer is None:
        return "absent"
    if not hasattr(consumer, _FUNC):
        if _is_initializing(consumer):
            return "initializing"
        raise RuntimeError(
            "MSA prefill consumer is loaded without "
            f"{_FUNC!r}; refusing an unreachable seam"
        )
    binding = getattr(consumer, _FUNC)
    if binding is original:
        return "original"
    if binding is dispatcher:
        return "dispatcher"
    raise RuntimeError(
        "MSA prefill consumer binding does not match the defining module; "
        "refusing to clobber an unknown call site"
    )


def _installed_state(source: ModuleType) -> tuple[object, object]:
    """Return the sealed original/dispatcher pair or fail on partial state."""

    try:
        original = getattr(source, _ORIG_ATTR)
        dispatcher = getattr(source, _DISPATCH_ATTR)
    except AttributeError as exc:
        raise RuntimeError("MSA prefill seam has incomplete patch state") from exc
    if original is dispatcher:
        raise RuntimeError("MSA prefill dispatcher aliases the stock function")
    if getattr(source, _FUNC, None) is not dispatcher:
        raise RuntimeError(
            "MSA prefill defining-module binding changed after installation"
        )
    return original, dispatcher


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Install an identity-checked dispatcher at both live call-site bindings."""

    import sys

    source = sys.modules.get(_SOURCE_MODULE)
    if source is None:
        return
    consumer = sys.modules.get(_CONSUMER_MODULE)

    if getattr(source, _PATCH_FLAG, False):
        original, dispatcher = _installed_state(source)
        state = _consumer_binding(
            consumer, original=original, dispatcher=dispatcher
        )
        if state == "original":
            setattr(consumer, _FUNC, dispatcher)
        return

    # Stale/private state is never adopted: doing so would let a prior partial patch
    # masquerade as this adapter's trusted original/dispatcher identity.
    if hasattr(source, _ORIG_ATTR) or hasattr(source, _DISPATCH_ATTR):
        raise RuntimeError("MSA prefill defining module has stale Optima patch state")

    original = getattr(source, _FUNC, None)
    if original is None:
        # Preserve the common adapter contract: an absent upstream chokepoint is owned
        # by the compatibility canary.  A loaded, mismatched consumer is still refused
        # below only once there is a defining callable to bind against.
        return
    if not callable(original):
        raise RuntimeError("MSA prefill defining-module chokepoint is not callable")

    dispatcher = make_msa_prefill_dispatcher(original, source, registry=registry)
    if not callable(dispatcher):
        raise RuntimeError("MSA prefill dispatcher factory returned a non-callable")
    if dispatcher is original:
        raise RuntimeError("MSA prefill dispatcher factory returned the stock function")

    # Validate the already-loaded caller before changing either module so failure is
    # atomic.  A consumer still importing is intentionally left alone: its pending
    # ``from`` assignment will read the patched source binding.
    consumer_state = _consumer_binding(
        consumer, original=original, dispatcher=dispatcher
    )

    setattr(source, _ORIG_ATTR, original)
    setattr(source, _DISPATCH_ATTR, dispatcher)
    setattr(source, _FUNC, dispatcher)
    setattr(source, _PATCH_FLAG, True)
    if consumer_state == "original":
        setattr(consumer, _FUNC, dispatcher)


def uninstall() -> None:
    """Restore both bindings, refusing to overwrite unexpected third-party drift."""

    import sys

    source = sys.modules.get(_SOURCE_MODULE)
    if source is None or not getattr(source, _PATCH_FLAG, False):
        return

    original, dispatcher = _installed_state(source)
    consumer = sys.modules.get(_CONSUMER_MODULE)
    consumer_state = _consumer_binding(
        consumer, original=original, dispatcher=dispatcher
    )

    # Validation above precedes every mutation, so an unknown consumer binding leaves
    # the installed seam intact.  A currently importing consumer will finish by reading
    # whichever source binding exists after this call (the restored original).
    if consumer_state == "dispatcher":
        setattr(consumer, _FUNC, original)
    setattr(source, _FUNC, original)
    delattr(source, _ORIG_ATTR)
    delattr(source, _DISPATCH_ATTR)
    setattr(source, _PATCH_FLAG, False)


def is_installed() -> bool:
    """True only when every currently reachable production binding is patched."""

    import sys

    source = sys.modules.get(_SOURCE_MODULE)
    if source is None or not getattr(source, _PATCH_FLAG, False):
        return False
    try:
        original, dispatcher = _installed_state(source)
        state = _consumer_binding(
            sys.modules.get(_CONSUMER_MODULE),
            original=original,
            dispatcher=dispatcher,
        )
    except RuntimeError:
        return False
    return state in {"absent", "initializing", "dispatcher"}
