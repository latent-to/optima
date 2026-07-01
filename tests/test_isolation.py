from types import SimpleNamespace
from unittest import TestCase, mock

from optima.eval import _launch


class IsolationTests(TestCase):
    def test_requested_isolation_fails_closed(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=False,
            allow_unsafe_no_isolation=False,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            with self.assertRaisesRegex(_launch.IsolationError, "could not be proven"):
                _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_framework_mode_requires_isolation_by_default(self):
        cfg = SimpleNamespace(
            isolate=False,
            framework_mode=True,
            allow_unsafe_no_isolation=False,
        )

        with self.assertRaisesRegex(_launch.IsolationError, "framework_mode requires"):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_unsafe_dev_override_allows_failed_isolation(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=True,
            allow_unsafe_no_isolation=True,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)


def test_prepare_and_entry_share_one_module_instance(tmp_path):
    # A (prepare, forward) op's callables must come from ONE module execution: the
    # seam/verify loaders pull both off a single load_module. Two load_entry calls
    # would re-run the body (side effects twice) and split module globals so state
    # written by prepare would be invisible to entry.
    src = tmp_path / "k.py"
    src.write_text(
        "COUNT = 0\n"
        "_STATE = {}\n"
        "def prepare(w13, w2):\n"
        "    _STATE['p'] = 1\n"
        "    return (w13, w2)\n"
        "def entry(*args):\n"
        "    return _STATE.get('p')\n"
    )
    from optima.sandbox import callable_from, load_module

    module = load_module(src)
    prepare = callable_from(module, "prepare")
    entry = callable_from(module, "entry")
    prepare(None, None)
    assert entry() == 1  # shared globals: entry sees what prepare wrote

    # and the documented hazard is real: a SECOND load is a fresh namespace
    from optima.sandbox import load_entry

    entry2 = load_entry(src, "entry")
    assert entry2() is None
