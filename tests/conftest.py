"""Test-suite-wide setup.

Stubs environment variables that `src.utils.config` requires at import
time. The observability layer (and any test that transitively imports it)
pulls in `config`, which raises KeyError if `NVIDIA_API_KEY` is missing —
that's correct behavior for the running app but breaks pure-function tests
that never make a real NVIDIA call.

`setdefault` preserves any value the developer has already set, so locally
their real key wins; in CI the stub takes effect.

pytest auto-discovers `conftest.py` and runs it before collecting any test
module, so the env vars below are in place by the time `from src.utils
import observability` is evaluated downstream.
"""

import os

os.environ.setdefault("NVIDIA_API_KEY", "test-stub-not-a-real-key")
