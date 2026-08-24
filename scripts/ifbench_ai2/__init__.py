"""IFBench (Allen AI, 2025) verifier chain — vendored verbatim.

Provenance: github.com/allenai/IFBench, files
`instructions.py`, `instructions_registry.py`, `instructions_util.py`.
Apache-2.0 (see LICENSE next to this file).

The vendored files use bare `import instructions` and `import
instructions_util` (top-level, not relative). Rather than patch the
vendored code, this package prepends its own directory to `sys.path`
at import time so those bare imports resolve to the sibling files.

Usage:
    from scripts.ifbench_ai2 import instructions_registry as reg
    from scripts.ifbench_ai2 import instructions_util as ifutil

Guard against name collision: if the outer program has its own
`instructions` module on `sys.path`, we prepend so ours wins for
this import; but be careful not to `import instructions` at the
outer level after loading this package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Now trigger the resolution: instructions_util → instructions → registry
import instructions_util  # noqa: E402,F401
import instructions  # noqa: E402,F401
import instructions_registry  # noqa: E402,F401
