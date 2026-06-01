"""Pytest configuration for Q-Drop-Integration tests.

Puts ``src/`` on ``sys.path`` so tests can import the project's flat-layout
packages (e.g. ``qdb.noise_schedule``) without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
