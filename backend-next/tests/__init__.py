"""Test package marker.

Makes ``tests`` importable so the shared LLM test doubles in
:mod:`tests._llm_test_support` can be imported explicitly
(``from tests._llm_test_support import MockLLMWire``) instead of relying on
pytest's rootdir-relative import mode. Without this marker the four LLM
regression suites would each have to carry their own copy of the mocked
transport, the wire recorder, and the concurrency latch.

The marker also disables pytest's rootdir insertion of the ``tests`` directory
into ``sys.path``, which older suites rely on for bare sibling imports
(``from test_prediction_service import ...``). Re-adding the directory here
keeps both import styles working, so the marker stops breaking those modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIRECTORY = str(Path(__file__).resolve().parent)
if _DIRECTORY not in sys.path:
    sys.path.insert(0, _DIRECTORY)
