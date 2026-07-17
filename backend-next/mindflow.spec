# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MindFlow Desktop App.

Packages the FastAPI application as a single executable with no terminal
window.

Hidden imports:
  - sklearn / hmmlearn: ML models (Wave 2 train pipeline)
  - aiosqlite: async SQLite driver (Wave 1 infrastructure)
  - pydantic: validation layer throughout
  - uuid6: UUIDv7 generation (ADR-006)
  - joblib.externals.loky: Parallel backend for sklearn
  - apscheduler: cron/interval scheduling (ADR-007)

Data files:
  - alembic/ + alembic.ini: database migration support at runtime

Excludes:
  - tkinter: GUI toolkit (unused in headless server)
  - test, unittest: test infrastructure (not needed at runtime)

See docs/redesign/04-architecture-design.md §7.1 for the canonical spec
definition.
"""

import os
import sys
from pathlib import Path

# Add src/ to Python path so PyInstaller can find the mindflow package
_HERE = Path(__file__).parent / "src"
sys.path.insert(0, str(_HERE.resolve()))

block_cipher = None

a = Analysis(
    ["src/mindflow/main.py"],
    pathex=[str(_HERE.parent.resolve())],
    binaries=[],
    datas=[
        ("alembic/", "alembic/"),
        ("alembic.ini", "."),
    ],
    hiddenimports=[
        "sklearn",
        "hmmlearn",
        "aiosqlite",
        "pydantic",
        "pydantic_settings",
        "uuid6",
        "joblib",
        "joblib.externals.loky",
        "joblib.externals.loky.backend",
        "apscheduler",
        "apscheduler.triggers.cron",
        "apscheduler.triggers.interval",
        "apscheduler.schedulers.asyncio",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "test",
        "unittest",
        "pytest",
        "hypothesis",
    ],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MindFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # No terminal window (desktop app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
