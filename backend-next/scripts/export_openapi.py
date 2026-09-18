"""Export the FastAPI OpenAPI document without starting the server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _BACKEND_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mindflow.app import create_app  # noqa: E402
from mindflow.config import LLMSettings, Settings  # noqa: E402


def export_openapi() -> dict[str, Any]:
    """Build and return the current FastAPI OpenAPI document."""
    # Schema generation must not read the user's provider config or initialise
    # file logging. No lifespan is entered and all paths are disposable.
    with TemporaryDirectory(prefix="mindflow-openapi-") as directory:
        settings = Settings(
            data_dir=Path(directory),
            db_url="sqlite+aiosqlite:///:memory:",
            models_dir=Path(directory) / "models",
            llm=LLMSettings(api_key=None, ollama_enabled=False),
        )
        return create_app(settings, configure_logging=False).openapi()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination JSON file")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(export_openapi(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported OpenAPI schema to {args.output}")


if __name__ == "__main__":
    main()
