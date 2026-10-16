#!/usr/bin/env python3
"""Run the FastAPI control plane (uvicorn)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    import uvicorn

    from wtb.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "wtb.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        log_level=settings.app.log_level.lower(),
    )


if __name__ == "__main__":
    main()
