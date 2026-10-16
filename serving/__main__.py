"""Start the API. Cloud Run sets PORT; local use can leave it at 8080."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("serving.app:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
