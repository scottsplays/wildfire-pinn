"""Start the research API on localhost.

The container command binds 0.0.0.0 instead. Set WILDFIREPINN_API_KEY first.
"""

from __future__ import annotations

import os

import uvicorn


def listen() -> tuple[str, int]:
    """Local runs bind to localhost. Containers set WILDFIREPINN_HOST=0.0.0.0.

    Cloud Run injects PORT. Vertex AI custom prediction injects AIP_HTTP_PORT.
    AIP_HTTP_PORT wins when both are set.
    """
    host = os.environ.get("WILDFIREPINN_HOST", "127.0.0.1")
    port = int(os.environ.get("AIP_HTTP_PORT") or os.environ.get("PORT") or "8080")
    return host, port


def main() -> None:
    host, port = listen()
    uvicorn.run(
        "deployment.wildfirepinn_api.service.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
