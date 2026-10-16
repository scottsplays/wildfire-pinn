"""API-key check. The key is read from the environment and is never logged."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

from deployment.wildfirepinn_api.limits import MIN_API_KEY_LENGTH


def configured_key() -> str:
    return os.environ.get("WILDFIREPINN_API_KEY", "")


def assert_key_configured() -> None:
    """Require an API key in local mode.

    Vertex IAM serving is the one mode that may start without a key.
    /predict then relies on platform IAM. Other routes stay closed.
    """
    if predict_uses_platform_iam():
        return
    key = configured_key()
    if len(key) < MIN_API_KEY_LENGTH:
        raise RuntimeError(
            "WILDFIREPINN_API_KEY must be set to at least "
            f"{MIN_API_KEY_LENGTH} characters before the service starts. "
            "Do not commit the key."
        )


def presented_key(request: Request) -> str | None:
    direct = request.headers.get("x-api-key")
    if direct:
        return direct
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        return token or None
    return None


def predict_uses_platform_iam() -> bool:
    """Vertex mode skips the application key on /predict only.

    The container must sit behind Vertex IAM. Cloud Run should leave this unset.
    """
    mode = os.environ.get("WILDFIREPINN_PREDICT_AUTH", "api_key")
    if mode == "api_key":
        return False
    if mode == "vertex_iam":
        return True
    raise RuntimeError("WILDFIREPINN_PREDICT_AUTH must be api_key or vertex_iam")


def require_api_key(request: Request) -> None:
    expected = configured_key()
    presented = presented_key(request)
    if len(expected) < MIN_API_KEY_LENGTH or not presented:
        raise HTTPException(status_code=401, detail="missing api key")
    if len(presented) != len(expected) or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="invalid api key")
