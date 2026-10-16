"""Authentication for the inference API.

Cloud Run is the IAM boundary: the service is deployed with
``--no-allow-unauthenticated``, and only principals granted
``roles/run.invoker`` can reach the container. This module also checks the
bearer token inside the process, so a prediction route is not left open if
the platform setting is wrong.

``iam`` mode verifies a Google-signed identity token whose audience is the
Cloud Run service URL. ``local`` mode compares the bearer token with
``WILDFIREPINN_LOCAL_TOKEN`` for tests on a workstation. The local token is
never given a default.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

log = logging.getLogger("wildfirepinn.serving")

UNAUTHORIZED = "Unauthorized"


class AuthError(Exception):
    def __init__(self, detail: str = UNAUTHORIZED):
        super().__init__(detail)
        self.detail = detail


def auth_mode() -> str:
    return os.environ.get("WILDFIREPINN_AUTH_MODE", "iam").strip().lower()


def assert_auth_configured() -> None:
    """Refuse to start unless prediction calls must present a real credential."""
    mode = auth_mode()
    if os.environ.get("K_SERVICE") and mode != "iam":
        raise RuntimeError(
            "Cloud Run sets K_SERVICE. This service must use WILDFIREPINN_AUTH_MODE=iam there."
        )
    if mode == "local":
        token = os.environ.get("WILDFIREPINN_LOCAL_TOKEN", "")
        if len(token) < 16:
            raise RuntimeError(
                "WILDFIREPINN_LOCAL_TOKEN must be set to at least 16 characters "
                "when WILDFIREPINN_AUTH_MODE=local."
            )
        return
    if mode == "iam":
        audience = os.environ.get("WILDFIREPINN_AUDIENCE", "").strip()
        if not audience.startswith("https://"):
            raise RuntimeError(
                "WILDFIREPINN_AUDIENCE must be the https URL of this Cloud Run service."
            )
        return
    raise RuntimeError("WILDFIREPINN_AUTH_MODE must be 'iam' or 'local'.")


def _bearer(header: str | None) -> str:
    if header is None:
        raise AuthError()
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError()
    return token.strip()


def _local_match(token: str) -> bool:
    expected = os.environ.get("WILDFIREPINN_LOCAL_TOKEN", "")
    got = hashlib.sha256(token.encode("utf-8")).digest()
    want = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(got, want)


def _verify_google_identity(token: str, audience: str) -> dict:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    claims = id_token.verify_oauth2_token(
        token,
        google_requests.Request(),
        audience=audience,
    )
    if not isinstance(claims, dict):
        raise AuthError()
    return claims


def require_invoker(authorization: str | None) -> dict:
    """Return verified caller claims, or raise AuthError. Does not log the token."""
    mode = auth_mode()
    try:
        token = _bearer(authorization)
    except AuthError:
        log.info("auth rejected reason=missing")
        raise
    try:
        if mode == "local":
            if not _local_match(token):
                raise AuthError()
            return {"mode": "local"}
        if mode == "iam":
            audience = os.environ.get("WILDFIREPINN_AUDIENCE", "").strip()
            claims = _verify_google_identity(token, audience)
            subject = str(claims.get("sub") or "")
            log.info("auth accepted mode=iam subject_set=%s", bool(subject))
            return {"mode": "iam", "subject_present": bool(subject)}
    except AuthError:
        log.info("auth rejected reason=invalid mode=%s", mode)
        raise
    except Exception:
        log.info("auth rejected reason=invalid mode=%s", mode)
        raise AuthError() from None
    log.info("auth rejected reason=unconfigured")
    raise AuthError()
