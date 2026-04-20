"""Cognito JWT validation for the proxy API."""

import logging

import jwt
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

JWKS_URL = (
    f"https://cognito-idp.{settings.COGNITO_REGION}.amazonaws.com"
    f"/{settings.COGNITO_USER_POOL_ID}/.well-known/jwks.json"
)
ISSUER = (
    f"https://cognito-idp.{settings.COGNITO_REGION}.amazonaws.com"
    f"/{settings.COGNITO_USER_POOL_ID}"
)


def _split_csv(value: str) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


class CognitoAuth:
    def __init__(self):
        self._jwks_client = jwt.PyJWKClient(JWKS_URL, cache_keys=True)
        # Settings are read once from env at startup; parse the CSVs here so
        # validate_token doesn't reallocate sets on every request.
        self._accepted_scopes: set[str] = _split_csv(settings.COGNITO_ACCEPTED_SCOPES)
        if settings.COGNITO_REQUIRED_SCOPE:
            self._accepted_scopes.add(settings.COGNITO_REQUIRED_SCOPE)
        self._accepted_client_ids: set[str] = _split_csv(settings.COGNITO_ACCEPTED_CLIENT_IDS)

    def validate_token(self, authorization: str) -> dict:
        """Validate a Bearer token and return decoded claims.

        Access is granted when either:
          - the token's client_id is in COGNITO_ACCEPTED_CLIENT_IDS (allow-list
            for shared M2M admin tokens such as CurationAPI-Admin), or
          - the token carries any scope in the accepted-scope set
            (COGNITO_REQUIRED_SCOPE plus COGNITO_ACCEPTED_SCOPES).

        Raises HTTPException(401) for invalid/missing tokens,
        HTTPException(403) when neither client_id nor scope is accepted.
        """
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = authorization[7:]  # strip "Bearer "
        if not token:
            raise HTTPException(status_code=401, detail="Empty token")

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=ISSUER,
                options={"verify_aud": False},  # client_credentials tokens have no aud
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

        client_id = claims.get("client_id") or ""
        if client_id and client_id in self._accepted_client_ids:
            return claims

        token_scopes = set(claims.get("scope", "").split())
        if token_scopes & self._accepted_scopes:
            return claims

        raise HTTPException(
            status_code=403,
            detail=(
                f"Token is not authorized for this service. "
                f"Requires scope in {{{', '.join(sorted(self._accepted_scopes)) or '(none configured)'}}} "
                f"or an allow-listed client_id."
            ),
        )
