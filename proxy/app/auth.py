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


class CognitoAuth:
    def __init__(self):
        self._jwks_client = jwt.PyJWKClient(JWKS_URL, cache_keys=True)

    def validate_token(self, authorization: str) -> dict:
        """Validate a Bearer token and return decoded claims.

        Raises HTTPException(401) for invalid/missing tokens,
        HTTPException(403) for missing required scope.
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

        # Check required scope
        scopes = claims.get("scope", "").split()
        if settings.COGNITO_REQUIRED_SCOPE not in scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required scope: {settings.COGNITO_REQUIRED_SCOPE}",
            )

        return claims
