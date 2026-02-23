"""Tests for Cognito JWT validation."""

import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from app.auth import CognitoAuth


class TestCognitoAuth:
    def _make_auth(self):
        """Create a CognitoAuth with mocked JWKS client."""
        with patch("app.auth.jwt.PyJWKClient"):
            return CognitoAuth()

    def test_missing_authorization_header(self):
        auth = self._make_auth()
        with pytest.raises(HTTPException) as exc:
            auth.validate_token("")
        assert exc.value.status_code == 401

    def test_missing_bearer_prefix(self):
        auth = self._make_auth()
        with pytest.raises(HTTPException) as exc:
            auth.validate_token("not-a-bearer-token")
        assert exc.value.status_code == 401

    def test_bearer_with_empty_token(self):
        auth = self._make_auth()
        with pytest.raises(HTTPException) as exc:
            auth.validate_token("Bearer ")
        assert exc.value.status_code == 401

    @patch("app.auth.jwt.decode")
    @patch("app.auth.jwt.PyJWKClient")
    def test_valid_token_with_correct_scope(self, mock_jwk_cls, mock_decode):
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key
        mock_decode.return_value = {
            "scope": "pdfx-api/extract",
            "exp": time.time() + 3600,
            "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_d3eK6SYpI",
        }

        auth = CognitoAuth()
        claims = auth.validate_token("Bearer fake.jwt.token")
        assert claims["scope"] == "pdfx-api/extract"

    @patch("app.auth.jwt.decode")
    @patch("app.auth.jwt.PyJWKClient")
    def test_missing_required_scope(self, mock_jwk_cls, mock_decode):
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key
        mock_decode.return_value = {
            "scope": "some-other/scope",
            "exp": time.time() + 3600,
        }

        auth = CognitoAuth()
        with pytest.raises(HTTPException) as exc:
            auth.validate_token("Bearer fake.jwt.token")
        assert exc.value.status_code == 403

    @patch("app.auth.jwt.PyJWKClient")
    def test_expired_token(self, mock_jwk_cls):
        import jwt as pyjwt
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key

        auth = CognitoAuth()
        # Simulate ExpiredSignatureError from jwt.decode
        with patch("app.auth.jwt.decode", side_effect=pyjwt.ExpiredSignatureError("expired")):
            with pytest.raises(HTTPException) as exc:
                auth.validate_token("Bearer expired.jwt.token")
            assert exc.value.status_code == 401
            assert "expired" in exc.value.detail.lower()

    @patch("app.auth.jwt.decode")
    @patch("app.auth.jwt.PyJWKClient")
    def test_multiple_scopes_with_required(self, mock_jwk_cls, mock_decode):
        """Token with multiple scopes should pass if required scope is present."""
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key
        mock_decode.return_value = {
            "scope": "openid pdfx-api/extract profile",
            "exp": time.time() + 3600,
        }

        auth = CognitoAuth()
        claims = auth.validate_token("Bearer multi.scope.token")
        assert "pdfx-api/extract" in claims["scope"]
