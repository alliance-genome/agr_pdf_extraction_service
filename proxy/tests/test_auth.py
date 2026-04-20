"""Tests for Cognito JWT validation."""

import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from app.auth import CognitoAuth
from app.config import settings


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

    @patch("app.auth.jwt.decode")
    @patch("app.auth.jwt.PyJWKClient")
    def test_accepted_client_id_without_pdfx_scope(
        self, mock_jwk_cls, mock_decode, monkeypatch
    ):
        """A token from CurationAPI-Admin (by client_id) is accepted with no PDFX scope."""
        monkeypatch.setattr(
            settings, "COGNITO_ACCEPTED_CLIENT_IDS", "curation-admin-client-id,other-admin"
        )
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key
        mock_decode.return_value = {
            "client_id": "curation-admin-client-id",
            "scope": "curation-api/admin",
            "exp": time.time() + 3600,
        }

        auth = CognitoAuth()
        claims = auth.validate_token("Bearer m2m.admin.token")
        assert claims["client_id"] == "curation-admin-client-id"

    @patch("app.auth.jwt.decode")
    @patch("app.auth.jwt.PyJWKClient")
    def test_unlisted_client_id_without_pdfx_scope_rejected(
        self, mock_jwk_cls, mock_decode, monkeypatch
    ):
        """A token from an unlisted client without the PDFX scope is still rejected."""
        monkeypatch.setattr(
            settings, "COGNITO_ACCEPTED_CLIENT_IDS", "curation-admin-client-id"
        )
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key
        mock_decode.return_value = {
            "client_id": "some-other-client",
            "scope": "some-other/scope",
            "exp": time.time() + 3600,
        }

        auth = CognitoAuth()
        with pytest.raises(HTTPException) as exc:
            auth.validate_token("Bearer fake.jwt.token")
        assert exc.value.status_code == 403

    @patch("app.auth.jwt.decode")
    @patch("app.auth.jwt.PyJWKClient")
    def test_accepted_scopes_env_admits_alternate_scope(
        self, mock_jwk_cls, mock_decode, monkeypatch
    ):
        """COGNITO_ACCEPTED_SCOPES adds alternative scopes to the accepted set."""
        monkeypatch.setattr(
            settings, "COGNITO_ACCEPTED_SCOPES", "curation-api/admin, other-api/read"
        )
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key
        mock_decode.return_value = {
            "scope": "curation-api/admin",
            "exp": time.time() + 3600,
        }

        auth = CognitoAuth()
        claims = auth.validate_token("Bearer alt.scope.token")
        assert "curation-api/admin" in claims["scope"]

    @patch("app.auth.jwt.decode")
    @patch("app.auth.jwt.PyJWKClient")
    def test_token_with_no_scope_claim_is_rejected(self, mock_jwk_cls, mock_decode):
        """A token missing the 'scope' claim entirely should be rejected (403)."""
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key
        mock_decode.return_value = {
            "client_id": "not-allow-listed",
            "exp": time.time() + 3600,
        }

        auth = CognitoAuth()
        with pytest.raises(HTTPException) as exc:
            auth.validate_token("Bearer no.scope.token")
        assert exc.value.status_code == 403

    @patch("app.auth.jwt.PyJWKClient")
    def test_accepted_client_id_does_not_bypass_expiry(self, mock_jwk_cls, monkeypatch):
        """Allow-listed client_id still requires a valid (non-expired) token."""
        import jwt as pyjwt

        monkeypatch.setattr(
            settings, "COGNITO_ACCEPTED_CLIENT_IDS", "curation-admin-client-id"
        )
        mock_key = MagicMock()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value = mock_key

        auth = CognitoAuth()
        with patch("app.auth.jwt.decode", side_effect=pyjwt.ExpiredSignatureError("expired")):
            with pytest.raises(HTTPException) as exc:
                auth.validate_token("Bearer expired.admin.token")
            assert exc.value.status_code == 401
