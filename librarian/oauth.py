"""
Clio v0.3 — OAuth authorization server.

Why this exists: claude.ai's connector layer performs OAuth dynamic client
registration on connect and will not accept "this server has no auth" for an
answer (anthropics/claude-ai-mcp #457, #402, #697). Clio was a no-auth server
behind an unguessable path; registration therefore failed intermittently.

This makes Clio a locked door instead of a hidden one.

Design, for a single-owner server:
  - Dynamic client registration is open (Claude must be able to register), but
    registration alone grants NOTHING. The gate is the /approve page, which
    requires the owner's passphrase before any authorization code is issued.
  - All state lives in a JSON sidecar beside the DB. Atomic writes.
  - No third-party identity provider. The passphrase is the resource owner.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

SCOPE = "librarian"
CODE_TTL = 300                       # 5 minutes, per OAuth guidance
ACCESS_TTL = 90 * 24 * 3600          # 90 days — a home server, not a bank
PENDING_TTL = 900                    # abandoned approval attempts expire
PBKDF2_ROUNDS = 240_000


def _hash_passphrase(passphrase: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS
    ).hex()


def redirect_with(redirect_uri: str, **params: str) -> str:
    """Append query params to a redirect URI, preserving any it already has."""
    u = urlparse(str(redirect_uri))
    query = u.query + "&" + urlencode(params) if u.query else urlencode(params)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, query, u.fragment))


class ClioOAuthProvider(OAuthAuthorizationServerProvider):
    """OAuth 2.1 provider backed by a JSON sidecar. One owner, one passphrase."""

    def __init__(self, store_path: str, public_url: str):
        self.path = store_path
        self.public_url = public_url.rstrip("/")
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "clients": {}, "codes": {}, "tokens": {}, "refresh": {}, "pending": {},
            "passphrase": None,
        }
        self._load()

    # ---------------------------------------------------------- persistence --
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
            self._data.update(loaded)
        except (json.JSONDecodeError, OSError):
            # A corrupt store must never take Clio down; it costs a re-auth.
            print(f"[oauth] WARNING: could not read {self.path}; starting fresh.")

    def _save(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _expire(self) -> None:
        now = time.time()
        for bucket in ("codes", "pending"):
            for k in [k for k, v in self._data[bucket].items()
                      if v.get("expires_at", 0) < now]:
                del self._data[bucket][k]
        for k in [k for k, v in self._data["tokens"].items()
                  if v.get("expires_at") and v["expires_at"] < now]:
            del self._data["tokens"][k]

    # ------------------------------------------------------------ passphrase --
    def set_passphrase(self, passphrase: str) -> None:
        salt = secrets.token_hex(16)
        with self._lock:
            self._data["passphrase"] = {
                "salt": salt, "hash": _hash_passphrase(passphrase, salt)
            }
            self._save()

    def has_passphrase(self) -> bool:
        return bool(self._data.get("passphrase"))

    def check_passphrase(self, passphrase: str) -> bool:
        p = self._data.get("passphrase")
        if not p:
            return False
        return hmac.compare_digest(_hash_passphrase(passphrase, p["salt"]), p["hash"])

    # --------------------------------------------------------------- clients --
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        raw = self._data["clients"].get(client_id)
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        # Open registration by design: Claude must be able to register before a
        # human can approve. A registered client holds no access until the
        # passphrase gate is passed.
        with self._lock:
            self._data["clients"][client_info.client_id] = client_info.model_dump(mode="json")
            self._save()

    # ------------------------------------------------------------- authorize --
    async def authorize(self, client: OAuthClientInformationFull,
                        params: AuthorizationParams) -> str:
        """Park the request and send the browser to the passphrase gate."""
        pending_id = secrets.token_urlsafe(24)
        with self._lock:
            self._expire()
            self._data["pending"][pending_id] = {
                "client_id": client.client_id,
                "client_name": client.client_name or client.client_id,
                "params": params.model_dump(mode="json"),
                "expires_at": time.time() + PENDING_TTL,
            }
            self._save()
        return f"{self.public_url}/approve?request={pending_id}"

    def complete_approval(self, pending_id: str) -> str | None:
        """Passphrase accepted: mint the code, return where to send the browser."""
        with self._lock:
            self._expire()
            pending = self._data["pending"].pop(pending_id, None)
            if not pending:
                return None
            params = pending["params"]
            code = secrets.token_urlsafe(32)          # >160 bits of entropy
            self._data["codes"][code] = {
                "code": code,
                "scopes": params.get("scopes") or [SCOPE],
                "expires_at": time.time() + CODE_TTL,
                "client_id": pending["client_id"],
                "code_challenge": params["code_challenge"],
                "redirect_uri": params["redirect_uri"],
                "redirect_uri_provided_explicitly": params["redirect_uri_provided_explicitly"],
                "resource": params.get("resource"),
                "subject": "owner",
            }
            self._save()
            extra = {"code": code}
            if params.get("state"):
                extra["state"] = params["state"]
            return redirect_with(params["redirect_uri"], **extra)

    def peek_pending(self, pending_id: str) -> dict[str, Any] | None:
        p = self._data["pending"].get(pending_id)
        return p if p and p.get("expires_at", 0) > time.time() else None

    # ------------------------------------------------------------ code/token --
    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        raw = self._data["codes"].get(authorization_code)
        if not raw or raw["client_id"] != client.client_id or raw["expires_at"] < time.time():
            return None
        return AuthorizationCode.model_validate(raw)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        with self._lock:
            if authorization_code.code not in self._data["codes"]:
                raise TokenError("invalid_grant", "Authorization code already used or expired")
            del self._data["codes"][authorization_code.code]   # single use, always
            access = secrets.token_urlsafe(32)
            refresh = secrets.token_urlsafe(32)
            self._data["tokens"][access] = {
                "token": access, "client_id": client.client_id,
                "scopes": authorization_code.scopes,
                "expires_at": int(time.time()) + ACCESS_TTL,
                "resource": authorization_code.resource,
                "subject": authorization_code.subject,
            }
            self._data["refresh"][refresh] = {
                "token": refresh, "client_id": client.client_id,
                "scopes": authorization_code.scopes, "expires_at": None,
                "subject": authorization_code.subject,
            }
            self._save()
        return OAuthToken(access_token=access, token_type="Bearer",
                          expires_in=ACCESS_TTL,
                          scope=" ".join(authorization_code.scopes),
                          refresh_token=refresh)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        raw = self._data["refresh"].get(refresh_token)
        if not raw or raw["client_id"] != client.client_id:
            return None
        return RefreshToken.model_validate(raw)

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        with self._lock:
            self._data["refresh"].pop(refresh_token.token, None)   # rotate
            access = secrets.token_urlsafe(32)
            new_refresh = secrets.token_urlsafe(32)
            granted = scopes or refresh_token.scopes
            self._data["tokens"][access] = {
                "token": access, "client_id": client.client_id, "scopes": granted,
                "expires_at": int(time.time()) + ACCESS_TTL, "resource": None,
                "subject": refresh_token.subject,
            }
            self._data["refresh"][new_refresh] = {
                "token": new_refresh, "client_id": client.client_id, "scopes": granted,
                "expires_at": None, "subject": refresh_token.subject,
            }
            self._save()
        return OAuthToken(access_token=access, token_type="Bearer",
                          expires_in=ACCESS_TTL, scope=" ".join(granted),
                          refresh_token=new_refresh)

    async def load_access_token(self, token: str) -> AccessToken | None:
        raw = self._data["tokens"].get(token)
        if not raw:
            return None
        if raw.get("expires_at") and raw["expires_at"] < time.time():
            return None
        return AccessToken.model_validate(raw)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        with self._lock:
            self._data["tokens"].pop(token.token, None)
            self._data["refresh"].pop(token.token, None)
            self._save()
