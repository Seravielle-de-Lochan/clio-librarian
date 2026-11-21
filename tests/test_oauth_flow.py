"""End-to-end OAuth test: discovery -> DCR -> passphrase gate -> token -> authed MCP call.

Exercises exactly the handshake claude.ai performs, including PKCE.
Run against a server started with --oauth on TEST_PORT.
"""
import base64
import hashlib
import json
import re
import secrets
import sys
import urllib.parse

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8399"
MCP_PATH = sys.argv[2] if len(sys.argv) > 2 else "/mcp"
PASSPHRASE = sys.argv[3] if len(sys.argv) > 3 else "test-passphrase"
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


c = httpx.Client(follow_redirects=False, timeout=60)

# 1. unauthenticated MCP call must be refused
r = c.post(BASE + MCP_PATH, json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                             "clientInfo": {"name": "t", "version": "0"}}},
           headers={"Accept": "application/json, text/event-stream"})
check("unauthenticated call rejected", r.status_code == 401, f"HTTP {r.status_code}")

# 2. discovery
r = c.get(BASE + "/.well-known/oauth-authorization-server")
check("authorization-server metadata", r.status_code == 200, f"HTTP {r.status_code}")
meta = r.json() if r.status_code == 200 else {}
for key in ("issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint"):
    check(f"metadata advertises {key}", key in meta, str(meta.get(key, ""))[:70])

r = c.get(BASE + "/.well-known/oauth-protected-resource" + MCP_PATH)
check("protected-resource metadata", r.status_code == 200, f"HTTP {r.status_code}")

# 3. dynamic client registration — the step that was failing
redirect_uri = "https://claude.ai/api/mcp/auth_callback"
r = c.post(meta["registration_endpoint"], json={
    "client_name": "Claude", "redirect_uris": [redirect_uri],
    "grant_types": ["authorization_code", "refresh_token"],
    "response_types": ["code"], "token_endpoint_auth_method": "client_secret_post",
})
check("dynamic client registration", r.status_code in (200, 201), f"HTTP {r.status_code}")
reg = r.json()
client_id, client_secret = reg.get("client_id"), reg.get("client_secret")
check("registration returned client_id", bool(client_id))

# 4. authorize -> should send the browser to the passphrase gate
verifier = secrets.token_urlsafe(64)
challenge = base64.urlsafe_b64encode(
    hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
state = secrets.token_urlsafe(16)
q = urllib.parse.urlencode({
    "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
    "code_challenge": challenge, "code_challenge_method": "S256",
    "state": state, "scope": "librarian",
})
r = c.get(f"{meta['authorization_endpoint']}?{q}")
check("authorize redirects to approval gate",
      r.status_code in (302, 307) and "/approve" in r.headers.get("location", ""),
      r.headers.get("location", "")[:70])
approve_url = r.headers.get("location", "")
request_id = urllib.parse.parse_qs(urllib.parse.urlparse(approve_url).query).get("request", [""])[0]

# 5. the gate itself
r = c.get(approve_url)
check("approval page renders", r.status_code == 200 and "Clio" in r.text)

r = c.post(BASE + "/approve", data={"request": request_id, "passphrase": "wrong-one"})
check("wrong passphrase refused", r.status_code == 401)

r = c.post(BASE + "/approve", data={"request": request_id, "passphrase": PASSPHRASE})
loc = r.headers.get("location", "")
check("correct passphrase redirects back to client",
      r.status_code == 302 and loc.startswith(redirect_uri), loc[:60])
qs = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
code = qs.get("code", [""])[0]
check("state round-tripped", qs.get("state", [""])[0] == state)

# 6. token exchange with PKCE
r = c.post(meta["token_endpoint"], data={
    "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
    "client_id": client_id, "client_secret": client_secret, "code_verifier": verifier,
})
check("token exchange", r.status_code == 200, f"HTTP {r.status_code} {r.text[:90]}")
tok = r.json() if r.status_code == 200 else {}
access, refresh = tok.get("access_token"), tok.get("refresh_token")
check("access token issued", bool(access))

# 7. code is single-use
r = c.post(meta["token_endpoint"], data={
    "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
    "client_id": client_id, "client_secret": client_secret, "code_verifier": verifier,
})
check("authorization code is single-use", r.status_code >= 400, f"HTTP {r.status_code}")

# 8. authenticated MCP call
r = c.post(BASE + MCP_PATH, json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                             "clientInfo": {"name": "t", "version": "0"}}},
           headers={"Accept": "application/json, text/event-stream",
                    "Authorization": f"Bearer {access}"})
check("authenticated MCP handshake", r.status_code == 200, f"HTTP {r.status_code}")
sid = r.headers.get("mcp-session-id", "")
body = r.text
check("server identifies as clio", "clio" in body)

# 9. tools reachable with the token — with a session id if one was issued,
#    without one if the server is stateless. Never skipped: a skipped check
#    reads as a pass, and this is the check that proves she can actually work.
print(f"       (server is {'SESSION-BASED' if sid else 'STATELESS'})")
hdr = {"Accept": "application/json, text/event-stream",
       "Authorization": f"Bearer {access}"}
if sid:
    hdr["mcp-session-id"] = sid
    c.post(BASE + MCP_PATH, headers=hdr,
           json={"jsonrpc": "2.0", "method": "notifications/initialized"})
r = c.post(BASE + MCP_PATH, headers=hdr,
           json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
names = set(re.findall(r'"name":"([a-z_]+)"', r.text))
check("all three tools advertised",
      {"memory_start", "search_memory", "recall_window"} <= names, str(sorted(names)))

# 9b. the actual point of stateless mode: a caller that never handshook at all
#     still gets served. This is what survives a nightly restart.
if not sid:
    r = c.post(BASE + MCP_PATH,
               headers={"Accept": "application/json, text/event-stream",
                        "Authorization": f"Bearer {access}"},
               json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    cold = set(re.findall(r'"name":"([a-z_]+)"', r.text))
    check("cold call with no session is served (restart-proof)",
          {"memory_start", "search_memory", "recall_window"} <= cold,
          f"HTTP {r.status_code} {sorted(cold)}")

# 10. refresh token works
r = c.post(meta["token_endpoint"], data={
    "grant_type": "refresh_token", "refresh_token": refresh,
    "client_id": client_id, "client_secret": client_secret, "scope": "librarian",
})
check("refresh token exchange", r.status_code == 200, f"HTTP {r.status_code} {r.text[:80]}")

# 11. a bad token is refused
r = c.post(BASE + MCP_PATH, json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                             "clientInfo": {"name": "t", "version": "0"}}},
           headers={"Accept": "application/json, text/event-stream",
                    "Authorization": "Bearer not-a-real-token"})
check("forged token rejected", r.status_code == 401, f"HTTP {r.status_code}")

print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
