"""Google Drive OAuth for the headless VPS (Drive API v3, drive.file scope).

Secrets live OUTSIDE git:
  secrets/google_oauth_client.json   (OAuth desktop-app credential, chmod 600)
  secrets/google_drive_token.json    (user token incl. refresh_token, chmod 600)

Never prints client_secret / access_token / refresh_token.
Google client libs are imported lazily so PPO training and unit tests run
without them installed.
"""
from __future__ import annotations

import json
import os
import stat

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRETS_DIR = os.path.join(PROJECT_ROOT, "secrets")
CLIENT_SECRET_PATH = os.path.join(SECRETS_DIR, "google_oauth_client.json")
TOKEN_PATH = os.path.join(SECRETS_DIR, "google_drive_token.json")


def _chmod600(path: str) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def client_secret_present() -> bool:
    return os.path.exists(CLIENT_SECRET_PATH)


def load_credentials():
    """Return google.oauth2.credentials.Credentials or None (missing/invalid).

    Refreshes expired credentials in place and re-saves. Never raises for
    missing files; raises only on genuinely corrupt token content (caller
    treats that as 'needs fresh setup').
    """
    if not os.path.exists(TOKEN_PATH):
        return None
    try:
        from google.oauth2.credentials import Credentials
    except ImportError:
        return None
    try:
        with open(TOKEN_PATH) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        creds = Credentials.from_authorized_user_info(data, SCOPES)
    except (ValueError, KeyError):
        return None
    if creds and creds.expired and creds.refresh_token:
        try:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            save_credentials(creds)
        except Exception:
            return None  # refresh failed -> needs re-auth, not a crash
    return creds if creds and creds.valid else None


def save_credentials(creds) -> None:
    os.makedirs(SECRETS_DIR, exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())
    _chmod600(TOKEN_PATH)


def get_drive_service():
    """Build a Drive v3 service or return None (no creds / no network)."""
    creds = load_credentials()
    if creds is None:
        return None
    try:
        from googleapiclient.discovery import build
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None


def auth_status() -> dict:
    """Machine-readable status without any secret material."""
    if not client_secret_present():
        return {"configured": False, "reason": "missing_client_secret",
                "client_secret_path": CLIENT_SECRET_PATH}
    creds = load_credentials()
    if creds is None:
        if not os.path.exists(TOKEN_PATH):
            return {"configured": False, "reason": "not_authorized",
                    "token_path": TOKEN_PATH}
        return {"configured": False, "reason": "token_invalid_or_refresh_failed",
                "token_path": TOKEN_PATH}
    return {"configured": True, "reason": "ok", "token_path": TOKEN_PATH,
            "scopes": SCOPES}


SETUP_GUIDE = """\
Google Drive first-time setup (run ONCE, from your Windows PC):

  1. The OAuth client JSON must already be at:
       /root/pvp-rl/secrets/google_oauth_client.json   (chmod 600)
     If missing, copy it up (PowerShell):
       scp -P 24534 <client_secret.json> root@103.249.117.202:/root/pvp-rl/secrets/google_oauth_client.json

  2. Open an SSH tunnel so the VPS callback reaches your browser:
       ssh -L 8765:localhost:8765 -p 24534 root@103.249.117.202

  3. In that SSH session run:
       cd /root/pvp-rl && . .venv/bin/activate && python -m tools.google_drive_auth --setup --port 8765

  4. Open the printed Google URL in your Windows browser, consent,
     and the VPS will save secrets/google_drive_token.json (chmod 600).
     Future refreshes are automatic; training never blocks on Drive.
"""


def run_setup(port: int = 8765) -> bool:
    """Interactive first-time OAuth + verify + folders + test upload.

    Returns True only if a test report actually landed in Drive.
    """
    if not client_secret_present():
        print(SETUP_GUIDE)
        print("SETUP_STATUS: missing_client_secret")
        return False
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("SETUP_STATUS: missing_google_deps "
              "(pip install -r requirements-drive.txt)")
        return False
    print("Opening OAuth flow WITHOUT a VPS browser.")
    print("If your browser does not open automatically, copy the printed URL")
    print("into your Windows PC browser (SSH tunnel from the guide required).")
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_PATH, SCOPES)
        creds = flow.run_local_server(port=port, open_browser=False)
    except Exception as e:
        print(f"SETUP_STATUS: oauth_flow_failed ({type(e).__name__})")
        return False
    save_credentials(creds)
    print("Token saved (chmod 600). Verifying Drive API access...")
    from tools import google_drive_reporter as gdr
    service = get_drive_service()
    if service is None:
        print("SETUP_STATUS: verify_failed")
        return False
    try:
        service.about().get(fields="user").execute()
    except Exception:
        print("SETUP_STATUS: verify_failed")
        return False
    print("Drive API OK. Creating folders + uploading one test report...")
    ok = gdr.setup_folders_and_test_upload(service)
    print("SETUP_STATUS:", "ok" if ok else "test_upload_failed")
    return ok


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.status or not a.setup:
        st = auth_status()
        print(json.dumps(st, indent=2))
        if not st["configured"] and st.get("reason") == "not_authorized":
            print(SETUP_GUIDE)
        return
    run_setup(port=a.port)


if __name__ == "__main__":
    main()
