import sys
import json
import datetime as dt
from pathlib import Path
import requests


AUTH_BASE = "https://auth.enterprise.wikimedia.com/v1"
API_BASE = "https://api.enterprise.wikimedia.com/v2"

# Configuration constants (edit with real credentials and desired output filename)
WMF_USER = "your_username"
WMF_PASS = "your_password"

# File locations are relative to this script directory
HERE = Path(__file__).resolve().parent
TOKEN_PATH = HERE / "wm_enterprise_token.json"
OUTPUT_PATH = HERE / "available_snapshots.json"


def login_and_store() -> str:
    """Login and persist tokens to TOKEN_PATH. Returns access_token."""
    username = WMF_USER
    password = WMF_PASS
    if not username or not password:
        print(
            "Set WMF_USER and WMF_PASS constants at the top of the script.",
            file=sys.stderr,
        )
        sys.exit(1)
    resp = requests.post(
        f"{AUTH_BASE}/login",
        json={"username": username, "password": password},
        timeout=60,
    )
    if not resp.ok:
        print(f"Login failed: {resp.status_code} {resp.text[:500]}", file=sys.stderr)
        sys.exit(2)
    tok = resp.json()
    tok["acquired_at"] = dt.datetime.utcnow().isoformat() + "Z"
    TOKEN_PATH.write_text(json.dumps(tok, indent=2))
    return tok["access_token"]


def load_access_token() -> str:
    if TOKEN_PATH.exists():
        try:
            tok = json.loads(TOKEN_PATH.read_text())
            return tok["access_token"]
        except Exception:
            pass
    return login_and_store()


def auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {load_access_token()}",
        "Accept": "application/json",
    }


def list_latest_ns0_snapshots():
    payload = {
        "fields": [
            "identifier",
            "version",
            "date_modified",
            "in_language",
            "is_part_of",
            "namespace",
            "size",
            "chunks",
        ],
        "filters": [{"field": "namespace.identifier", "value": 0}],
    }
    r = requests.post(
        f"{API_BASE}/snapshots", json=payload, headers=auth_headers(), timeout=120
    )
    if not r.ok:
        print("List snapshots failed:", r.status_code, r.text[:500], file=sys.stderr)
        sys.exit(3)
    snaps = r.json() or []

    # Determine latest snapshot date among results
    import re

    def try_parse_date(s):
        for key in ("date_modified", "date_created", "date_published"):
            v = s.get(key)
            if v:
                try:
                    return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                except Exception:
                    pass
        ver = s.get("version")
        if isinstance(ver, dict):
            for key in ("date_modified", "date_created", "identifier", "value"):
                v = ver.get(key)
                if not v:
                    continue
                try:
                    return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                except Exception:
                    pass
                m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", str(v))
                if m:
                    y, M, d = map(int, m.groups())
                    return dt.datetime(y, M, d, tzinfo=dt.timezone.utc)
        elif isinstance(ver, str):
            try:
                return dt.datetime.fromisoformat(ver.replace("Z", "+00:00"))
            except Exception:
                pass
            m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", ver)
            if m:
                y, M, d = map(int, m.groups())
                return dt.datetime(y, M, d, tzinfo=dt.timezone.utc)
        return None

    dated = [(s, try_parse_date(s)) for s in snaps]
    parsed_dates = [d for _, d in dated if d is not None]
    if parsed_dates:
        latest_dt = max(parsed_dates)
        latest_snaps = [s for s, d in dated if d and d.date() == latest_dt.date()]
    else:
        latest_snaps = snaps
    return latest_snaps


def main():
    latest_snaps = list_latest_ns0_snapshots()
    OUTPUT_PATH.write_text(json.dumps(latest_snaps))
    print(f"Wrote {len(latest_snaps)} snapshots → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
