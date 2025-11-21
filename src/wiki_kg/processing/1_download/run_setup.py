import json
import requests
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def main():
    # Configuration
    LOGIN_URL = "https://auth.enterprise.wikimedia.com/v1/login"
    SNAPSHOTS_URL = "https://api.enterprise.wikimedia.com/v2/snapshots"

    # Credentials
    payload = {
        "username": os.getenv("WIKIMEDIA_USERNAME"),
        "password": os.getenv("WIKIMEDIA_PASSWORD"),
    }

    # Get current directory for file saving
    current_dir = Path(__file__).parent
    token_file = current_dir / "token.json"
    snapshots_file = current_dir / "snapshots.json"

    print(f"Authenticating with {LOGIN_URL}...")

    # 1. Login
    try:
        response = requests.post(
            LOGIN_URL, headers={"Content-Type": "application/json"}, json=payload
        )
        response.raise_for_status()
        token_data = response.json()

        # Save token information
        with open(token_file, "w") as f:
            json.dump(token_data, f, indent=2)
        print(f"Token saved to {token_file}")

        # Extract access token (adjust field name if different, usually 'access_token')
        access_token = token_data.get("access_token")
        if not access_token:
            print("Error: No 'access_token' found in login response.")
            return

    except requests.exceptions.RequestException as e:
        print(f"Login failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(e.response.text)
        return

    print(f"Fetching snapshots from {SNAPSHOTS_URL}...")

    # 2. Get Snapshots
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        response = requests.get(SNAPSHOTS_URL, headers=headers)
        response.raise_for_status()
        snapshots_data = response.json()

        # Save snapshots information
        with open(snapshots_file, "w") as f:
            json.dump(snapshots_data, f, indent=2)
        print(f"Snapshots saved to {snapshots_file}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch snapshots: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(e.response.text)
        return


if __name__ == "__main__":
    main()
