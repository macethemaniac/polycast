#!/usr/bin/env python3
"""
Derive or create Polymarket (CLOB) API credentials using the bundled
`py-clob-client` library.

Usage (PowerShell):

  $env:PRIVATE_KEY = "<your_private_key>"
  $env:CHAIN_ID = "137"  # optional
  python arbitrage_mvp/scripts/derive_polymarket_creds.py

The script will print a JSON object with `api_key`, `secret`, and `passphrase`.
It will also print a PowerShell command you can copy to set `POLYMARKET_API_KEY`.

Security: This script uses your private key locally only. Do not paste
secrets into chat. Run the script on your machine and copy the readonly key
back into the environment if you want me to re-run connectivity tests.
"""
import os
import sys
import json

# Ensure the vendored py-clob-client is importable
VEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendors", "py-clob-client"))
if VEND_PATH not in sys.path:
    sys.path.insert(0, VEND_PATH)

try:
    from py_clob_client.client import ClobClient
except Exception as e:
    print("ERROR: Failed to import py-clob-client. Make sure vendors/py-clob-client is present.")
    print(str(e))
    sys.exit(3)


def main():
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        print("ERROR: PRIVATE_KEY env var not set. Export it and re-run the script.")
        print("Example (PowerShell): $env:PRIVATE_KEY = '0x...' ; python arbitrage_mvp/scripts/derive_polymarket_creds.py")
        sys.exit(2)

    host = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
    chain_id = int(os.getenv("CHAIN_ID", "137"))
    funder = os.getenv("FUNDER")
    nonce = os.getenv("NONCE")
    nonce_val = int(nonce) if nonce else None

    client = ClobClient(host, chain_id, key=private_key, funder=funder)

    try:
        creds = client.create_or_derive_api_creds(nonce=nonce_val)
    except Exception as e:
        print("ERROR: Failed to create/derive API creds:", str(e))
        sys.exit(1)

    if not creds:
        print("ERROR: No creds returned from server")
        sys.exit(1)

    out = {
        "api_key": getattr(creds, "api_key", None) or creds.get("apiKey") if isinstance(creds, dict) else getattr(creds, "api_key", None),
        "secret": getattr(creds, "api_secret", None) or creds.get("secret") if isinstance(creds, dict) else getattr(creds, "api_secret", None),
        "passphrase": getattr(creds, "api_passphrase", None) or creds.get("passphrase") if isinstance(creds, dict) else getattr(creds, "api_passphrase", None),
    }

    print(json.dumps(out, indent=2))
    print("\nPowerShell: set POLYMARKET_API_KEY for this session:")
    print(f'$env:POLYMARKET_API_KEY="{out.get("api_key")}"')


if __name__ == '__main__':
    main()
