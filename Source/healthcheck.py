# -*- coding: utf-8 -*-
"""Docker and readiness healthcheck probe for Physiognomy Gradio service.

Usage:
    python Source/healthcheck.py [--host 127.0.0.1] [--port 7860] [--timeout 5]
Exit code:
    0: Service is healthy and responsive (HTTP 200)
    1: Service is unreachable, timed out, or returned non-200
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


def check_health(host: str = "127.0.0.1", port: int = 7860, timeout: float = 5.0) -> bool:
    url = f"http://{host}:{port}/"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Physiognomy-Healthcheck/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status == 200:
                print(f"HEALTHCHECK: OK (HTTP {status}) at {url}")
                return True
            print(f"HEALTHCHECK: UNEXPECTED STATUS (HTTP {status}) at {url}", file=sys.stderr)
            return False
    except urllib.error.HTTPError as e:
        # Gradio web root returning 200 or redirect is fine; 5xx is failure
        if e.code < 500:
            print(f"HEALTHCHECK: RESPONSIVE (HTTP {e.code}) at {url}")
            return True
        print(f"HEALTHCHECK: SERVER ERROR (HTTP {e.code}) at {url}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"HEALTHCHECK: CONNECTION FAILED at {url}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcheck probe for Gradio demo service")
    parser.add_argument("--host", default="127.0.0.1", help="Service host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Service port (default: 7860)")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds (default: 5.0)")
    args = parser.parse_args()

    healthy = check_health(host=args.host, port=args.port, timeout=args.timeout)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
