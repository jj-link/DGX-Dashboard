#!/usr/bin/env python3
"""DGX Dashboard executable and WSGI composition entry point."""

from __future__ import annotations

from dgx_dashboard import create_app, load_config
from waitress import serve


SETTINGS = load_config()
app = create_app(SETTINGS)

HOST = SETTINGS.server.host
PORT = SETTINGS.server.port
REFRESH = SETTINGS.server.refresh_interval


def main() -> None:
    print(f"Dashboard: http://{HOST}:{PORT}")
    print(f"Monitoring {len(SETTINGS.inference_servers)} inference server(s)")
    serve(app, host=HOST, port=PORT, threads=8, channel_timeout=120)


if __name__ == "__main__":
    main()
