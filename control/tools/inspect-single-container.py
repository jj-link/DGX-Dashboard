#!/usr/bin/env python3
"""Validate one single-device serving container from Docker inspect JSON."""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("identity", "verify"))
    parser.add_argument("container")
    parser.add_argument("image")
    parser.add_argument("served")
    parser.add_argument("profile", choices=("rtx6000", "spark"))
    args = parser.parse_args()

    try:
        rows = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as error:
        fail(f"invalid Docker inspection response: {error}")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        fail("Docker inspection must return exactly one container")
    container = rows[0]
    if container.get("Name") != f"/{args.container}":
        fail(f"Docker returned the wrong container for {args.container!r}")

    config = container.get("Config")
    state = container.get("State")
    if not isinstance(config, dict) or not isinstance(state, dict):
        fail("Docker inspection is missing Config or State")
    if config.get("Image") != args.image:
        fail(
            f"container {args.container!r} image is {config.get('Image')!r}; "
            f"expected {args.image!r}"
        )
    environment = config.get("Env")
    served_values = [
        value.removeprefix("SERVED=")
        for value in environment if isinstance(value, str) and value.startswith("SERVED=")
    ] if isinstance(environment, list) else []
    if served_values != [args.served]:
        fail(
            f"container {args.container!r} SERVED identity is {served_values!r}; "
            f"expected {[args.served]!r}"
        )

    status = state.get("Status")
    if not isinstance(status, str) or not status:
        fail("Docker inspection is missing the container state")
    if args.mode == "identity":
        print(status)
        return
    if state.get("Running") is not True or status != "running":
        fail(f"container {args.container!r} is {status}, not running")

    host_config = container.get("HostConfig")
    bindings = host_config.get("PortBindings") if isinstance(host_config, dict) else None
    binding = bindings.get("8000/tcp") if isinstance(bindings, dict) else None
    if not isinstance(binding, list) or len(binding) != 1 or not isinstance(binding[0], dict):
        fail(f"container {args.container!r} must publish exactly one 8000/tcp binding")
    host_ip = binding[0].get("HostIp")
    host_port = binding[0].get("HostPort")
    try:
        address = ipaddress.ip_address(host_ip)
        port = int(host_port)
    except (TypeError, ValueError):
        fail(f"container {args.container!r} has an invalid published address")
    if address.version != 4 or not 1 <= port <= 65535:
        fail(f"container {args.container!r} has an invalid published address")
    if args.profile == "rtx6000" and address != ipaddress.ip_address("127.0.0.1"):
        fail(f"RTX container {args.container!r} is not bound to loopback")
    if args.profile == "spark" and address not in ipaddress.ip_network("100.64.0.0/10"):
        fail(f"Spark container {args.container!r} is not bound to a Tailscale IPv4 address")
    print(f"http://{address}:{port}/v1")


if __name__ == "__main__":
    main()
