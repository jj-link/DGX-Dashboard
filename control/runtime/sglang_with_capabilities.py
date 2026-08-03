"""Launch SGLang after attaching the dashboard model-capability contract."""

from __future__ import annotations

import os
import sys

from model_capabilities import ModelCapabilitiesMiddleware
from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def main() -> None:
    load_plugins()

    # run_server imports this same module when it starts the HTTP server, so the
    # middleware is attached to SGLang's real application rather than a proxy.
    from sglang.srt.entrypoints.http_server import app

    app.add_middleware(ModelCapabilitiesMiddleware)
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
