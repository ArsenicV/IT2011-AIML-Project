from __future__ import annotations

import argparse
import sys
from pathlib import Path

from textual_serve.server import Server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the DRiST Textual app in a browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()

    app_path = Path(__file__).resolve().with_name("app.py")
    command = f'"{sys.executable}" "{app_path}"'
    Server(command, host=arguments.host, port=arguments.port, title="DRiST").serve(
        debug=arguments.debug
    )


if __name__ == "__main__":
    main()