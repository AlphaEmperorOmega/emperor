from __future__ import annotations

import argparse
import os


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Emperor Workbench API.")
    parser.add_argument(
        "--host",
        default=os.environ.get("WORKBENCH_BACKEND_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=_port(os.environ.get("WORKBENCH_BACKEND_PORT", "9999")),
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "workbench.backend.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reset_contextvars=True,
    )


if __name__ == "__main__":
    main()
