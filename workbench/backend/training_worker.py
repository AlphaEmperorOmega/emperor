"""Stable executable shim for persisted Training Job worker commands."""

from workbench.backend.training_jobs.worker import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
