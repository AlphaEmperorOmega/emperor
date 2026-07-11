"""Stable executable shim for Training Job cgroup wrapper commands."""

from workbench.backend.training_jobs.cgroup_worker import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
