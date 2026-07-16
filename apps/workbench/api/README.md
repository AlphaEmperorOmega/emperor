# Emperor Workbench API

This package owns the FastAPI application and backend capabilities for the
Emperor Model Workbench. Its installable Python namespace is
`emperor_workbench`; tests remain outside that import package under `tests/`.
The wheel excludes tests, while the source distribution includes them for
contract and installed-product verification.

Production is organized by semantic capability. Each supported package
`__init__.py` is a curated public Interface, underscore-prefixed Modules are
private Implementations, and the FastAPI composition root lives entirely under
`emperor_workbench.api`. Tests mirror that ownership under `architecture/`,
`contract/`, `unit/`, `integration/`, `e2e/`, `fixtures/`, and `support/`.

From this directory, install the API and its development tools, then run its
focused checks:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
```

Start the installed application with `emperor-workbench`, or run
`python -m emperor_workbench` while developing.
