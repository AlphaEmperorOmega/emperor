from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = TESTS_ROOT.parent
REPOSITORY_ROOT = API_ROOT.parents[2]

__all__ = ["API_ROOT", "REPOSITORY_ROOT", "TESTS_ROOT"]
