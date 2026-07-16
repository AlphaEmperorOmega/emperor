from __future__ import annotations

from collections.abc import Iterator

import pytest

from emperor_workbench.project_adapter import ProjectAdapterClient
from tests.support.model_packages import install_project_adapter


@pytest.fixture(scope="session", autouse=True)
def project_adapter_fixture() -> Iterator[ProjectAdapterClient]:
    with ProjectAdapterClient() as client:
        install_project_adapter(client)
        try:
            yield client
        finally:
            install_project_adapter(None)
