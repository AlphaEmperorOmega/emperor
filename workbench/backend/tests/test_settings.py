from __future__ import annotations

import os
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from workbench.backend.core.config import (
    DEFAULT_SNAPSHOTS_ROOT,
    DEFAULT_TRUSTED_HOSTS,
    LOCAL_FRONTEND_ORIGINS,
    WorkbenchApiSettings,
    get_workbench_api_settings,
)
from workbench.backend.core.limits import (
    DEFAULT_LOG_ARCHIVE_UPLOAD_CONCURRENCY,
    DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
    DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
    DEFAULT_MAX_LOG_ARCHIVE_UPLOAD_SIZE,
)

SETTINGS_ENV_NAMES = (
    "WORKBENCH_API_AUTH_MODE",
    "WORKBENCH_API_TOKEN",
    "WORKBENCH_API_CORS_ORIGINS",
    "WORKBENCH_API_TRUSTED_HOSTS",
    "WORKBENCH_API_SNAPSHOTS_ROOT",
    "WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS",
    "WORKBENCH_API_ALLOW_LOG_IMPORTS",
    "WORKBENCH_API_MAX_UPLOAD_SIZE",
    "WORKBENCH_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE",
    "WORKBENCH_API_MAX_LOG_ARCHIVE_MEMBER_COUNT",
    "WORKBENCH_API_MAX_LOG_ARCHIVE_PATH_BYTES",
    "WORKBENCH_API_LOG_ARCHIVE_UPLOAD_CONCURRENCY",
)


@contextmanager
def isolated_settings_env(**values: str) -> Iterator[None]:
    original = {name: os.environ.get(name) for name in SETTINGS_ENV_NAMES}
    try:
        for name in SETTINGS_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(values)
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class WorkbenchApiSettingsTests(unittest.TestCase):
    def test_defaults_allow_only_local_api_hosts(self) -> None:
        with isolated_settings_env():
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.trusted_hosts, DEFAULT_TRUSTED_HOSTS)
        self.assertNotIn("testserver", settings.trusted_hosts)

    def test_env_parses_trusted_hosts_json_array(self) -> None:
        with isolated_settings_env(
            WORKBENCH_API_TRUSTED_HOSTS='["api.example.com","testserver"]',
        ):
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.trusted_hosts, ["api.example.com", "testserver"])

    def test_legacy_settings_module_reexports_canonical_settings(self) -> None:
        from workbench.backend import settings as legacy_settings

        self.assertIs(legacy_settings.LOCAL_FRONTEND_ORIGINS, LOCAL_FRONTEND_ORIGINS)
        self.assertIs(legacy_settings.WorkbenchApiSettings, WorkbenchApiSettings)
        self.assertIs(
            legacy_settings.get_workbench_api_settings,
            get_workbench_api_settings,
        )

    def test_defaults_keep_local_development_unauthenticated_and_read_only(
        self,
    ) -> None:
        with isolated_settings_env():
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.auth_mode, "none")
        self.assertIsNone(settings.token)
        self.assertIs(settings.allow_unsafe_local_mutations, False)
        self.assertIs(settings.log_imports_enabled, False)
        self.assertIsNone(settings.max_upload_size)
        self.assertIsNone(settings.max_log_archive_extracted_size)
        self.assertEqual(
            settings.effective_max_upload_size,
            DEFAULT_MAX_LOG_ARCHIVE_UPLOAD_SIZE,
        )
        self.assertEqual(
            settings.effective_max_log_archive_extracted_size,
            DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
        )
        self.assertEqual(
            settings.max_log_archive_member_count,
            DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
        )
        self.assertEqual(
            settings.max_log_archive_path_bytes,
            DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
        )
        self.assertEqual(
            settings.log_archive_upload_concurrency,
            DEFAULT_LOG_ARCHIVE_UPLOAD_CONCURRENCY,
        )

    def test_bearer_mode_defaults_disable_log_imports(self) -> None:
        settings = WorkbenchApiSettings(auth_mode="bearer", token="secret-token")

        self.assertIs(settings.log_imports_enabled, False)

    def test_hosted_or_explicit_log_imports_use_default_size_limits(self) -> None:
        bearer_settings = WorkbenchApiSettings(
            auth_mode="bearer",
            token="secret-token",
            allow_log_imports=True,
        )
        local_override_settings = WorkbenchApiSettings(allow_log_imports=True)

        for settings in (bearer_settings, local_override_settings):
            with self.subTest(settings=settings):
                self.assertEqual(
                    settings.effective_max_upload_size,
                    DEFAULT_MAX_LOG_ARCHIVE_UPLOAD_SIZE,
                )
                self.assertEqual(
                    settings.effective_max_log_archive_extracted_size,
                    DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
                )

    def test_none_cannot_disable_log_import_size_limits(self) -> None:
        settings = WorkbenchApiSettings(
            max_upload_size=None,
            max_log_archive_extracted_size=None,
        )

        self.assertEqual(
            settings.effective_max_upload_size,
            DEFAULT_MAX_LOG_ARCHIVE_UPLOAD_SIZE,
        )
        self.assertEqual(
            settings.effective_max_log_archive_extracted_size,
            DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
        )

    def test_defaults_keep_local_development_cors_origins(self) -> None:
        with isolated_settings_env():
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.cors_origins, LOCAL_FRONTEND_ORIGINS)
        self.assertIsNot(settings.cors_origins, LOCAL_FRONTEND_ORIGINS)

    def test_default_snapshots_root_is_owned_by_workbench(self) -> None:
        with isolated_settings_env():
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.snapshots_root, DEFAULT_SNAPSHOTS_ROOT)
        self.assertEqual(Path(settings.snapshots_root).name, "snapshots")
        self.assertEqual(Path(settings.snapshots_root).parent.name, "workbench")

    def test_bearer_mode_accepts_non_empty_token(self) -> None:
        settings = WorkbenchApiSettings(auth_mode="bearer", token="secret-token")

        self.assertEqual(settings.auth_mode, "bearer")
        self.assertEqual(settings.token, "secret-token")

    def test_bearer_mode_rejects_missing_or_empty_token(self) -> None:
        for token in (None, "", "   "):
            with self.subTest(token=token):
                kwargs = {"auth_mode": "bearer"}
                if token is not None:
                    kwargs["token"] = token

                with self.assertRaises(ValidationError):
                    WorkbenchApiSettings(**kwargs)

    def test_invalid_auth_mode_rejects(self) -> None:
        with self.assertRaises(ValidationError):
            WorkbenchApiSettings(auth_mode="basic", token="secret-token")

    def test_env_parses_auth_mode_and_token(self) -> None:
        with isolated_settings_env(
            WORKBENCH_API_AUTH_MODE="bearer",
            WORKBENCH_API_TOKEN="env-secret",
        ):
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.auth_mode, "bearer")
        self.assertEqual(settings.token, "env-secret")

    def test_env_parses_unsafe_local_mutation_opt_in(self) -> None:
        with isolated_settings_env(WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS="true"):
            settings = WorkbenchApiSettings()

        self.assertIs(settings.allow_unsafe_local_mutations, True)
        self.assertIs(settings.log_imports_enabled, False)

    def test_env_parses_log_import_override(self) -> None:
        with isolated_settings_env(WORKBENCH_API_ALLOW_LOG_IMPORTS="false"):
            settings = WorkbenchApiSettings()

        self.assertIs(settings.allow_log_imports, False)
        self.assertIs(settings.log_imports_enabled, False)

        with isolated_settings_env(WORKBENCH_API_ALLOW_LOG_IMPORTS="true"):
            settings = WorkbenchApiSettings()

        self.assertIs(settings.allow_log_imports, True)
        self.assertIs(settings.log_imports_enabled, True)

    def test_env_parses_log_upload_size_limits(self) -> None:
        with isolated_settings_env(
            WORKBENCH_API_MAX_UPLOAD_SIZE="1024",
            WORKBENCH_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE="2048",
            WORKBENCH_API_MAX_LOG_ARCHIVE_MEMBER_COUNT="100",
            WORKBENCH_API_MAX_LOG_ARCHIVE_PATH_BYTES="4096",
            WORKBENCH_API_LOG_ARCHIVE_UPLOAD_CONCURRENCY="2",
        ):
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.max_upload_size, 1024)
        self.assertEqual(settings.max_log_archive_extracted_size, 2048)
        self.assertEqual(settings.effective_max_upload_size, 1024)
        self.assertEqual(settings.effective_max_log_archive_extracted_size, 2048)
        self.assertEqual(settings.max_log_archive_member_count, 100)
        self.assertEqual(settings.max_log_archive_path_bytes, 4096)
        self.assertEqual(settings.log_archive_upload_concurrency, 2)

    def test_env_parses_cors_origins_json_array(self) -> None:
        origins = [
            "https://workbench.example.com",
            "https://admin.example.com",
        ]

        with isolated_settings_env(
            WORKBENCH_API_CORS_ORIGINS=(
                '["https://workbench.example.com","https://admin.example.com"]'
            ),
        ):
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.cors_origins, origins)


if __name__ == "__main__":
    unittest.main()
