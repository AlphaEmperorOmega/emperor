from __future__ import annotations

import os
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from viewer.backend.core.config import (
    DEFAULT_SNAPSHOTS_ROOT,
    LOCAL_FRONTEND_ORIGINS,
    ViewerApiSettings,
    get_viewer_api_settings,
)
from viewer.backend.core.limits import (
    DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
    DEFAULT_MAX_LOG_ARCHIVE_UPLOAD_SIZE,
)

SETTINGS_ENV_NAMES = (
    "VIEWER_API_AUTH_MODE",
    "VIEWER_API_TOKEN",
    "VIEWER_API_CORS_ORIGINS",
    "VIEWER_API_SNAPSHOTS_ROOT",
    "VIEWER_API_ALLOW_UNSAFE_LOCAL_MUTATIONS",
    "VIEWER_API_ALLOW_LOG_IMPORTS",
    "VIEWER_API_MAX_UPLOAD_SIZE",
    "VIEWER_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE",
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


class ViewerApiSettingsTests(unittest.TestCase):
    def test_legacy_settings_module_reexports_canonical_settings(self) -> None:
        from viewer.backend import settings as legacy_settings

        self.assertIs(legacy_settings.LOCAL_FRONTEND_ORIGINS, LOCAL_FRONTEND_ORIGINS)
        self.assertIs(legacy_settings.ViewerApiSettings, ViewerApiSettings)
        self.assertIs(
            legacy_settings.get_viewer_api_settings,
            get_viewer_api_settings,
        )

    def test_defaults_keep_local_development_unauthenticated(self) -> None:
        with isolated_settings_env():
            settings = ViewerApiSettings()

        self.assertEqual(settings.auth_mode, "none")
        self.assertIsNone(settings.token)
        self.assertIs(settings.allow_unsafe_local_mutations, False)
        self.assertIs(settings.log_imports_enabled, True)
        self.assertIsNone(settings.max_upload_size)
        self.assertIsNone(settings.max_log_archive_extracted_size)
        self.assertIsNone(settings.effective_max_upload_size)
        self.assertIsNone(settings.effective_max_log_archive_extracted_size)

    def test_bearer_mode_defaults_disable_log_imports(self) -> None:
        settings = ViewerApiSettings(auth_mode="bearer", token="secret-token")

        self.assertIs(settings.log_imports_enabled, False)

    def test_hosted_or_explicit_log_imports_use_default_size_limits(self) -> None:
        bearer_settings = ViewerApiSettings(
            auth_mode="bearer",
            token="secret-token",
            allow_log_imports=True,
        )
        local_override_settings = ViewerApiSettings(allow_log_imports=True)

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

    def test_code_can_explicitly_disable_log_import_size_limits(self) -> None:
        settings = ViewerApiSettings(
            max_upload_size=None,
            max_log_archive_extracted_size=None,
        )

        self.assertIsNone(settings.effective_max_upload_size)
        self.assertIsNone(settings.effective_max_log_archive_extracted_size)

    def test_defaults_keep_local_development_cors_origins(self) -> None:
        with isolated_settings_env():
            settings = ViewerApiSettings()

        self.assertEqual(settings.cors_origins, LOCAL_FRONTEND_ORIGINS)
        self.assertIsNot(settings.cors_origins, LOCAL_FRONTEND_ORIGINS)

    def test_default_snapshots_root_is_owned_by_viewer(self) -> None:
        with isolated_settings_env():
            settings = ViewerApiSettings()

        self.assertEqual(settings.snapshots_root, DEFAULT_SNAPSHOTS_ROOT)
        self.assertEqual(Path(settings.snapshots_root).name, "snapshots")
        self.assertEqual(Path(settings.snapshots_root).parent.name, "viewer")

    def test_bearer_mode_accepts_non_empty_token(self) -> None:
        settings = ViewerApiSettings(auth_mode="bearer", token="secret-token")

        self.assertEqual(settings.auth_mode, "bearer")
        self.assertEqual(settings.token, "secret-token")

    def test_bearer_mode_rejects_missing_or_empty_token(self) -> None:
        for token in (None, "", "   "):
            with self.subTest(token=token):
                kwargs = {"auth_mode": "bearer"}
                if token is not None:
                    kwargs["token"] = token

                with self.assertRaises(ValidationError):
                    ViewerApiSettings(**kwargs)

    def test_invalid_auth_mode_rejects(self) -> None:
        with self.assertRaises(ValidationError):
            ViewerApiSettings(auth_mode="basic", token="secret-token")

    def test_env_parses_auth_mode_and_token(self) -> None:
        with isolated_settings_env(
            VIEWER_API_AUTH_MODE="bearer",
            VIEWER_API_TOKEN="env-secret",
        ):
            settings = ViewerApiSettings()

        self.assertEqual(settings.auth_mode, "bearer")
        self.assertEqual(settings.token, "env-secret")

    def test_env_parses_unsafe_local_mutation_opt_in(self) -> None:
        with isolated_settings_env(VIEWER_API_ALLOW_UNSAFE_LOCAL_MUTATIONS="true"):
            settings = ViewerApiSettings()

        self.assertIs(settings.allow_unsafe_local_mutations, True)
        self.assertIs(settings.log_imports_enabled, True)

    def test_env_parses_log_import_override(self) -> None:
        with isolated_settings_env(VIEWER_API_ALLOW_LOG_IMPORTS="false"):
            settings = ViewerApiSettings()

        self.assertIs(settings.allow_log_imports, False)
        self.assertIs(settings.log_imports_enabled, False)

        with isolated_settings_env(VIEWER_API_ALLOW_LOG_IMPORTS="true"):
            settings = ViewerApiSettings()

        self.assertIs(settings.allow_log_imports, True)
        self.assertIs(settings.log_imports_enabled, True)

    def test_env_parses_log_upload_size_limits(self) -> None:
        with isolated_settings_env(
            VIEWER_API_MAX_UPLOAD_SIZE="1024",
            VIEWER_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE="2048",
        ):
            settings = ViewerApiSettings()

        self.assertEqual(settings.max_upload_size, 1024)
        self.assertEqual(settings.max_log_archive_extracted_size, 2048)
        self.assertEqual(settings.effective_max_upload_size, 1024)
        self.assertEqual(settings.effective_max_log_archive_extracted_size, 2048)

    def test_env_parses_cors_origins_json_array(self) -> None:
        origins = [
            "https://viewer.example.com",
            "https://admin.example.com",
        ]

        with isolated_settings_env(
            VIEWER_API_CORS_ORIGINS=(
                '["https://viewer.example.com","https://admin.example.com"]'
            ),
        ):
            settings = ViewerApiSettings()

        self.assertEqual(settings.cors_origins, origins)


if __name__ == "__main__":
    unittest.main()
