"""Runtime configuration for the Viewer backend."""

from functools import lru_cache

from viewer.backend.settings import LOCAL_FRONTEND_ORIGINS, ViewerApiSettings


@lru_cache(maxsize=1)
def get_viewer_api_settings() -> ViewerApiSettings:
    return ViewerApiSettings()


__all__ = ["LOCAL_FRONTEND_ORIGINS", "ViewerApiSettings", "get_viewer_api_settings"]
