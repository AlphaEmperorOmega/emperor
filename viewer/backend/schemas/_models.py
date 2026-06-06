"""Model discovery and configuration schemas."""

from __future__ import annotations

from viewer.backend.schemas._base import ApiResponseModel, ConfigValue


class ModelsResponse(ApiResponseModel):
    models: list[str]


class PresetResponse(ApiResponseModel):
    name: str
    label: str
    description: str


class PresetsResponse(ApiResponseModel):
    model: str
    presets: list[PresetResponse]


class DatasetResponse(ApiResponseModel):
    name: str
    label: str
    inputDim: int
    outputDim: int


class DatasetsResponse(ApiResponseModel):
    model: str
    datasets: list[DatasetResponse]


class MonitorOptionResponse(ApiResponseModel):
    name: str
    label: str
    description: str
    kinds: list[str]
    defaultEnabled: bool = False


class MonitorsResponse(ApiResponseModel):
    model: str
    monitors: list[MonitorOptionResponse]


class ConfigFieldResponse(ApiResponseModel):
    key: str
    configKey: str
    flag: str
    label: str
    section: str
    type: str
    default: ConfigValue
    nullable: bool
    choices: list[ConfigValue]
    locked: bool = False
    lockedValue: ConfigValue = None
    lockedReason: str = ""


class ConfigSchemaResponse(ApiResponseModel):
    model: str
    fields: list[ConfigFieldResponse]


class SearchAxisResponse(ApiResponseModel):
    key: str
    configKey: str
    searchKey: str
    label: str
    section: str
    type: str
    values: list[ConfigValue]
    locked: bool = False
    lockedValue: ConfigValue = None
    lockedReason: str = ""


class SearchSpaceResponse(ApiResponseModel):
    model: str
    preset: str | None = None
    axes: list[SearchAxisResponse]
