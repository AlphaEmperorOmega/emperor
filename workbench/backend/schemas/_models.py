"""Model discovery and configuration schemas."""

from __future__ import annotations

from pydantic import Field

from workbench.backend.schemas._base import ApiResponseModel, ConfigValue


class ModelIdentityResponse(ApiResponseModel):
    modelType: str
    model: str


class ModelsResponse(ApiResponseModel):
    models: list[ModelIdentityResponse]


class PresetResponse(ApiResponseModel):
    name: str
    label: str
    description: str


class PresetsResponse(ApiResponseModel):
    modelType: str
    model: str
    presets: list[PresetResponse]


class DatasetResponse(ApiResponseModel):
    name: str
    label: str
    inputDim: int
    outputDim: int


class DatasetGroupResponse(ApiResponseModel):
    experimentTask: str
    label: str
    datasets: list[DatasetResponse]


class DatasetsResponse(ApiResponseModel):
    modelType: str
    model: str
    defaultExperimentTask: str
    datasetGroups: list[DatasetGroupResponse]


class MonitorOptionResponse(ApiResponseModel):
    name: str
    label: str
    description: str
    kinds: list[str]
    defaultEnabled: bool = False


class MonitorsResponse(ApiResponseModel):
    modelType: str
    model: str
    monitors: list[MonitorOptionResponse]


class ConfigFieldResponse(ApiResponseModel):
    key: str
    configKey: str
    flag: str
    label: str
    section: str
    sectionPath: list[str] = Field(min_length=1)
    description: str = ""
    type: str
    default: ConfigValue
    nullable: bool
    choices: list[ConfigValue]
    maximum: int | float | None = None
    locked: bool = False
    lockedValue: ConfigValue = None
    lockedReason: str = ""


class ConfigSchemaResponse(ApiResponseModel):
    modelType: str
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
    lockedByPresets: list[str] = Field(default_factory=list)
    lockReasons: list[str] = Field(default_factory=list)


class SearchSpaceResponse(ApiResponseModel):
    modelType: str
    model: str
    preset: str | None = None
    axes: list[SearchAxisResponse]
