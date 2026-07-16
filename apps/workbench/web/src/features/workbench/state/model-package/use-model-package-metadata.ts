import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchModels,
  type ModelIdentity,
} from "@/lib/api/model-catalog";
import {
  fetchConfigSchema,
  fetchDatasets,
  fetchMonitors,
  fetchPresets,
  fetchSearchSpace,
} from "@/lib/api/model-metadata-client";
import {
  type ConfigField,
  type DatasetGroup,
  type MonitorOption,
  type Preset,
  type SearchAxis,
} from "@/lib/api/models";
import { workbenchQueryKeys } from "@/lib/query-keys";

const STATIC_METADATA_STALE_TIME_MS = 5 * 60_000;
const EMPTY_MODELS: ModelIdentity[] = [];
const EMPTY_PRESETS: Preset[] = [];
const EMPTY_DATASET_GROUPS: DatasetGroup[] = [];
const EMPTY_MONITORS: MonitorOption[] = [];
const EMPTY_FIELDS: ConfigField[] = [];
const EMPTY_SEARCH_AXES: SearchAxis[] = [];

export type ModelPackageMetadataSelection = Readonly<{
  modelPackage: ModelIdentity;
  preset: string;
  searchPresets?: readonly string[];
}>;

type MetadataReadiness = Readonly<{
  isLoading: boolean;
  isReady: boolean;
  isError: boolean;
  error: Error | null;
}>;

function readiness(query: {
  isLoading: boolean;
  isSuccess: boolean;
  isError: boolean;
  error: Error | null;
}): MetadataReadiness {
  return {
    isLoading: query.isLoading,
    isReady: query.isSuccess,
    isError: query.isError,
    error: query.error,
  };
}

function useRuntimeDefaultsSchemaQuery(
  selection: ModelPackageMetadataSelection,
  enabled: boolean,
) {
  const { modelType, model } = selection.modelPackage;
  return useQuery({
    queryKey: workbenchQueryKeys.configSchema(
      modelType,
      model,
      selection.preset,
    ),
    queryFn: ({ signal }) =>
      fetchConfigSchema(selection.modelPackage, selection.preset, { signal }),
    enabled:
      enabled &&
      modelType.length > 0 &&
      model.length > 0 &&
      selection.preset.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });
}

export function useRuntimeDefaultsSchema(
  selection: ModelPackageMetadataSelection,
  options: { enabled?: boolean } = {},
) {
  const query = useRuntimeDefaultsSchemaQuery(
    selection,
    options.enabled ?? true,
  );
  return useMemo(
    () => ({
      fields: query.data?.fields ?? EMPTY_FIELDS,
      ...readiness(query),
    }),
    [query],
  );
}

export function useModelPackageMetadata(
  selection: ModelPackageMetadataSelection,
  options: {
    includeSearchMetadata?: boolean;
    normalizePresetSelection?: boolean;
    protectedReadsEnabled?: boolean;
  } = {},
) {
  const {
    includeSearchMetadata = true,
    normalizePresetSelection = false,
    protectedReadsEnabled = true,
  } = options;
  const { modelType, model } = selection.modelPackage;
  const hasModelPackage = modelType.length > 0 && model.length > 0;
  const modelPackagesQuery = useQuery({
    queryKey: workbenchQueryKeys.models(),
    queryFn: ({ signal }) => fetchModels({ signal }),
    enabled: protectedReadsEnabled,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });
  const presetsQuery = useQuery({
    queryKey: workbenchQueryKeys.presets(modelType, model),
    queryFn: ({ signal }) => fetchPresets(selection.modelPackage, { signal }),
    enabled: protectedReadsEnabled && hasModelPackage,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });
  const availablePresetNames =
    presetsQuery.data?.presets.map((preset) => preset.name) ?? [];
  const requestedSearchPresets = selection.searchPresets ?? [];
  const resolvedPreset =
    normalizePresetSelection && presetsQuery.isSuccess
      ? availablePresetNames.includes(selection.preset)
        ? selection.preset
        : requestedSearchPresets.find((preset) =>
              availablePresetNames.includes(preset),
            ) ??
          availablePresetNames[0] ??
          ""
      : selection.preset;
  const resolvedSearchPresets =
    normalizePresetSelection && presetsQuery.isSuccess
      ? requestedSearchPresets.filter(
          (preset, index) =>
            availablePresetNames.includes(preset) &&
            requestedSearchPresets.indexOf(preset) === index,
        )
      : requestedSearchPresets;
  const effectiveSelection = {
    ...selection,
    preset: resolvedPreset,
    searchPresets: resolvedSearchPresets,
  };
  const searchPresets =
    resolvedSearchPresets.length > 0
      ? resolvedSearchPresets
      : resolvedPreset
        ? [resolvedPreset]
        : [];
  const datasetsQuery = useQuery({
    queryKey: workbenchQueryKeys.datasets(modelType, model),
    queryFn: ({ signal }) => fetchDatasets(selection.modelPackage, { signal }),
    enabled: protectedReadsEnabled && hasModelPackage,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });
  const monitorsQuery = useQuery({
    queryKey: workbenchQueryKeys.monitors(modelType, model),
    queryFn: ({ signal }) => fetchMonitors(selection.modelPackage, { signal }),
    enabled: protectedReadsEnabled && hasModelPackage,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });
  const runtimeDefaultsQuery = useRuntimeDefaultsSchemaQuery(
    effectiveSelection,
    protectedReadsEnabled,
  );
  const searchMetadataQuery = useQuery({
    queryKey: workbenchQueryKeys.searchSpace(
      modelType,
      model,
      resolvedPreset,
      searchPresets,
    ),
    queryFn: ({ signal }) =>
      fetchSearchSpace(
        selection.modelPackage,
        resolvedPreset,
        searchPresets,
        { signal },
      ),
    enabled:
      protectedReadsEnabled &&
      includeSearchMetadata &&
      hasModelPackage &&
      resolvedPreset.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  return useMemo(
    () => ({
      modelPackages: {
        records: modelPackagesQuery.data?.models ?? EMPTY_MODELS,
        ...readiness(modelPackagesQuery),
      },
      presets: {
        records: presetsQuery.data?.presets ?? EMPTY_PRESETS,
        ...readiness(presetsQuery),
      },
      datasetMetadata: {
        groups: datasetsQuery.data?.datasetGroups ?? EMPTY_DATASET_GROUPS,
        defaultExperimentTask:
          datasetsQuery.data?.defaultExperimentTask ?? "",
        ...readiness(datasetsQuery),
      },
      monitorMetadata: {
        records: monitorsQuery.data?.monitors ?? EMPTY_MONITORS,
        ...readiness(monitorsQuery),
      },
      runtimeDefaults: {
        fields: runtimeDefaultsQuery.data?.fields ?? EMPTY_FIELDS,
        ...readiness(runtimeDefaultsQuery),
      },
      searchMetadata: {
        axes: searchMetadataQuery.data?.axes ?? EMPTY_SEARCH_AXES,
        ...readiness(searchMetadataQuery),
      },
    }),
    [
      datasetsQuery,
      modelPackagesQuery,
      monitorsQuery,
      presetsQuery,
      runtimeDefaultsQuery,
      searchMetadataQuery,
    ],
  );
}
