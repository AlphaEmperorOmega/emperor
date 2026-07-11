import { useQuery } from "@tanstack/react-query";
import {
  fetchConfigSchema,
  fetchDatasets,
  fetchMonitors,
  fetchModels,
  fetchPresets,
  fetchSearchSpace,
} from "@/lib/api";
import { workbenchQueryKeys } from "@/lib/query-keys";

// Model registry, presets, datasets, monitors, config schema, and search space
// describe the backend's on-disk model definitions. They only change when
// backend code is edited (which restarts the dev server). A long stale window
// avoids refetching them every time a consumer remounts (e.g. switching
// workspaces) while still picking up backend changes on the next remount past
// this window.
const STATIC_METADATA_STALE_TIME_MS = 5 * 60_000;

export function useConfigSchemaQuery(
  selectedModelType: string,
  selectedModel: string,
  selectedPreset: string,
  options: { enabled?: boolean } = {},
) {
  const { enabled = true } = options;
  const selectedIdentity = {
    modelType: selectedModelType,
    model: selectedModel,
  };
  return useQuery({
    queryKey: workbenchQueryKeys.configSchema(
      selectedModelType,
      selectedModel,
      selectedPreset,
    ),
    queryFn: ({ signal }) =>
      fetchConfigSchema(selectedIdentity, selectedPreset, { signal }),
    enabled:
      enabled &&
      selectedModelType.length > 0 &&
      selectedModel.length > 0 &&
      selectedPreset.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });
}

export function useWorkbenchQueries(
  selectedModelType: string,
  selectedModel: string,
  selectedPreset: string,
  selectedTrainingPresets: readonly string[] = [],
  options: {
    includeSearchSpace?: boolean;
    protectedReadsEnabled?: boolean;
  } = {},
) {
  const {
    includeSearchSpace = true,
    protectedReadsEnabled = true,
  } = options;
  const selectedIdentity = {
    modelType: selectedModelType,
    model: selectedModel,
  };
  const hasSelectedModel =
    selectedModelType.length > 0 && selectedModel.length > 0;
  const searchSpacePresets =
    selectedTrainingPresets.length > 0
      ? selectedTrainingPresets
      : selectedPreset
        ? [selectedPreset]
        : [];
  const modelsQuery = useQuery({
    queryKey: workbenchQueryKeys.models(),
    queryFn: ({ signal }) => fetchModels({ signal }),
    enabled: protectedReadsEnabled,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const presetsQuery = useQuery({
    queryKey: workbenchQueryKeys.presets(selectedModelType, selectedModel),
    queryFn: ({ signal }) => fetchPresets(selectedIdentity, { signal }),
    enabled: protectedReadsEnabled && hasSelectedModel,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const datasetsQuery = useQuery({
    queryKey: workbenchQueryKeys.datasets(selectedModelType, selectedModel),
    queryFn: ({ signal }) => fetchDatasets(selectedIdentity, { signal }),
    enabled: protectedReadsEnabled && hasSelectedModel,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const monitorsQuery = useQuery({
    queryKey: workbenchQueryKeys.monitors(selectedModelType, selectedModel),
    queryFn: ({ signal }) => fetchMonitors(selectedIdentity, { signal }),
    enabled: protectedReadsEnabled && hasSelectedModel,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const schemaQuery = useConfigSchemaQuery(
    selectedModelType,
    selectedModel,
    selectedPreset,
    { enabled: protectedReadsEnabled },
  );

  const searchSpaceQuery = useQuery({
    queryKey: workbenchQueryKeys.searchSpace(
      selectedModelType,
      selectedModel,
      selectedPreset,
      searchSpacePresets,
    ),
    queryFn: ({ signal }) =>
      fetchSearchSpace(
        selectedIdentity,
        selectedPreset,
        searchSpacePresets,
        { signal },
      ),
    enabled:
      protectedReadsEnabled &&
      includeSearchSpace &&
      hasSelectedModel &&
      selectedPreset.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  return {
    modelsQuery,
    presetsQuery,
    datasetsQuery,
    monitorsQuery,
    schemaQuery,
    searchSpaceQuery,
  };
}
