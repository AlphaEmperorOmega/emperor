import { useQuery } from "@tanstack/react-query";
import {
  type Capabilities,
  fetchCapabilities,
  fetchConfigSchema,
  fetchDatasets,
  fetchHealth,
  fetchMonitors,
  fetchModels,
  fetchPresets,
  fetchSearchSpace,
} from "@/lib/api";
import { viewerQueryKeys } from "@/lib/query-keys";

export const LOCAL_DEFAULT_CAPABILITIES: Capabilities = {
  authMode: "none",
  trainingEnabled: true,
  trainingCancellationCapability: "unsupported",
  logDeletionEnabled: true,
  configSnapshotsEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: true,
  maxUploadSize: null,
  dataSourcesEnabled: false,
  dataSources: [],
};

// Model registry, presets, datasets, monitors, config schema, and search space
// describe the backend's on-disk model definitions. They only change when
// backend code is edited (which restarts the dev server). A long stale window
// avoids refetching them every time a consumer remounts (e.g. switching
// workspaces) while still picking up backend changes on the next remount past
// this window.
const STATIC_METADATA_STALE_TIME_MS = 5 * 60_000;

export function useCapabilitiesQuery() {
  return useQuery({
    queryKey: viewerQueryKeys.capabilities(),
    queryFn: fetchCapabilities,
    retry: false,
    initialData: LOCAL_DEFAULT_CAPABILITIES,
  });
}

export function useViewerQueries(
  selectedModelType: string,
  selectedModel: string,
  selectedPreset: string,
  selectedTrainingPresets: readonly string[] = [],
  options: { includeSearchSpace?: boolean } = {},
) {
  const { includeSearchSpace = true } = options;
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
  const healthQuery = useQuery({
    queryKey: viewerQueryKeys.health(),
    queryFn: fetchHealth,
    retry: false,
    refetchInterval: 10000,
  });
  const capabilitiesQuery = useCapabilitiesQuery();

  const modelsQuery = useQuery({
    queryKey: viewerQueryKeys.models(),
    queryFn: fetchModels,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const presetsQuery = useQuery({
    queryKey: viewerQueryKeys.presets(selectedModelType, selectedModel),
    queryFn: () => fetchPresets(selectedIdentity),
    enabled: hasSelectedModel,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const datasetsQuery = useQuery({
    queryKey: viewerQueryKeys.datasets(selectedModelType, selectedModel),
    queryFn: () => fetchDatasets(selectedIdentity),
    enabled: hasSelectedModel,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const monitorsQuery = useQuery({
    queryKey: viewerQueryKeys.monitors(selectedModelType, selectedModel),
    queryFn: () => fetchMonitors(selectedIdentity),
    enabled: hasSelectedModel,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const schemaQuery = useQuery({
    queryKey: viewerQueryKeys.configSchema(
      selectedModelType,
      selectedModel,
      selectedPreset,
    ),
    queryFn: () => fetchConfigSchema(selectedIdentity, selectedPreset),
    enabled: hasSelectedModel && selectedPreset.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const searchSpaceQuery = useQuery({
    queryKey: viewerQueryKeys.searchSpace(
      selectedModelType,
      selectedModel,
      selectedPreset,
      searchSpacePresets,
    ),
    queryFn: () =>
      fetchSearchSpace(
        selectedIdentity,
        selectedPreset,
        searchSpacePresets,
      ),
    enabled: includeSearchSpace && hasSelectedModel && selectedPreset.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  return {
    healthQuery,
    capabilitiesQuery,
    modelsQuery,
    presetsQuery,
    datasetsQuery,
    monitorsQuery,
    schemaQuery,
    searchSpaceQuery,
  };
}
