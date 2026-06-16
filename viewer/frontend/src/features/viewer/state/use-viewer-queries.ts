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
  logDeletionEnabled: true,
  configSnapshotsEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: false,
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

export function useViewerQueries(selectedModel: string, selectedPreset: string) {
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
    queryKey: viewerQueryKeys.presets(selectedModel),
    queryFn: () => fetchPresets(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const datasetsQuery = useQuery({
    queryKey: viewerQueryKeys.datasets(selectedModel),
    queryFn: () => fetchDatasets(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const monitorsQuery = useQuery({
    queryKey: viewerQueryKeys.monitors(selectedModel),
    queryFn: () => fetchMonitors(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const schemaQuery = useQuery({
    queryKey: viewerQueryKeys.configSchema(selectedModel, selectedPreset),
    queryFn: () => fetchConfigSchema(selectedModel, selectedPreset),
    enabled: selectedModel.length > 0 && selectedPreset.length > 0,
    retry: false,
    staleTime: STATIC_METADATA_STALE_TIME_MS,
  });

  const searchSpaceQuery = useQuery({
    queryKey: viewerQueryKeys.searchSpace(selectedModel, selectedPreset),
    queryFn: () => fetchSearchSpace(selectedModel, selectedPreset),
    enabled: selectedModel.length > 0 && selectedPreset.length > 0,
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
