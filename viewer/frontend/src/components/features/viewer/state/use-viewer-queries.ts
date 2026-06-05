import { useQuery } from "@tanstack/react-query";
import {
  fetchConfigSchema,
  fetchDatasets,
  fetchHealth,
  fetchMonitors,
  fetchModels,
  fetchPresets,
  fetchSearchSpace,
} from "@/lib/api";
import { viewerQueryKeys } from "@/lib/query-keys";

export function useViewerQueries(selectedModel: string, selectedPreset: string) {
  const healthQuery = useQuery({
    queryKey: viewerQueryKeys.health(),
    queryFn: fetchHealth,
    retry: false,
    refetchInterval: 10000,
  });

  const modelsQuery = useQuery({
    queryKey: viewerQueryKeys.models(),
    queryFn: fetchModels,
    retry: false,
  });

  const presetsQuery = useQuery({
    queryKey: viewerQueryKeys.presets(selectedModel),
    queryFn: () => fetchPresets(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
  });

  const datasetsQuery = useQuery({
    queryKey: viewerQueryKeys.datasets(selectedModel),
    queryFn: () => fetchDatasets(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
  });

  const monitorsQuery = useQuery({
    queryKey: viewerQueryKeys.monitors(selectedModel),
    queryFn: () => fetchMonitors(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
  });

  const schemaQuery = useQuery({
    queryKey: viewerQueryKeys.configSchema(selectedModel, selectedPreset),
    queryFn: () => fetchConfigSchema(selectedModel, selectedPreset),
    enabled: selectedModel.length > 0 && selectedPreset.length > 0,
    retry: false,
  });

  const searchSpaceQuery = useQuery({
    queryKey: viewerQueryKeys.searchSpace(selectedModel, selectedPreset),
    queryFn: () => fetchSearchSpace(selectedModel, selectedPreset),
    enabled: selectedModel.length > 0 && selectedPreset.length > 0,
    retry: false,
  });

  return {
    healthQuery,
    modelsQuery,
    presetsQuery,
    datasetsQuery,
    monitorsQuery,
    schemaQuery,
    searchSpaceQuery,
  };
}
