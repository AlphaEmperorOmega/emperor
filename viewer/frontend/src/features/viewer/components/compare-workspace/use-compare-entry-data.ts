import { useMemo } from "react";
import { useQueries } from "@tanstack/react-query";
import {
  fetchConfigSchema,
  fetchDatasets,
  fetchMonitors,
  fetchPresets,
  inspectModel,
} from "@/lib/api";
import { viewerQueryKeys } from "@/lib/query-keys";
import type { CompareEntry, CompareEntryData } from "./derive";

function entryIdentity(entry: CompareEntry) {
  return { modelType: entry.modelType, model: entry.model };
}

export type CompareEntryQueryState = {
  entryData: CompareEntryData[];
  presetNamesByEntry: Map<string, string[]>;
  presetDefaultsSignature: string;
};

export function useCompareEntryData(
  entries: CompareEntry[],
): CompareEntryQueryState {
  const presetQueries = useQueries({
    queries: entries.map((entry) => ({
      queryKey: viewerQueryKeys.presets(entry.modelType, entry.model),
      queryFn: () => fetchPresets(entryIdentity(entry)),
      enabled: entry.model.length > 0,
      retry: false,
    })),
  });
  const datasetQueries = useQueries({
    queries: entries.map((entry) => ({
      queryKey: viewerQueryKeys.datasets(entry.modelType, entry.model),
      queryFn: () => fetchDatasets(entryIdentity(entry)),
      enabled: entry.model.length > 0,
      retry: false,
    })),
  });
  const monitorQueries = useQueries({
    queries: entries.map((entry) => ({
      queryKey: viewerQueryKeys.monitors(entry.modelType, entry.model),
      queryFn: () => fetchMonitors(entryIdentity(entry)),
      enabled: entry.model.length > 0,
      retry: false,
    })),
  });
  const schemaQueries = useQueries({
    queries: entries.map((entry) => ({
      queryKey: viewerQueryKeys.configSchema(
        entry.modelType,
        entry.model,
        entry.preset,
      ),
      queryFn: () => fetchConfigSchema(entryIdentity(entry), entry.preset),
      enabled: entry.model.length > 0 && entry.preset.length > 0,
      retry: false,
    })),
  });
  const inspectTargets = entries.map((entry, index) => ({
    entry,
    dataset: datasetQueries[index]?.data?.datasets[0]?.name ?? "",
  }));
  const inspectQueries = useQueries({
    queries: inspectTargets.map((targetEntry) => ({
      queryKey: viewerQueryKeys.comparisonInspection(
        targetEntry.entry.modelType,
        targetEntry.entry.model,
        targetEntry.entry.preset,
        targetEntry.dataset,
      ),
      queryFn: () =>
        inspectModel({
          modelType: targetEntry.entry.modelType,
          model: targetEntry.entry.model,
          preset: targetEntry.entry.preset,
          dataset: targetEntry.dataset,
          overrides: {},
        }),
      enabled:
        targetEntry.entry.model.length > 0 &&
        targetEntry.entry.preset.length > 0 &&
        targetEntry.dataset.length > 0,
      retry: false,
    })),
  });

  const presetNamesByEntry = useMemo(
    () =>
      new Map(
        entries.map((entry, index) => [
          entry.id,
          presetQueries[index]?.data?.presets.map((preset) => preset.name) ?? [],
        ]),
      ),
    [entries, presetQueries],
  );
  const presetDefaultsSignature = Array.from(presetNamesByEntry.entries())
    .map(([id, names]) => `${id}:${names.join("|")}`)
    .join(";");

  const entryData = entries.map<CompareEntryData>((entry, index) => {
    const queryError =
      presetQueries[index]?.error ??
      datasetQueries[index]?.error ??
      monitorQueries[index]?.error ??
      schemaQueries[index]?.error ??
      inspectQueries[index]?.error;
    return {
      entry,
      presets: presetQueries[index]?.data?.presets ?? [],
      datasets: datasetQueries[index]?.data?.datasets ?? [],
      monitors: monitorQueries[index]?.data?.monitors ?? [],
      fields: schemaQueries[index]?.data?.fields ?? [],
      inspection: inspectQueries[index]?.data,
      dataset: inspectTargets[index]?.dataset ?? "",
      isLoading: Boolean(
        presetQueries[index]?.isLoading ||
          datasetQueries[index]?.isLoading ||
          monitorQueries[index]?.isLoading ||
          schemaQueries[index]?.isLoading ||
          inspectQueries[index]?.isLoading,
      ),
      error: queryError,
    };
  });

  return { entryData, presetNamesByEntry, presetDefaultsSignature };
}
