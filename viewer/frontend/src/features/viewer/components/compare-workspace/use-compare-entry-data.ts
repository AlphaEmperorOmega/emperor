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

type QueryTarget<TValue> = {
  key: readonly unknown[];
  value: TValue;
};

function entryIdentity(entry: CompareEntry) {
  return { modelType: entry.modelType, model: entry.model };
}

function queryKeySignature(key: readonly unknown[]) {
  return JSON.stringify(key);
}

function uniqueQueryTargets<TValue>(targets: QueryTarget<TValue>[]) {
  const indexes: number[] = [];
  const indexesByKey = new Map<string, number>();
  const unique: QueryTarget<TValue>[] = [];

  targets.forEach((target) => {
    const key = queryKeySignature(target.key);
    const existingIndex = indexesByKey.get(key);
    if (existingIndex !== undefined) {
      indexes.push(existingIndex);
      return;
    }

    const nextIndex = unique.length;
    indexesByKey.set(key, nextIndex);
    indexes.push(nextIndex);
    unique.push(target);
  });

  return { indexes, unique };
}

function mapQueryResults<TResult>(
  results: TResult[],
  indexes: number[],
): Array<TResult | undefined> {
  return indexes.map((index) => results[index]);
}

export type CompareEntryQueryState = {
  entryData: CompareEntryData[];
  presetNamesByEntry: Map<string, string[]>;
  presetDefaultsSignature: string;
};

export function useCompareEntryData(
  entries: CompareEntry[],
): CompareEntryQueryState {
  const presetTargets = useMemo(
    () =>
      uniqueQueryTargets(
        entries.map((entry) => ({
          key: viewerQueryKeys.presets(entry.modelType, entry.model),
          value: entry,
        })),
      ),
    [entries],
  );
  const presetQueries = useQueries({
    queries: presetTargets.unique.map(({ value: entry }) => ({
      queryKey: viewerQueryKeys.presets(entry.modelType, entry.model),
      queryFn: () => fetchPresets(entryIdentity(entry)),
      enabled: entry.model.length > 0,
      retry: false,
    })),
  });
  const entryPresetQueries = mapQueryResults(presetQueries, presetTargets.indexes);

  const datasetTargets = useMemo(
    () =>
      uniqueQueryTargets(
        entries.map((entry) => ({
          key: viewerQueryKeys.datasets(entry.modelType, entry.model),
          value: entry,
        })),
      ),
    [entries],
  );
  const datasetQueries = useQueries({
    queries: datasetTargets.unique.map(({ value: entry }) => ({
      queryKey: viewerQueryKeys.datasets(entry.modelType, entry.model),
      queryFn: () => fetchDatasets(entryIdentity(entry)),
      enabled: entry.model.length > 0,
      retry: false,
    })),
  });
  const entryDatasetQueries = mapQueryResults(datasetQueries, datasetTargets.indexes);

  const monitorTargets = useMemo(
    () =>
      uniqueQueryTargets(
        entries.map((entry) => ({
          key: viewerQueryKeys.monitors(entry.modelType, entry.model),
          value: entry,
        })),
      ),
    [entries],
  );
  const monitorQueries = useQueries({
    queries: monitorTargets.unique.map(({ value: entry }) => ({
      queryKey: viewerQueryKeys.monitors(entry.modelType, entry.model),
      queryFn: () => fetchMonitors(entryIdentity(entry)),
      enabled: entry.model.length > 0,
      retry: false,
    })),
  });
  const entryMonitorQueries = mapQueryResults(monitorQueries, monitorTargets.indexes);

  const schemaTargets = useMemo(
    () =>
      uniqueQueryTargets(
        entries.map((entry) => ({
          key: viewerQueryKeys.configSchema(
            entry.modelType,
            entry.model,
            entry.preset,
          ),
          value: entry,
        })),
      ),
    [entries],
  );
  const schemaQueries = useQueries({
    queries: schemaTargets.unique.map(({ value: entry }) => ({
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
  const entrySchemaQueries = mapQueryResults(schemaQueries, schemaTargets.indexes);
  const inspectTargets = entries.map((entry, index) => ({
    entry,
    dataset: entryDatasetQueries[index]?.data?.datasets[0]?.name ?? "",
  }));
  const uniqueInspectTargets = useMemo(
    () =>
      uniqueQueryTargets(
        inspectTargets.map((targetEntry) => ({
          key: viewerQueryKeys.comparisonInspection(
            targetEntry.entry.modelType,
            targetEntry.entry.model,
            targetEntry.entry.preset,
            targetEntry.dataset,
          ),
          value: targetEntry,
        })),
      ),
    [inspectTargets],
  );
  const inspectQueries = useQueries({
    queries: uniqueInspectTargets.unique.map(({ value: targetEntry }) => ({
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
  const entryInspectQueries = mapQueryResults(
    inspectQueries,
    uniqueInspectTargets.indexes,
  );

  const presetNamesByEntry = useMemo(
    () =>
      new Map(
        entries.map((entry, index) => [
          entry.id,
          entryPresetQueries[index]?.data?.presets.map((preset) => preset.name) ?? [],
        ]),
      ),
    [entries, entryPresetQueries],
  );
  const presetDefaultsSignature = Array.from(presetNamesByEntry.entries())
    .map(([id, names]) => `${id}:${names.join("|")}`)
    .join(";");

  const entryData = entries.map<CompareEntryData>((entry, index) => {
    const queryError =
      entryPresetQueries[index]?.error ??
      entryDatasetQueries[index]?.error ??
      entryMonitorQueries[index]?.error ??
      entrySchemaQueries[index]?.error ??
      entryInspectQueries[index]?.error;
    return {
      entry,
      presets: entryPresetQueries[index]?.data?.presets ?? [],
      datasets: entryDatasetQueries[index]?.data?.datasets ?? [],
      monitors: entryMonitorQueries[index]?.data?.monitors ?? [],
      fields: entrySchemaQueries[index]?.data?.fields ?? [],
      inspection: entryInspectQueries[index]?.data,
      dataset: inspectTargets[index]?.dataset ?? "",
      isLoading: Boolean(
        entryPresetQueries[index]?.isLoading ||
          entryDatasetQueries[index]?.isLoading ||
          entryMonitorQueries[index]?.isLoading ||
          entrySchemaQueries[index]?.isLoading ||
          entryInspectQueries[index]?.isLoading,
      ),
      error: queryError,
    };
  });

  return { entryData, presetNamesByEntry, presetDefaultsSignature };
}
