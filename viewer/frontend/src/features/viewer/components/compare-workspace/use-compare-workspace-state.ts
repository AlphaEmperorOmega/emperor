import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useCompareTargetState,
} from "@/features/viewer/providers/viewer-providers";
import {
  buildCompareModelOptions,
  changedConfigRows,
  createCompareEntry,
  isReadyEntry,
  MAX_COMPARE_TARGETS,
  statRows,
  type CompareEntry,
  type CompareEntryData,
  type CompareModelOption,
  type ConfigDiffRow,
  type StatRow,
} from "./derive";
import { useCompareEntryData } from "./use-compare-entry-data";

const emptyModels: string[] = [];

export type CompareWorkspaceState = {
  catalog: {
    isLoading: boolean;
    isError: boolean;
    error: unknown;
  };
  modelOptions: CompareModelOption[];
  entries: CompareEntryData[];
  readyEntryCount: number;
  changedStats: StatRow[];
  configRows: ConfigDiffRow[];
  canAddEntry: boolean;
  canResetEntries: boolean;
  canRemoveEntry: boolean;
  addEntry: () => void;
  resetEntries: () => void;
  removeEntry: (id: string) => void;
  updateEntry: (id: string, patch: Partial<CompareEntry>) => void;
  applyAsTarget: (entry: CompareEntry) => void;
};

export function useCompareWorkspaceState({
  onUseTarget,
}: {
  onUseTarget: () => void;
}): CompareWorkspaceState {
  const target = useCompareTargetState();
  const models = target.catalog.models ?? emptyModels;
  const nextId = useRef(1);
  const [entries, setEntries] = useState<CompareEntry[]>([]);

  const allocateId = useCallback(() => {
    const id = `compare-${nextId.current}`;
    nextId.current += 1;
    return id;
  }, []);

  useEffect(() => {
    if (entries.length > 0 || models.length === 0) {
      return;
    }
    const primaryModel =
      target.selectedModel && models.includes(target.selectedModel)
        ? target.selectedModel
        : models[0];
    const secondaryModel = models.find((model) => model !== primaryModel) ?? models[1];
    const nextEntries = [
      createCompareEntry(allocateId(), primaryModel, target.selectedPreset),
    ];
    if (secondaryModel) {
      nextEntries.push(createCompareEntry(allocateId(), secondaryModel, ""));
    }
    setEntries(nextEntries);
  }, [
    allocateId,
    entries.length,
    models,
    target.selectedModel,
    target.selectedPreset,
  ]);

  const { entryData, presetNamesByEntry, presetDefaultsSignature } =
    useCompareEntryData(entries);

  useEffect(() => {
    setEntries((current) => {
      let changed = false;
      const nextEntries = current.map((entry) => {
        const presetNames = presetNamesByEntry.get(entry.id) ?? [];
        if (presetNames.length === 0) {
          if (!entry.preset) {
            return entry;
          }
          changed = true;
          return { ...entry, preset: "" };
        }
        if (entry.preset && presetNames.includes(entry.preset)) {
          return entry;
        }
        changed = true;
        return { ...entry, preset: presetNames[0] };
      });
      return changed ? nextEntries : current;
    });
  }, [presetDefaultsSignature, presetNamesByEntry]);

  const updateEntry = useCallback((id: string, patch: Partial<CompareEntry>) => {
    setEntries((current) =>
      current.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)),
    );
  }, []);

  const addEntry = useCallback(() => {
    setEntries((current) => {
      if (current.length >= MAX_COMPARE_TARGETS || models.length === 0) {
        return current;
      }
      const nextModel =
        models.find((model) => !current.some((entry) => entry.model === model)) ??
        models[0];
      return [...current, createCompareEntry(allocateId(), nextModel, "")];
    });
  }, [allocateId, models]);

  const removeEntry = useCallback((id: string) => {
    setEntries((current) => current.filter((entry) => entry.id !== id));
  }, []);

  const resetEntries = useCallback(() => {
    setEntries([]);
  }, []);

  const applyAsTarget = useCallback(
    (entry: CompareEntry) => {
      if (!entry.model || !entry.preset) {
        return;
      }
      target.selectModel(entry.model);
      target.selectPreset(entry.preset);
      onUseTarget();
    },
    [onUseTarget, target],
  );

  const modelOptions = useMemo(() => buildCompareModelOptions(models), [models]);
  const changedStats = statRows(entryData).filter((row) => row.changed);
  const configRows = changedConfigRows(entryData);

  return {
    catalog: {
      isLoading: target.catalog.isLoading,
      isError: target.catalog.isError,
      error: target.catalog.error,
    },
    modelOptions,
    entries: entryData,
    readyEntryCount: entryData.filter(isReadyEntry).length,
    changedStats,
    configRows,
    canAddEntry: entries.length < MAX_COMPARE_TARGETS && models.length > 0,
    canResetEntries: models.length > 0,
    canRemoveEntry: entries.length > 1,
    addEntry,
    resetEntries,
    removeEntry,
    updateEntry,
    applyAsTarget,
  };
}
