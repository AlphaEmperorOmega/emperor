import { useCallback, useEffect, useMemo, useState } from "react";
import {
  configKeyToken,
  configSectionsFields,
  groupConfigFieldsBySectionPath,
  normalizeAdaptiveOptionOverrides,
  normalizeConfigOverrides,
  type OverrideValues,
} from "@/lib/config";
import {
  createConfigSnapshot,
  groupConfigSnapshotsByPreset,
  type ConfigSnapshot,
  type ConfigSnapshotCreateResult,
  validateConfigSnapshotCandidate,
  validateConfigSnapshotName,
} from "@/lib/config-snapshots";
import {
  useConfigSchemaQuery,
} from "@/features/workbench/state/use-workbench-queries";
import {
  useConfigSnapshotRecords,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";

type SnapshotEditorSession =
  | {
      kind: "draft";
      modelType: string;
      model: string;
      preset: string;
      sourceSnapshot?: ConfigSnapshot;
    }
  | {
      kind: "edit";
      modelType: string;
      model: string;
      preset: string;
      sourceSnapshot: ConfigSnapshot;
    };

type BeginSnapshotDraft = {
  modelType: string;
  model: string;
  preset: string;
  overrides?: OverrideValues;
  sourceSnapshot?: ConfigSnapshot;
};

function createSnapshotId() {
  return globalThis.crypto?.randomUUID?.() ?? `snapshot-${Date.now()}`;
}

function overrideValuesEqual(left: OverrideValues, right: OverrideValues) {
  const leftEntries = Object.entries(left);
  const rightEntries = Object.entries(right);
  return (
    leftEntries.length === rightEntries.length &&
    leftEntries.every(([key, value]) => right[key] === value)
  );
}

function withoutOverride(overrides: OverrideValues, key: string) {
  const token = configKeyToken(key);
  return Object.fromEntries(
    Object.entries(overrides).filter(
      ([overrideKey]) => configKeyToken(overrideKey) !== token,
    ),
  ) as OverrideValues;
}

/**
 * Modal Config Snapshot editor session. It has its own Model Package/preset,
 * schema read, and draft so opening it from Training cannot mutate the active
 * Inspection target merely to materialize editor state.
 */
export function useConfigSnapshotEditorState({
  protectedReadsEnabled = true,
}: {
  protectedReadsEnabled?: boolean;
} = {}) {
  const [session, setSession] = useState<SnapshotEditorSession | null>(null);
  const [draft, setDraft] = useState<OverrideValues>({});
  const modelType = session?.modelType ?? "";
  const model = session?.model ?? "";
  const preset = session?.preset ?? "";
  const schemaQuery = useConfigSchemaQuery(modelType, model, preset, {
    enabled: protectedReadsEnabled,
  });
  const snapshotRecords = useConfigSnapshotRecords(
    { modelType, model },
    { enabled: protectedReadsEnabled },
  );
  const configSections = useMemo(
    () => groupConfigFieldsBySectionPath(schemaQuery.data?.fields ?? []),
    [schemaQuery.data?.fields],
  );
  const configFields = useMemo(
    () => configSectionsFields(configSections),
    [configSections],
  );
  const selectedSnapshot = useMemo(() => {
    if (session?.kind !== "edit") {
      return undefined;
    }
    const current = snapshotRecords.records.find(
      (snapshot) => snapshot.id === session.sourceSnapshot.id,
    );
    if (snapshotRecords.status.isReady) {
      return current;
    }
    return current ?? session.sourceSnapshot;
  }, [session, snapshotRecords.records, snapshotRecords.status.isReady]);
  const snapshotGroups = useMemo(
    () => groupConfigSnapshotsByPreset(snapshotRecords.records, [preset]),
    [preset, snapshotRecords.records],
  );

  useEffect(() => {
    if (configFields.length === 0) {
      return;
    }
    setDraft((current) => {
      const normalized = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, current),
      );
      return overrideValuesEqual(current, normalized) ? current : normalized;
    });
  }, [configFields, model, modelType, preset, session?.kind]);

  const beginDraft = useCallback(
    ({
      modelType: nextModelType,
      model: nextModel,
      preset: nextPreset,
      overrides = {},
      sourceSnapshot,
    }: BeginSnapshotDraft) => {
      if (!nextModelType || !nextModel || !nextPreset) {
        return false;
      }
      setSession({
        kind: "draft",
        modelType: nextModelType,
        model: nextModel,
        preset: nextPreset,
        sourceSnapshot,
      });
      setDraft({ ...overrides });
      return true;
    },
    [],
  );

  const beginEdit = useCallback((snapshot: ConfigSnapshot) => {
    setSession({
      kind: "edit",
      modelType: snapshot.modelType,
      model: snapshot.model,
      preset: snapshot.preset,
      sourceSnapshot: snapshot,
    });
    setDraft({ ...snapshot.overrides });
    return true;
  }, []);

  const beginDuplicate = useCallback(
    (snapshot: ConfigSnapshot) =>
      beginDraft({
        modelType: snapshot.modelType,
        model: snapshot.model,
        preset: snapshot.preset,
        overrides: snapshot.overrides,
        sourceSnapshot: snapshot,
      }),
    [beginDraft],
  );

  const updateOverride = useCallback(
    (key: string, value: string) => {
      setDraft((current) =>
        normalizeAdaptiveOptionOverrides(
          configFields,
          normalizeConfigOverrides(configFields, {
            ...current,
            [key]: value,
          }),
        ),
      );
    },
    [configFields],
  );
  const clearOverride = useCallback(
    (key: string) => {
      setDraft((current) =>
        normalizeAdaptiveOptionOverrides(
          configFields,
          normalizeConfigOverrides(
            configFields,
            withoutOverride(current, key),
          ),
        ),
      );
    },
    [configFields],
  );
  const reset = useCallback(() => setDraft({}), []);
  const close = useCallback(() => {
    setSession(null);
    setDraft({});
  }, []);
  const load = useCallback(
    (snapshotId: string) => {
      const snapshot = snapshotRecords.records.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot) {
        return false;
      }
      return session?.kind === "edit"
        ? beginEdit(snapshot)
        : beginDuplicate(snapshot);
    },
    [beginDuplicate, beginEdit, session?.kind, snapshotRecords.records],
  );
  const rename = useCallback(
    (snapshotId: string, name: string) => {
      const snapshot = snapshotRecords.records.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot) {
        return false;
      }
      const validation = validateConfigSnapshotName({
        modelType: snapshot.modelType,
        model: snapshot.model,
        preset: snapshot.preset,
        name,
        snapshots: snapshotRecords.records,
        excludeSnapshotId: snapshot.id,
      });
      if (!validation.ok) {
        return false;
      }
      snapshotRecords.actions.rename({ id: snapshot.id, name: validation.name });
      return true;
    },
    [snapshotRecords.actions, snapshotRecords.records],
  );
  const remove = useCallback(
    (snapshotId: string) => snapshotRecords.actions.remove(snapshotId),
    [snapshotRecords.actions],
  );

  const save = useCallback(
    (name: string): ConfigSnapshotCreateResult => {
      if (!session || !snapshotRecords.status.isReady) {
        return { ok: false, error: "Config Snapshot records are still loading." };
      }
      const normalizedDraft = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, draft),
      );
      if (session.kind === "draft") {
        const result = createConfigSnapshot({
          id: createSnapshotId(),
          name,
          modelType: session.modelType,
          model: session.model,
          preset: session.preset,
          fields: configFields,
          overrides: normalizedDraft,
          snapshots: snapshotRecords.records,
          createdAt: new Date().toISOString(),
        });
        if (result.ok) {
          snapshotRecords.actions.create({
            modelType: result.snapshot.modelType,
            model: result.snapshot.model,
            preset: result.snapshot.preset,
            name: result.snapshot.name,
            overrides: result.snapshot.overrides,
          });
        }
        return result;
      }
      if (!selectedSnapshot) {
        return { ok: false, error: "The selected Config Snapshot is unavailable." };
      }
      const nameValidation = validateConfigSnapshotName({
        modelType: selectedSnapshot.modelType,
        model: selectedSnapshot.model,
        preset: selectedSnapshot.preset,
        name,
        snapshots: snapshotRecords.records,
        excludeSnapshotId: selectedSnapshot.id,
      });
      if (!nameValidation.ok) {
        return nameValidation;
      }
      const validation = validateConfigSnapshotCandidate({
        modelType: selectedSnapshot.modelType,
        model: selectedSnapshot.model,
        preset: selectedSnapshot.preset,
        fields: configFields,
        overrides: normalizedDraft,
        snapshots: snapshotRecords.records,
        excludeSnapshotId: selectedSnapshot.id,
      });
      if (!validation.ok) {
        return validation;
      }
      const snapshot = {
        ...selectedSnapshot,
        name: nameValidation.name,
        overrides: validation.overrides,
      };
      snapshotRecords.actions.update({
        id: selectedSnapshot.id,
        input: {
          name: nameValidation.name,
          overrides: validation.overrides,
        },
      });
      return { ok: true, snapshot };
    },
    [
      configFields,
      draft,
      selectedSnapshot,
      session,
      snapshotRecords.actions,
      snapshotRecords.records,
      snapshotRecords.status.isReady,
    ],
  );

  const value = useMemo(
    () => ({
      session: {
        modelType,
        model,
        preset,
        selectedSnapshot,
        records: snapshotRecords.records,
        recordGroups: snapshotGroups,
        configSections,
        fieldCount: configFields.length,
        draft,
        status: {
          isLoading: schemaQuery.isLoading || snapshotRecords.status.isLoading,
        },
      },
      actions: {
        beginDraft,
        beginEdit,
        beginDuplicate,
        updateOverride,
        clearOverride,
        reset,
        load,
        rename,
        remove,
        save,
        close,
      },
    }),
    [
      beginDraft,
      beginDuplicate,
      beginEdit,
      clearOverride,
      close,
      configFields.length,
      configSections,
      draft,
      model,
      modelType,
      preset,
      reset,
      load,
      remove,
      rename,
      save,
      schemaQuery.isLoading,
      selectedSnapshot,
      snapshotRecords.status.isLoading,
      snapshotRecords.records,
      snapshotGroups,
      updateOverride,
    ],
  );

  return value;
}

export type ConfigSnapshotEditorState = ReturnType<
  typeof useConfigSnapshotEditorState
>;
