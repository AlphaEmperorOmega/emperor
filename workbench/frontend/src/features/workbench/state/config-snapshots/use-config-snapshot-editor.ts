import { useCallback, useEffect, useMemo, useState } from "react";
import {
  configSectionsFields,
  groupConfigFieldsBySectionPath,
  runtimeDefaultsEditor,
  type OverrideValues,
} from "@/lib/config";
import {
  createConfigSnapshot,
  groupConfigSnapshotsByPreset,
  type ConfigSnapshotCreateResult,
  validateConfigSnapshotCandidate,
  validateConfigSnapshotName,
} from "@/lib/config-snapshots";
import { useRuntimeDefaultsSchema } from "@/features/workbench/state/model-package/use-model-package-metadata";
import {
  useConfigSnapshotRecords,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";
import {
  type ConfigSnapshotEditorSessionState,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-editor-session";

function createSnapshotId() {
  return globalThis.crypto?.randomUUID?.() ?? `snapshot-${Date.now()}`;
}

/**
 * Modal Config Snapshot editor session. It has its own Model Package/preset,
 * schema read, and draft so opening it from Training cannot mutate the active
 * Inspection target merely to materialize editor state.
 */
export function useConfigSnapshotEditorState({
  sessionState,
  protectedReadsEnabled = true,
}: {
  sessionState: ConfigSnapshotEditorSessionState;
  protectedReadsEnabled?: boolean;
}) {
  const { session, actions: sessionActions } = sessionState;
  const [draft, setDraft] = useState<OverrideValues>(
    () => ({ ...(session?.overrides ?? {}) }),
  );
  const modelType = session?.modelType ?? "";
  const model = session?.model ?? "";
  const preset = session?.preset ?? "";
  const schemaSelection = useMemo(
    () => ({ modelPackage: { modelType, model }, preset }),
    [model, modelType, preset],
  );
  const runtimeDefaultsSchema = useRuntimeDefaultsSchema(schemaSelection, {
    enabled: protectedReadsEnabled,
  });
  const snapshotRecords = useConfigSnapshotRecords(
    { modelType, model },
    { enabled: protectedReadsEnabled },
  );
  const configSections = useMemo(
    () => groupConfigFieldsBySectionPath(runtimeDefaultsSchema.fields),
    [runtimeDefaultsSchema.fields],
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
    setDraft({ ...(session?.overrides ?? {}) });
  }, [session?.overrides]);

  useEffect(() => {
    if (configFields.length === 0) {
      return;
    }
    setDraft((current) =>
      runtimeDefaultsEditor.normalize(configFields, current),
    );
  }, [configFields, model, modelType, preset, session?.kind]);

  const updateOverride = useCallback(
    (key: string, value: string) => {
      setDraft((current) =>
        runtimeDefaultsEditor.edit(configFields, current, key, value),
      );
    },
    [configFields],
  );
  const clearOverride = useCallback(
    (key: string) => {
      setDraft((current) =>
        runtimeDefaultsEditor.clear(configFields, current, key),
      );
    },
    [configFields],
  );
  const reset = useCallback(() => setDraft({}), []);
  const close = useCallback(() => {
    if (snapshotRecords.status.mutation.phase === "pending") {
      return false;
    }
    snapshotRecords.actions.dismissMutation();
    sessionActions.close();
    setDraft({});
    return true;
  }, [
    sessionActions,
    snapshotRecords.actions,
    snapshotRecords.status.mutation.phase,
  ]);
  const clearForConnectionChange = useCallback(() => {
    snapshotRecords.actions.clearForConnectionChange();
    sessionActions.close();
    setDraft({});
  }, [sessionActions, snapshotRecords.actions]);
  const load = useCallback(
    (snapshotId: string) => {
      const snapshot = snapshotRecords.records.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot) {
        return false;
      }
      return session?.kind === "edit"
        ? sessionActions.beginEdit(snapshot)
        : sessionActions.beginDuplicate(snapshot);
    },
    [session?.kind, sessionActions, snapshotRecords.records],
  );
  const rename = useCallback(
    async (snapshotId: string, name: string) => {
      const snapshot = snapshotRecords.records.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot) {
        return {
          ok: false as const,
          error: "The selected Config Snapshot is unavailable.",
        };
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
        return validation;
      }
      return snapshotRecords.actions.rename({
        id: snapshot.id,
        name: validation.name,
      });
    },
    [snapshotRecords.actions, snapshotRecords.records],
  );
  const remove = useCallback(
    (snapshotId: string) => snapshotRecords.actions.remove(snapshotId),
    [snapshotRecords.actions],
  );

  const save = useCallback(
    async (name: string): Promise<ConfigSnapshotCreateResult> => {
      if (!session || !snapshotRecords.status.isReady) {
        return { ok: false, error: "Config Snapshot records are still loading." };
      }
      if (snapshotRecords.status.mutation.phase === "pending") {
        return {
          ok: false,
          error: "Another Config Snapshot change is still pending.",
        };
      }
      const normalizedDraft = runtimeDefaultsEditor.normalize(
        configFields,
        draft,
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
        if (!result.ok) {
          return result;
        }
        const outcome = await snapshotRecords.actions.create({
          modelType: result.snapshot.modelType,
          model: result.snapshot.model,
          preset: result.snapshot.preset,
          name: result.snapshot.name,
          overrides: result.snapshot.overrides,
        });
        if (!outcome.ok) {
          return { ok: false, error: outcome.error };
        }
        if (!outcome.record) {
          return {
            ok: false,
            error: "The backend did not return the created Config Snapshot.",
          };
        }
        return { ok: true, snapshot: outcome.record };
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
      const outcome = await snapshotRecords.actions.update({
        id: selectedSnapshot.id,
        input: {
          name: nameValidation.name,
          overrides: validation.overrides,
        },
      });
      if (!outcome.ok) {
        return { ok: false, error: outcome.error };
      }
      if (!outcome.record) {
        return {
          ok: false,
          error: "The backend did not return the updated Config Snapshot.",
        };
      }
      return { ok: true, snapshot: outcome.record };
    },
    [
      configFields,
      draft,
      selectedSnapshot,
      session,
      snapshotRecords.actions,
      snapshotRecords.records,
      snapshotRecords.status.isReady,
      snapshotRecords.status.mutation.phase,
    ],
  );
  const retrySave = useCallback(async (): Promise<ConfigSnapshotCreateResult> => {
    const outcome = await snapshotRecords.actions.retry();
    if (!outcome) {
      return { ok: false, error: "There is no failed Config Snapshot save to retry." };
    }
    if (!outcome.ok) {
      return { ok: false, error: outcome.error };
    }
    if (
      (outcome.kind !== "create" && outcome.kind !== "update") ||
      !outcome.record
    ) {
      return { ok: false, error: "The failed change was not a Config Snapshot save." };
    }
    return { ok: true, snapshot: outcome.record };
  }, [snapshotRecords.actions]);

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
          isLoading:
            runtimeDefaultsSchema.isLoading || snapshotRecords.status.isLoading,
          mutation: snapshotRecords.status.mutation,
        },
      },
      actions: {
        beginDraft: sessionActions.beginDraft,
        beginEdit: sessionActions.beginEdit,
        beginDuplicate: sessionActions.beginDuplicate,
        updateOverride,
        clearOverride,
        reset,
        load,
        rename,
        remove,
        save,
        retryMutation: snapshotRecords.actions.retry,
        retrySave,
        dismissMutation: snapshotRecords.actions.dismissMutation,
        close,
        clearForConnectionChange,
      },
    }),
    [
      clearOverride,
      clearForConnectionChange,
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
      retrySave,
      save,
      runtimeDefaultsSchema.isLoading,
      selectedSnapshot,
      sessionActions,
      snapshotRecords.status.isLoading,
      snapshotRecords.status.mutation,
      snapshotRecords.actions,
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
