import { useCallback, useMemo, useState } from "react";
import {
  configSectionsFields,
  groupConfigFieldsBySectionPath,
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
import { runtimeDefaultsEditor } from "@/features/workbench/state/runtime-defaults/runtime-defaults";
import { useConfigSnapshotRecords } from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";
import { type ConfigSnapshotEditorSessionState } from "@/features/workbench/state/config-snapshots/use-config-snapshot-editor-session";

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
  const [draftState, setDraftState] = useState<{
    session: typeof session;
    draft: OverrideValues;
  }>(() => ({
    session,
    draft: { ...(session?.overrides ?? {}) },
  }));
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
  const sessionDraft =
    draftState.session === session
      ? draftState.draft
      : { ...(session?.overrides ?? {}) };
  const draft =
    configFields.length > 0
      ? runtimeDefaultsEditor.normalize(configFields, sessionDraft)
      : sessionDraft;

  const updateOverride = useCallback(
    (key: string, value: string) => {
      setDraftState({
        session,
        draft: runtimeDefaultsEditor.edit(configFields, draft, key, value),
      });
    },
    [configFields, draft, session],
  );
  const clearOverride = useCallback(
    (key: string) => {
      setDraftState({
        session,
        draft: runtimeDefaultsEditor.clear(configFields, draft, key),
      });
    },
    [configFields, draft, session],
  );
  const reset = useCallback(
    () => setDraftState({ session, draft: {} }),
    [session],
  );
  const close = useCallback(() => {
    if (snapshotRecords.status.mutation.phase === "pending") {
      return false;
    }
    snapshotRecords.actions.dismissMutation();
    sessionActions.close();
    setDraftState({ session: null, draft: {} });
    return true;
  }, [
    sessionActions,
    snapshotRecords.actions,
    snapshotRecords.status.mutation.phase,
  ]);
  const clearForConnectionChange = useCallback(() => {
    snapshotRecords.actions.clearForConnectionChange();
    sessionActions.close();
    setDraftState({ session: null, draft: {} });
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
        const creationResult = createConfigSnapshot({
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
        if (!creationResult.ok) {
          return creationResult;
        }
        const persistenceResult = await snapshotRecords.actions.create({
          modelType: creationResult.snapshot.modelType,
          model: creationResult.snapshot.model,
          preset: creationResult.snapshot.preset,
          name: creationResult.snapshot.name,
          overrides: creationResult.snapshot.overrides,
        });
        if (!persistenceResult.ok) {
          return { ok: false, error: persistenceResult.error };
        }
        if (!persistenceResult.record) {
          return {
            ok: false,
            error: "The backend did not return the created Config Snapshot.",
          };
        }
        return { ok: true, snapshot: persistenceResult.record };
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
      const updateResult = await snapshotRecords.actions.update({
        id: selectedSnapshot.id,
        input: {
          name: nameValidation.name,
          overrides: validation.overrides,
        },
      });
      if (!updateResult.ok) {
        return { ok: false, error: updateResult.error };
      }
      if (!updateResult.record) {
        return {
          ok: false,
          error: "The backend did not return the updated Config Snapshot.",
        };
      }
      return { ok: true, snapshot: updateResult.record };
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
    const retryResult = await snapshotRecords.actions.retry();
    if (!retryResult) {
      return {
        ok: false,
        error: "There is no failed Config Snapshot save to retry.",
      };
    }
    if (!retryResult.ok) {
      return { ok: false, error: retryResult.error };
    }
    if (
      (retryResult.kind !== "create" && retryResult.kind !== "update") ||
      !retryResult.record
    ) {
      return {
        ok: false,
        error: "The failed change was not a Config Snapshot save.",
      };
    }
    return { ok: true, snapshot: retryResult.record };
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
