import { useCallback, useMemo, useState } from "react";
import { type OverrideValues } from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

export type SnapshotEditorSession =
  | {
      kind: "draft";
      modelType: string;
      model: string;
      preset: string;
      overrides: OverrideValues;
      sourceSnapshot?: ConfigSnapshot;
    }
  | {
      kind: "edit";
      modelType: string;
      model: string;
      preset: string;
      overrides: OverrideValues;
      sourceSnapshot: ConfigSnapshot;
    };

type BeginSnapshotDraft = {
  modelType: string;
  model: string;
  preset: string;
  overrides?: OverrideValues;
  sourceSnapshot?: ConfigSnapshot;
};

/** Lightweight launcher retained in the initial Workbench provider. */
export function useConfigSnapshotEditorSessionState() {
  const [session, setSession] = useState<SnapshotEditorSession | null>(null);
  const beginDraft = useCallback(
    ({
      modelType,
      model,
      preset,
      overrides = {},
      sourceSnapshot,
    }: BeginSnapshotDraft) => {
      if (!modelType || !model || !preset) {
        return false;
      }
      setSession({
        kind: "draft",
        modelType,
        model,
        preset,
        overrides: { ...overrides },
        sourceSnapshot,
      });
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
      overrides: { ...snapshot.overrides },
      sourceSnapshot: snapshot,
    });
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
  const close = useCallback(() => setSession(null), []);
  const actions = useMemo(
    () => ({ beginDraft, beginEdit, beginDuplicate, close }),
    [beginDraft, beginDuplicate, beginEdit, close],
  );
  return useMemo(() => ({ session, actions }), [actions, session]);
}

export type ConfigSnapshotEditorSessionState = ReturnType<
  typeof useConfigSnapshotEditorSessionState
>;
