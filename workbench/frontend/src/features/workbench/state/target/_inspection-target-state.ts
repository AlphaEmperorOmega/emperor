import { useCallback, useMemo, useReducer } from "react";

export type HistoricalExperimentTarget = {
  runId: string;
  experiment: string;
  preset: string;
  dataset: string;
  experimentTask?: string | null;
};

type InspectionTargetState =
  | { kind: "preset"; preset: string }
  | { kind: "snapshot"; snapshotId: string; preset: string }
  | { kind: "historical-run"; run: HistoricalExperimentTarget };

type InspectionTargetTransition =
  | { type: "select-preset"; preset: string }
  | { type: "select-snapshot"; snapshotId: string; preset: string }
  | { type: "select-historical-run"; run: HistoricalExperimentTarget };

function inspectionTargetReducer(
  current: InspectionTargetState,
  transition: InspectionTargetTransition,
): InspectionTargetState {
  if (transition.type === "select-preset") {
    return current.kind === "preset" && current.preset === transition.preset
      ? current
      : { kind: "preset", preset: transition.preset };
  }
  if (transition.type === "select-snapshot") {
    return current.kind === "snapshot" &&
      current.snapshotId === transition.snapshotId &&
      current.preset === transition.preset
      ? current
      : {
          kind: "snapshot",
          snapshotId: transition.snapshotId,
          preset: transition.preset,
        };
  }
  return current.kind === "historical-run" &&
    current.run.runId === transition.run.runId &&
    current.run.experiment === transition.run.experiment &&
    current.run.preset === transition.run.preset &&
    current.run.dataset === transition.run.dataset &&
    current.run.experimentTask === transition.run.experimentTask
    ? current
    : { kind: "historical-run", run: transition.run };
}

type RestorationState = {
  intentGeneration: number;
  pendingGeneration: number | null;
};

type RestorationTransition =
  | { type: "user-intent" }
  | { type: "settle"; generation: number };

function restorationReducer(
  current: RestorationState,
  transition: RestorationTransition,
): RestorationState {
  if (transition.type === "user-intent") {
    return {
      intentGeneration: current.intentGeneration + 1,
      pendingGeneration: null,
    };
  }
  return current.pendingGeneration === transition.generation
    ? { ...current, pendingGeneration: null }
    : current;
}

export function useInspectionTargetState({
  initialPreset,
  restorePersistedTarget,
}: {
  initialPreset: string;
  restorePersistedTarget: boolean;
}) {
  const [target, dispatchTarget] = useReducer(inspectionTargetReducer, {
    kind: "preset",
    preset: initialPreset,
  });
  const [restoration, dispatchRestoration] = useReducer(restorationReducer, {
    intentGeneration: 0,
    pendingGeneration: restorePersistedTarget ? 0 : null,
  });

  const toPreset = useCallback((preset: string) => {
    dispatchTarget({ type: "select-preset", preset });
  }, []);
  const toSnapshot = useCallback((snapshotId: string, preset: string) => {
    dispatchTarget({ type: "select-snapshot", snapshotId, preset });
  }, []);
  const toHistoricalRun = useCallback((run: HistoricalExperimentTarget) => {
    dispatchTarget({ type: "select-historical-run", run });
  }, []);
  const cancelRestoration = useCallback(() => {
    dispatchRestoration({ type: "user-intent" });
  }, []);
  const settleRestoration = useCallback(() => {
    if (restoration.pendingGeneration !== null) {
      dispatchRestoration({
        type: "settle",
        generation: restoration.pendingGeneration,
      });
    }
  }, [restoration.pendingGeneration]);

  const transitions = useMemo(
    () => ({ toPreset, toSnapshot, toHistoricalRun }),
    [toHistoricalRun, toPreset, toSnapshot],
  );
  const restorationLifecycle = useMemo(
    () => ({
      isRestoring: restoration.pendingGeneration !== null,
      cancel: cancelRestoration,
      settle: settleRestoration,
    }),
    [cancelRestoration, restoration.pendingGeneration, settleRestoration],
  );

  return useMemo(
    () => ({ target, transitions, restoration: restorationLifecycle }),
    [restorationLifecycle, target, transitions],
  );
}
