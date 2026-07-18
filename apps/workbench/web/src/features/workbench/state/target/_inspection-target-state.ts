import { useCallback, useReducer } from "react";
import type { ModelIdentity } from "@/lib/api/model-catalog";
import type { OverrideValues } from "@/lib/config";

export type HistoricalExperimentTarget = Readonly<{
  runId: string;
  experiment: string;
  preset: string;
  dataset: string;
  experimentTask?: string | null;
}>;

export type InspectionTarget =
  | Readonly<{ kind: "preset"; preset: string }>
  | Readonly<{ kind: "snapshot"; snapshotId: string; preset: string }>
  | Readonly<{ kind: "historical-run"; run: HistoricalExperimentTarget }>;

export type InspectionTransition = Readonly<{
  revision: number;
  cause: "target-changed" | "inspection-refreshed";
}>;

export type InspectionRestoration = Readonly<{
  phase: "restoring-browser" | "rebuilding-connection" | "settled";
  requestedSnapshotId: string;
}>;

export type InspectionTargetBrowserMode =
  | "preset"
  | "snapshot"
  | "experiment";

export type InspectionTargetLifecycleState = Readonly<{
  modelPackage: ModelIdentity;
  selectedPreset: string;
  target: InspectionTarget;
  browserMode: InspectionTargetBrowserMode;
  hasReadPersistedTargetSelection: boolean;
  experimentTask: string;
  datasets: string[];
  runtimeDefaults: Readonly<{
    preset: OverrideValues;
    active: OverrideValues;
  }>;
  restoration: InspectionRestoration;
  transition: InspectionTransition;
  connectionRevision: number;
}>;

export type InspectionTargetLifecycleEvent =
  | Readonly<{
      type: "browser-target-restored";
      modelPackage: ModelIdentity;
      preset: string;
      browserMode: InspectionTargetBrowserMode;
      requestedSnapshotId: string;
    }>
  | Readonly<{
      type: "browser-mode-selected";
      browserMode: InspectionTargetBrowserMode;
    }>
  | Readonly<{ type: "model-package-selected"; modelPackage: ModelIdentity }>
  | Readonly<{ type: "preset-selected"; preset: string }>
  | Readonly<{ type: "preset-metadata-selected"; preset: string }>
  | Readonly<{
      type: "snapshot-selected";
      snapshotId: string;
      preset: string;
      runtimeDefaults: OverrideValues;
    }>
  | Readonly<{
      type: "historical-run-selected";
      run: HistoricalExperimentTarget;
      catalogPreset?: string;
      catalogDataset?: string;
    }>
  | Readonly<{
      type: "experiment-task-selected";
      experimentTask: string;
      datasets: string[];
    }>
  | Readonly<{ type: "datasets-selected"; datasets: string[] }>
  | Readonly<{
      type: "runtime-defaults-edited";
      presetRuntimeDefaults: OverrideValues;
      activeRuntimeDefaults: OverrideValues;
    }>
  | Readonly<{
      type: "runtime-defaults-normalized";
      presetRuntimeDefaults: OverrideValues;
      activeRuntimeDefaults: OverrideValues;
      expectedRevision: number;
    }>
  | Readonly<{ type: "runtime-defaults-reset" }>
  | Readonly<{
      type: "metadata-refreshed";
      preset: string;
      experimentTask: string;
      datasets: string[];
      expectedRevision: number;
    }>
  | Readonly<{
      type: "snapshot-runtime-defaults-refreshed";
      snapshotId: string;
      preset: string;
      runtimeDefaults: OverrideValues;
      expectedRevision: number;
    }>
  | Readonly<{ type: "missing-snapshot-fallback" }>
  | Readonly<{
      type: "restoration-snapshot-resolved";
      snapshotId: string;
      preset: string;
      runtimeDefaults: OverrideValues;
    }>
  | Readonly<{ type: "restoration-settled" }>
  | Readonly<{ type: "restoration-cancelled" }>
  | Readonly<{ type: "connection-reset" }>
  | Readonly<{ type: "inspection-refreshed" }>;

function sameModelPackage(left: ModelIdentity, right: ModelIdentity) {
  return left.modelType === right.modelType && left.model === right.model;
}

function sameValues(left: readonly string[], right: readonly string[]) {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function sameOverrides(left: OverrideValues, right: OverrideValues) {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key) => left[key] === right[key])
  );
}

function sameHistoricalRun(
  left: HistoricalExperimentTarget,
  right: HistoricalExperimentTarget,
) {
  return (
    left.runId === right.runId &&
    left.experiment === right.experiment &&
    left.preset === right.preset &&
    left.dataset === right.dataset &&
    left.experimentTask === right.experimentTask
  );
}

function withTargetChange(
  current: InspectionTargetLifecycleState,
  nextState: Omit<InspectionTargetLifecycleState, "transition">,
): InspectionTargetLifecycleState {
  return {
    ...nextState,
    transition: {
      revision: current.transition.revision + 1,
      cause: "target-changed",
    },
  };
}

function settleRestoration(
  current: InspectionTargetLifecycleState,
): InspectionTargetLifecycleState {
  return current.restoration.phase === "settled"
    ? current
    : {
        ...current,
        restoration: { phase: "settled", requestedSnapshotId: "" },
      };
}

export function createInspectionTargetLifecycleState({
  modelPackage,
  preset,
  restoreSnapshotId = "",
  restorePersistedTarget = false,
}: {
  modelPackage: ModelIdentity;
  preset: string;
  restoreSnapshotId?: string;
  restorePersistedTarget?: boolean;
}): InspectionTargetLifecycleState {
  return {
    modelPackage,
    selectedPreset: preset,
    target: { kind: "preset", preset },
    browserMode: "preset",
    hasReadPersistedTargetSelection: !restorePersistedTarget,
    experimentTask: "",
    datasets: [],
    runtimeDefaults: { preset: {}, active: {} },
    restoration: restorePersistedTarget
      ? {
          phase: "restoring-browser",
          requestedSnapshotId: restoreSnapshotId,
        }
      : { phase: "settled", requestedSnapshotId: "" },
    transition: { revision: 0, cause: "target-changed" },
    connectionRevision: 0,
  };
}

export function inspectionTargetLifecycleReducer(
  current: InspectionTargetLifecycleState,
  event: InspectionTargetLifecycleEvent,
): InspectionTargetLifecycleState {
  if (event.type === "browser-target-restored") {
    return withTargetChange(current, {
      ...current,
      modelPackage: event.modelPackage,
      selectedPreset: event.preset,
      target: { kind: "preset", preset: event.preset },
      browserMode: event.browserMode,
      hasReadPersistedTargetSelection: true,
      experimentTask: "",
      datasets: [],
      runtimeDefaults: { preset: {}, active: {} },
      restoration: {
        phase: "restoring-browser",
        requestedSnapshotId: event.requestedSnapshotId,
      },
    });
  }
  if (event.type === "browser-mode-selected") {
    return current.browserMode === event.browserMode
      ? settleRestoration(current)
      : {
          ...current,
          browserMode: event.browserMode,
          restoration: { phase: "settled", requestedSnapshotId: "" },
        };
  }
  if (event.type === "inspection-refreshed") {
    return {
      ...current,
      transition: {
        revision: current.transition.revision + 1,
        cause: "inspection-refreshed",
      },
    };
  }
  if (event.type === "connection-reset") {
    return {
      ...current,
      restoration: {
        phase: "rebuilding-connection",
        requestedSnapshotId:
          current.target.kind === "snapshot" ? current.target.snapshotId : "",
      },
      connectionRevision: current.connectionRevision + 1,
    };
  }
  if (event.type === "restoration-settled") {
    return settleRestoration({
      ...current,
      hasReadPersistedTargetSelection: true,
    });
  }
  if (event.type === "restoration-cancelled") {
    return settleRestoration(current);
  }
  if (event.type === "model-package-selected") {
    if (sameModelPackage(current.modelPackage, event.modelPackage)) {
      return settleRestoration(current);
    }
    return withTargetChange(current, {
      ...current,
      modelPackage: event.modelPackage,
      selectedPreset: "",
      target: { kind: "preset", preset: "" },
      browserMode: "preset",
      experimentTask: "",
      datasets: [],
      runtimeDefaults: { preset: {}, active: {} },
      restoration: { phase: "settled", requestedSnapshotId: "" },
    });
  }
  if (event.type === "preset-metadata-selected") {
    if (current.selectedPreset === event.preset) {
      return current;
    }
    const target =
      current.target.kind === "preset"
        ? ({ kind: "preset", preset: event.preset } as const)
        : current.target;
    return withTargetChange(current, {
      ...current,
      selectedPreset: event.preset,
      target,
      runtimeDefaults:
        target.kind === "preset"
          ? { ...current.runtimeDefaults, active: current.runtimeDefaults.preset }
          : current.runtimeDefaults,
    });
  }
  if (event.type === "preset-selected") {
    const nextTarget = { kind: "preset" as const, preset: event.preset };
    if (
      current.target.kind === "preset" &&
      current.target.preset === event.preset &&
      current.selectedPreset === event.preset &&
      current.browserMode === "preset"
    ) {
      return settleRestoration(current);
    }
    return withTargetChange(current, {
      ...current,
      selectedPreset: event.preset,
      target: nextTarget,
      browserMode: "preset",
      runtimeDefaults: {
        ...current.runtimeDefaults,
        active: current.runtimeDefaults.preset,
      },
      restoration: { phase: "settled", requestedSnapshotId: "" },
    });
  }
  if (event.type === "snapshot-selected") {
    if (
      current.target.kind === "snapshot" &&
      current.target.snapshotId === event.snapshotId &&
      current.target.preset === event.preset &&
      current.browserMode === "snapshot" &&
      sameOverrides(current.runtimeDefaults.active, event.runtimeDefaults)
    ) {
      return settleRestoration(current);
    }
    return withTargetChange(current, {
      ...current,
      selectedPreset: event.preset,
      target: {
        kind: "snapshot",
        snapshotId: event.snapshotId,
        preset: event.preset,
      },
      browserMode: "snapshot",
      runtimeDefaults: {
        ...current.runtimeDefaults,
        active: { ...event.runtimeDefaults },
      },
      restoration: { phase: "settled", requestedSnapshotId: "" },
    });
  }
  if (event.type === "historical-run-selected") {
    const nextPreset = event.catalogPreset || current.selectedPreset;
    const nextTask = event.run.experimentTask || current.experimentTask;
    const nextDatasets = event.catalogDataset
      ? [event.catalogDataset]
      : current.datasets;
    if (
      current.target.kind === "historical-run" &&
      sameHistoricalRun(current.target.run, event.run) &&
      current.browserMode === "experiment" &&
      current.selectedPreset === nextPreset &&
      current.experimentTask === nextTask &&
      sameValues(current.datasets, nextDatasets) &&
      sameOverrides(current.runtimeDefaults.preset, {})
    ) {
      return settleRestoration(current);
    }
    return withTargetChange(current, {
      ...current,
      selectedPreset: nextPreset,
      target: { kind: "historical-run", run: event.run },
      browserMode: "experiment",
      experimentTask: nextTask,
      datasets: [...nextDatasets],
      runtimeDefaults: { preset: {}, active: {} },
      restoration: { phase: "settled", requestedSnapshotId: "" },
    });
  }
  if (event.type === "experiment-task-selected") {
    if (
      current.experimentTask === event.experimentTask &&
      sameValues(current.datasets, event.datasets)
    ) {
      return settleRestoration(current);
    }
    const nextState: Omit<InspectionTargetLifecycleState, "transition"> = {
      ...current,
      experimentTask: event.experimentTask,
      datasets: [...event.datasets],
      restoration: { phase: "settled", requestedSnapshotId: "" },
    };
    return current.target.kind === "historical-run"
      ? { ...nextState, transition: current.transition }
      : withTargetChange(current, nextState);
  }
  if (event.type === "datasets-selected") {
    if (sameValues(current.datasets, event.datasets)) {
      return current;
    }
    return withTargetChange(current, {
      ...current,
      datasets: [...event.datasets],
    });
  }
  if (event.type === "runtime-defaults-edited") {
    const exitsCompleteTarget = current.target.kind !== "preset";
    if (
      !exitsCompleteTarget &&
      sameOverrides(
        current.runtimeDefaults.preset,
        event.presetRuntimeDefaults,
      ) &&
      sameOverrides(
        current.runtimeDefaults.active,
        event.activeRuntimeDefaults,
      )
    ) {
      return settleRestoration(current);
    }
    return withTargetChange(current, {
      ...current,
      target: { kind: "preset", preset: current.selectedPreset },
      browserMode: "preset",
      runtimeDefaults: {
        preset: { ...event.presetRuntimeDefaults },
        active: { ...event.activeRuntimeDefaults },
      },
      restoration: { phase: "settled", requestedSnapshotId: "" },
    });
  }
  if (event.type === "runtime-defaults-reset") {
    if (
      current.target.kind === "preset" &&
      sameOverrides(current.runtimeDefaults.preset, {})
    ) {
      return settleRestoration(current);
    }
    return withTargetChange(current, {
      ...current,
      target: { kind: "preset", preset: current.selectedPreset },
      browserMode: "preset",
      runtimeDefaults: { preset: {}, active: {} },
      restoration: { phase: "settled", requestedSnapshotId: "" },
    });
  }
  if (event.type === "runtime-defaults-normalized") {
    if (event.expectedRevision !== current.transition.revision) {
      return current;
    }
    const active = current.target.kind === "preset"
      ? event.activeRuntimeDefaults
      : current.runtimeDefaults.active;
    if (
      sameOverrides(
        current.runtimeDefaults.preset,
        event.presetRuntimeDefaults,
      ) &&
      sameOverrides(current.runtimeDefaults.active, active)
    ) {
      return current;
    }
    return {
      ...current,
      runtimeDefaults: {
        preset: { ...event.presetRuntimeDefaults },
        active: { ...active },
      },
    };
  }
  if (event.type === "metadata-refreshed") {
    if (event.expectedRevision !== current.transition.revision) {
      return current;
    }
    const target =
      current.target.kind === "preset"
        ? ({ kind: "preset", preset: event.preset } as const)
        : current.target;
    const active =
      target.kind === "preset"
        ? current.runtimeDefaults.preset
        : current.runtimeDefaults.active;
    const semanticallyChanged =
      current.selectedPreset !== event.preset ||
      current.experimentTask !== event.experimentTask ||
      !sameValues(current.datasets, event.datasets) ||
      (target.kind === "preset" &&
        (current.target.kind !== "preset" ||
          current.target.preset !== event.preset));
    const browserMode =
      current.target.kind === "preset" &&
      current.selectedPreset !== event.preset
        ? "preset"
        : current.browserMode;
    if (!semanticallyChanged && browserMode === current.browserMode) {
      return current;
    }
    return {
      ...current,
      selectedPreset: event.preset,
      target,
      browserMode,
      experimentTask: event.experimentTask,
      datasets: [...event.datasets],
      runtimeDefaults: { ...current.runtimeDefaults, active },
    };
  }
  if (event.type === "snapshot-runtime-defaults-refreshed") {
    if (event.expectedRevision !== current.transition.revision) {
      return current;
    }
    if (
      current.target.kind !== "snapshot" ||
      current.target.snapshotId !== event.snapshotId
    ) {
      return current;
    }
    if (
      current.selectedPreset === event.preset &&
      current.target.preset === event.preset &&
      sameOverrides(current.runtimeDefaults.active, event.runtimeDefaults)
    ) {
      return current;
    }
    return withTargetChange(current, {
      ...current,
      selectedPreset: event.preset,
      target: {
        kind: "snapshot",
        snapshotId: event.snapshotId,
        preset: event.preset,
      },
      browserMode: "snapshot",
      runtimeDefaults: {
        ...current.runtimeDefaults,
        active: { ...event.runtimeDefaults },
      },
    });
  }
  if (event.type === "restoration-snapshot-resolved") {
    return withTargetChange(current, {
      ...current,
      selectedPreset: event.preset,
      target: {
        kind: "snapshot",
        snapshotId: event.snapshotId,
        preset: event.preset,
      },
      browserMode: "snapshot",
      runtimeDefaults: {
        ...current.runtimeDefaults,
        active: { ...event.runtimeDefaults },
      },
      restoration: { phase: "settled", requestedSnapshotId: "" },
    });
  }
  return withTargetChange(current, {
    ...current,
    target: { kind: "preset", preset: current.selectedPreset },
    browserMode: "preset",
    runtimeDefaults: {
      ...current.runtimeDefaults,
      active: current.runtimeDefaults.preset,
    },
    restoration: { phase: "settled", requestedSnapshotId: "" },
  });
}

export function useInspectionTargetLifecycle(
  initialState: InspectionTargetLifecycleState,
) {
  const [state, dispatch] = useReducer(
    inspectionTargetLifecycleReducer,
    initialState,
  );
  const send = useCallback(
    (event: InspectionTargetLifecycleEvent) => dispatch(event),
    [],
  );
  return { state, send };
}
