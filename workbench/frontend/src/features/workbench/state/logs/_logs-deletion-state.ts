import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  createLogRunDeletePlan,
  deleteLogExperiment,
  deleteLogRuns,
  type LogExperimentDeleteResponse,
  type LogRun,
  type LogRunDeleteFilters,
  type LogRunDeletePlan,
  type LogRunDeleteResponse,
} from "@/lib/api";
import { type ChecklistOption } from "@/features/workbench/state/logs/logs-selectors";
import { buildLogRunDeleteFilters } from "@/features/workbench/state/logs/_logs-selection-state";

type LogsDeletePhase =
  | "planning"
  | "planFailed"
  | "ready"
  | "mutating"
  | "mutationFailed";

export type LogsSubsetDeleteTarget = {
  kind: "preset";
  value: string;
  experiment: string;
  key: string;
};

type ExperimentOperation = {
  kind: "experiment";
  option: ChecklistOption;
  phase: Exclude<LogsDeletePhase, "planning" | "planFailed">;
  error: unknown;
};

type SubsetOperation = {
  kind: "preset";
  target: LogsSubsetDeleteTarget;
  filters: LogRunDeleteFilters;
  phase: LogsDeletePhase;
  plan: LogRunDeletePlan | undefined;
  error: unknown;
};

type InternalOperation = ExperimentOperation | SubsetOperation;

export type LogsDeletionOperation =
  | ExperimentOperation
  | Omit<SubsetOperation, "filters">;

export type LogsDeletion = {
  enabled: boolean;
  presetTargetExperiment: string | null;
  operation: LogsDeletionOperation | null;
  actions: {
    openExperiment: (option: ChecklistOption) => void;
    openPreset: (option: ChecklistOption) => void;
    cancel: () => void;
    retryPlan: () => Promise<void>;
    confirm: () => Promise<void>;
  };
};

export type LogsDeletionImplementation = LogsDeletion & {
  clearForConnectionChange: () => void;
};

function logDeletionDisabledError() {
  return new Error("Log deletion is disabled by backend capabilities.");
}

function presetDeleteTarget({
  experiment,
  option,
  runs,
}: {
  experiment: string;
  option: ChecklistOption;
  runs: LogRun[];
}): { target: LogsSubsetDeleteTarget; filters: LogRunDeleteFilters } | null {
  const targetRuns = runs.filter(
    (run) => run.experiment === experiment && run.preset === option.value,
  );
  if (targetRuns.length === 0) {
    return null;
  }
  const filters = buildLogRunDeleteFilters(targetRuns);
  return {
    target: {
      kind: "preset",
      value: option.value,
      experiment,
      key: JSON.stringify({
        kind: "preset",
        value: option.value,
        experiment,
        filters,
      }),
    },
    filters,
  };
}

export function useLogsDeletionState({
  active,
  enabled,
  runs,
  selectedExperiments,
  onExperimentDeleted,
  onRunsDeleted,
}: {
  active: boolean;
  enabled: boolean;
  runs: LogRun[];
  selectedExperiments: Set<string>;
  onExperimentDeleted: (result: LogExperimentDeleteResponse) => void;
  onRunsDeleted: (
    result: LogRunDeleteResponse,
    target: LogsSubsetDeleteTarget,
  ) => void;
}): LogsDeletionImplementation {
  const [operation, setOperation] = useState<InternalOperation | null>(null);
  const operationRef = useRef<InternalOperation | null>(null);
  const revisionRef = useRef(0);
  const connectionGenerationRef = useRef(0);
  const experimentMutation = useMutation({ mutationFn: deleteLogExperiment });
  const planMutation = useMutation({ mutationFn: createLogRunDeletePlan });
  const subsetMutation = useMutation({ mutationFn: deleteLogRuns });
  const selectedExperimentValues = Array.from(selectedExperiments);
  const presetTargetExperiment =
    selectedExperimentValues.length === 1 ? selectedExperimentValues[0] : null;

  const publish = useCallback((next: InternalOperation | null) => {
    operationRef.current = next;
    setOperation(next);
  }, []);

  const cancel = useCallback(() => {
    revisionRef.current += 1;
    publish(null);
    experimentMutation.reset();
    planMutation.reset();
    subsetMutation.reset();
  }, [experimentMutation, planMutation, publish, subsetMutation]);
  const clearForConnectionChange = useCallback(() => {
    connectionGenerationRef.current += 1;
    cancel();
  }, [cancel]);

  useEffect(() => {
    if (active && enabled) {
      return;
    }
    if (operationRef.current) {
      cancel();
    }
  }, [active, cancel, enabled]);

  useEffect(() => {
    const current = operationRef.current;
    if (
      current?.kind !== "preset" ||
      current.target.experiment === presetTargetExperiment
    ) {
      return;
    }
    cancel();
  }, [cancel, presetTargetExperiment]);

  const openExperiment = useCallback(
    (option: ChecklistOption) => {
      if (!enabled) {
        return;
      }
      revisionRef.current += 1;
      experimentMutation.reset();
      planMutation.reset();
      subsetMutation.reset();
      publish({
        kind: "experiment",
        option,
        phase: "ready",
        error: null,
      });
    },
    [enabled, experimentMutation, planMutation, publish, subsetMutation],
  );

  const loadPlan = useCallback(
    async (current: SubsetOperation) => {
      if (!enabled) {
        throw logDeletionDisabledError();
      }
      const revision = ++revisionRef.current;
      const connectionGeneration = connectionGenerationRef.current;
      const planning: SubsetOperation = {
        ...current,
        phase: "planning",
        plan: undefined,
        error: null,
      };
      publish(planning);
      try {
        const plan = await planMutation.mutateAsync(current.filters);
        if (
          revisionRef.current === revision &&
          connectionGenerationRef.current === connectionGeneration
        ) {
          publish({ ...planning, phase: "ready", plan });
        }
      } catch (error) {
        if (
          revisionRef.current === revision &&
          connectionGenerationRef.current === connectionGeneration
        ) {
          publish({ ...planning, phase: "planFailed", error });
        }
        throw error;
      }
    },
    [enabled, planMutation, publish],
  );

  const openPreset = useCallback(
    (option: ChecklistOption) => {
      if (!enabled || !presetTargetExperiment) {
        return;
      }
      const prepared = presetDeleteTarget({
        experiment: presetTargetExperiment,
        option,
        runs,
      });
      if (!prepared) {
        return;
      }
      experimentMutation.reset();
      planMutation.reset();
      subsetMutation.reset();
      const next: SubsetOperation = {
        kind: "preset",
        target: prepared.target,
        filters: prepared.filters,
        phase: "planning",
        plan: undefined,
        error: null,
      };
      void loadPlan(next).catch(() => undefined);
    },
    [
      enabled,
      experimentMutation,
      loadPlan,
      planMutation,
      presetTargetExperiment,
      runs,
      subsetMutation,
    ],
  );

  const retryPlan = useCallback(async () => {
    const current = operationRef.current;
    if (current?.kind !== "preset") {
      return;
    }
    await loadPlan(current);
  }, [loadPlan]);

  const confirm = useCallback(async () => {
    const current = operationRef.current;
    if (!current) {
      return;
    }
    if (!enabled) {
      const error = logDeletionDisabledError();
      publish({ ...current, phase: "mutationFailed", error });
      throw error;
    }
    if (current.kind === "preset" && !current.plan?.canDelete) {
      return;
    }
    const revision = revisionRef.current;
    const connectionGeneration = connectionGenerationRef.current;
    publish({ ...current, phase: "mutating", error: null });
    try {
      if (current.kind === "experiment") {
        const result = await experimentMutation.mutateAsync(current.option.value);
        if (connectionGenerationRef.current === connectionGeneration) {
          onExperimentDeleted(result);
        }
      } else {
        const result = await subsetMutation.mutateAsync(current.filters);
        if (connectionGenerationRef.current === connectionGeneration) {
          onRunsDeleted(result, current.target);
        }
      }
      if (
        revisionRef.current === revision &&
        connectionGenerationRef.current === connectionGeneration
      ) {
        revisionRef.current += 1;
        publish(null);
      }
    } catch (error) {
      if (
        revisionRef.current === revision &&
        connectionGenerationRef.current === connectionGeneration
      ) {
        publish({ ...current, phase: "mutationFailed", error });
      }
      throw error;
    }
  }, [enabled, experimentMutation, onExperimentDeleted, onRunsDeleted, publish, subsetMutation]);

  const publicOperation: LogsDeletionOperation | null = operation
    ? operation.kind === "preset"
      ? {
          kind: operation.kind,
          target: operation.target,
          phase: operation.phase,
          plan: operation.plan,
          error: operation.error,
        }
      : operation
    : null;

  return {
    enabled,
    presetTargetExperiment,
    operation: publicOperation,
    actions: {
      openExperiment,
      openPreset,
      cancel,
      retryPlan,
      confirm,
    },
    clearForConnectionChange,
  };
}
