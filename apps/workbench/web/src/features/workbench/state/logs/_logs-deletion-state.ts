import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  createMutationRequestOptions,
  type MutationRequestOptions,
} from "@/lib/api/client";
import {
  createLogPresetDeletePlan,
  deleteLogExperiment,
  deleteLogPreset,
  type LogPresetDeleteTarget,
  type LogExperimentDeleteResponse,
  type LogRunDeletePlan,
  type LogRunDeleteResponse,
} from "@/lib/api/deletion";
import { type ChecklistOption } from "@/features/workbench/state/logs/logs-selectors";

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
  mutation: MutationRequestOptions;
};

type SubsetOperation = {
  kind: "preset";
  target: LogsSubsetDeleteTarget;
  phase: LogsDeletePhase;
  plan: LogRunDeletePlan | undefined;
  error: unknown;
  mutation: MutationRequestOptions;
};

type InternalOperation = ExperimentOperation | SubsetOperation;

export type LogsDeletionOperation =
  | Omit<ExperimentOperation, "mutation">
  | Omit<SubsetOperation, "mutation">;

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
}: {
  experiment: string;
  option: ChecklistOption;
}): LogsSubsetDeleteTarget {
  return {
    kind: "preset",
    value: option.value,
    experiment,
    key: JSON.stringify({
      kind: "preset",
      value: option.value,
      experiment,
    }),
  };
}

function semanticPresetTarget(
  target: LogsSubsetDeleteTarget,
): LogPresetDeleteTarget {
  return { experiment: target.experiment, preset: target.value };
}

export function useLogsDeletionState({
  active,
  enabled,
  selectedExperiments,
  onExperimentDeleted,
  onRunsDeleted,
}: {
  active: boolean;
  enabled: boolean;
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
  const experimentMutation = useMutation({
    mutationFn: ({
      experiment,
      mutation,
    }: {
      experiment: string;
      mutation: MutationRequestOptions;
    }) => deleteLogExperiment(experiment, mutation),
  });
  const planMutation = useMutation({ mutationFn: createLogPresetDeletePlan });
  const subsetMutation = useMutation({
    mutationFn: ({
      target,
      mutation,
    }: {
      target: LogPresetDeleteTarget;
      mutation: MutationRequestOptions;
    }) => deleteLogPreset(target, mutation),
  });
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
        mutation: createMutationRequestOptions(),
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
        const plan = await planMutation.mutateAsync(
          semanticPresetTarget(current.target),
        );
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
      const target = presetDeleteTarget({
        experiment: presetTargetExperiment,
        option,
      });
      experimentMutation.reset();
      planMutation.reset();
      subsetMutation.reset();
      const next: SubsetOperation = {
        kind: "preset",
        target,
        phase: "planning",
        plan: undefined,
        error: null,
        mutation: createMutationRequestOptions(),
      };
      void loadPlan(next).catch(() => undefined);
    },
    [
      enabled,
      experimentMutation,
      loadPlan,
      planMutation,
      presetTargetExperiment,
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
        const result = await experimentMutation.mutateAsync({
          experiment: current.option.value,
          mutation: current.mutation,
        });
        if (connectionGenerationRef.current === connectionGeneration) {
          onExperimentDeleted(result);
        }
      } else {
        const result = await subsetMutation.mutateAsync({
          target: semanticPresetTarget(current.target),
          mutation: current.mutation,
        });
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
      : {
          kind: operation.kind,
          option: operation.option,
          phase: operation.phase,
          error: operation.error,
        }
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
