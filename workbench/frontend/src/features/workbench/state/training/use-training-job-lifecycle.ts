import { useCallback, useMemo } from "react";
import { useTrainingJobExecution } from "@/features/workbench/state/training/use-training-job-execution";
import { useTrainingJobPolling } from "@/features/workbench/state/training/use-training-job-polling";

/** Composed compatibility hook; production keeps polling and execution split. */
export function useTrainingJobLifecycle({
  enabled = true,
  onJobStarted,
}: {
  enabled?: boolean;
  onJobStarted?: (logFolder: string) => void;
} = {}) {
  const polling = useTrainingJobPolling({ enabled, onJobStarted });
  const execution = useTrainingJobExecution({ enabled, polling });
  const clearExecution = execution.clearForConnectionChange;
  const clearPolling = polling.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    clearExecution();
    clearPolling();
  }, [clearExecution, clearPolling]);

  return useMemo(
    () => ({ ...execution, clearForConnectionChange }),
    [clearForConnectionChange, execution],
  );
}

export type TrainingJobLifecycle = ReturnType<typeof useTrainingJobLifecycle>;
