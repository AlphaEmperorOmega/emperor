import { useCallback, useMemo, useState } from "react";
import { type TrainingJob } from "@/lib/api";

const terminalStatuses = new Set(["completed", "failed", "cancelled"]);

type ActiveTrainingJobStateInput = {
  onJobStarted?: (logFolder: string) => void;
};

export function useActiveTrainingJobState({
  onJobStarted,
}: ActiveTrainingJobStateInput = {}) {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeTrainingJob, setActiveTrainingJob] = useState<
    TrainingJob | undefined
  >();

  const handleTrainingJobChange = useCallback(
    (job: TrainingJob | undefined) => {
      setActiveTrainingJob((current) => {
        if (
          current &&
          job &&
          current.id === job.id &&
          terminalStatuses.has(current.status) &&
          !terminalStatuses.has(job.status)
        ) {
          return current;
        }
        return job;
      });
      if (job?.logFolder) {
        onJobStarted?.(job.logFolder);
      }
    },
    [onJobStarted],
  );

  return useMemo(
    () => ({
      activeJobId,
      setActiveJobId,
      activeTrainingJob,
      onJobChange: handleTrainingJobChange,
    }),
    [activeJobId, activeTrainingJob, handleTrainingJobChange],
  );
}
