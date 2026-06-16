import { useCallback, useMemo, useState } from "react";
import { type TrainingJob } from "@/lib/api";

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
      setActiveTrainingJob(job);
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
