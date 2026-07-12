import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchTrainingJob, type TrainingJob } from "@/lib/api";
import { trainingQueryKeys } from "@/lib/query-keys";
import { errorMessage } from "@/lib/utils";

export const terminalTrainingStatuses = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export function useTrainingJobPolling({
  enabled = true,
  onJobStarted,
}: {
  enabled?: boolean;
  onJobStarted?: (logFolder: string) => void;
} = {}) {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeTrainingJob, setActiveTrainingJob] = useState<TrainingJob>();
  const activeJobRef = useRef<TrainingJob | undefined>(undefined);
  const notifiedJobRef = useRef("");
  const queryClient = useQueryClient();

  const publishJob = useCallback(
    (job: TrainingJob | undefined) => {
      const current = activeJobRef.current;
      if (
        current &&
        job &&
        current.id === job.id &&
        terminalTrainingStatuses.has(current.status) &&
        !terminalTrainingStatuses.has(job.status)
      ) {
        return;
      }
      activeJobRef.current = job;
      setActiveTrainingJob(job);
      if (job?.logFolder) {
        const notificationIdentity = `${job.id}\u0000${job.logFolder}`;
        if (notifiedJobRef.current !== notificationIdentity) {
          notifiedJobRef.current = notificationIdentity;
          onJobStarted?.(job.logFolder);
        }
      }
    },
    [onJobStarted],
  );

  const activateJob = useCallback(
    (job: TrainingJob) => {
      setActiveJobId(job.id);
      publishJob(job);
    },
    [publishJob],
  );
  const jobQuery = useQuery({
    queryKey: trainingQueryKeys.job(activeJobId),
    queryFn: ({ signal }) => fetchTrainingJob(activeJobId ?? "", { signal }),
    enabled: enabled && activeJobId !== null,
    refetchInterval: (query) => {
      const status = (query.state.data as TrainingJob | undefined)?.status;
      return status && terminalTrainingStatuses.has(status) ? false : 1000;
    },
  });

  useEffect(() => {
    if (activeJobId === null) {
      publishJob(undefined);
      return;
    }
    if (jobQuery.data?.id === activeJobId) {
      publishJob(jobQuery.data);
    }
  }, [activeJobId, jobQuery.data, publishJob]);

  const resetObservedJob = useCallback(() => {
    const jobId = activeJobRef.current?.id;
    if (jobId) {
      void queryClient.cancelQueries({
        queryKey: trainingQueryKeys.job(jobId),
        exact: true,
      });
    }
    setActiveJobId(null);
    activeJobRef.current = undefined;
    notifiedJobRef.current = "";
    publishJob(undefined);
  }, [publishJob, queryClient]);
  const clearForConnectionChange = useCallback(() => {
    resetObservedJob();
  }, [resetObservedJob]);
  const isRunning =
    activeTrainingJob?.status === "running" ||
    activeTrainingJob?.status === "queued";

  return useMemo(
    () => ({
      activateJob,
      clearForConnectionChange,
      isRunning,
      job: activeTrainingJob,
      pollingError: jobQuery.isError ? errorMessage(jobQuery.error) : "",
      resetObservedJob,
    }),
    [
      activateJob,
      activeTrainingJob,
      clearForConnectionChange,
      isRunning,
      jobQuery.error,
      jobQuery.isError,
      resetObservedJob,
    ],
  );
}

export type TrainingJobPolling = ReturnType<typeof useTrainingJobPolling>;
