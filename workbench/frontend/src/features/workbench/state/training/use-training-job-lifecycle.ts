import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  cancelTrainingJob,
  createTrainingJob,
  fetchTrainingJob,
  type TrainingJob,
  type TrainingJobCreateInput,
} from "@/lib/api";
import { useLogQueryCache } from "@/features/workbench/state/logs/use-log-query-cache";
import { trainingQueryKeys } from "@/lib/query-keys";
import { errorMessage } from "@/lib/utils";

const terminalStatuses = new Set(["completed", "failed", "cancelled"]);

type TrainingLogRefreshSnapshot = {
  jobId: string | null;
  logDir: string | null;
  terminalStatus: string | null;
};

type TrainingLogRefreshAction = "none" | "lists" | "details";

type TrainingMutationError =
  | { action: "create"; message: string }
  | { action: "cancel"; jobId: string; message: string };

const emptyLogRefreshSnapshot: TrainingLogRefreshSnapshot = {
  jobId: null,
  logDir: null,
  terminalStatus: null,
};

function resolveTrainingLogRefresh(
  previous: TrainingLogRefreshSnapshot,
  job: TrainingJob | undefined,
): {
  action: TrainingLogRefreshAction;
  snapshot: TrainingLogRefreshSnapshot;
} {
  if (!job) {
    return { action: "none", snapshot: emptyLogRefreshSnapshot };
  }
  const terminalStatus = terminalStatuses.has(job.status) ? job.status : null;
  const snapshot = {
    jobId: job.id,
    logDir: job.logDir ?? null,
    terminalStatus,
  };
  const jobChanged = previous.jobId !== job.id;
  const reachedTerminal = Boolean(
    terminalStatus &&
      (jobChanged || previous.terminalStatus !== terminalStatus),
  );
  if (reachedTerminal) {
    return { action: "details", snapshot };
  }
  const gainedLogDir = Boolean(
    snapshot.logDir && (jobChanged || previous.logDir !== snapshot.logDir),
  );
  return {
    action: gainedLogDir ? "lists" : "none",
    snapshot,
  };
}

export function useTrainingJobLifecycle({
  enabled = true,
  onJobStarted,
}: {
  enabled?: boolean;
  onJobStarted?: (logFolder: string) => void;
} = {}) {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeTrainingJob, setActiveTrainingJob] = useState<TrainingJob>();
  const [mutationError, setMutationError] =
    useState<TrainingMutationError | null>(null);
  const activeJobRef = useRef<TrainingJob | undefined>(undefined);
  const notifiedJobRef = useRef("");
  const mutationGenerationRef = useRef(0);
  const logRefreshSnapshotRef = useRef<TrainingLogRefreshSnapshot>(
    emptyLogRefreshSnapshot,
  );
  const queryClient = useQueryClient();
  const { invalidateLogLists, refreshAfterMutation } = useLogQueryCache();

  const publishJob = useCallback(
    (job: TrainingJob | undefined) => {
      const current = activeJobRef.current;
      if (
        current &&
        job &&
        current.id === job.id &&
        terminalStatuses.has(current.status) &&
        !terminalStatuses.has(job.status)
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

  const jobQuery = useQuery({
    queryKey: trainingQueryKeys.job(activeJobId),
    queryFn: ({ signal }) => fetchTrainingJob(activeJobId ?? "", { signal }),
    enabled: enabled && activeJobId !== null,
    refetchInterval: (query) => {
      const status = (query.state.data as TrainingJob | undefined)?.status;
      return status && terminalStatuses.has(status) ? false : 1000;
    },
  });
  const polledJob = jobQuery.data;

  useEffect(() => {
    if (activeJobId === null) {
      publishJob(undefined);
      return;
    }
    if (polledJob?.id === activeJobId) {
      publishJob(polledJob);
    }
  }, [activeJobId, polledJob, publishJob]);
  useEffect(() => {
    const { action, snapshot } = resolveTrainingLogRefresh(
      logRefreshSnapshotRef.current,
      activeTrainingJob,
    );
    logRefreshSnapshotRef.current = snapshot;
    if (action === "details") {
      void refreshAfterMutation();
      return;
    }
    if (action === "lists") {
      void invalidateLogLists();
    }
  }, [activeTrainingJob, invalidateLogLists, refreshAfterMutation]);

  const createMutation = useMutation({
    mutationFn: createTrainingJob,
    onMutate: () => {
      setMutationError(null);
      return { generation: mutationGenerationRef.current };
    },
    onSuccess: (job, _request, context) => {
      if (context.generation !== mutationGenerationRef.current) {
        return;
      }
      queryClient.setQueryData(trainingQueryKeys.job(job.id), job);
      setMutationError(null);
      setActiveJobId(job.id);
      publishJob(job);
      void refreshAfterMutation();
    },
    onError: (error, _request, context) => {
      if (context?.generation !== mutationGenerationRef.current) {
        return;
      }
      setMutationError({ action: "create", message: errorMessage(error) });
    },
  });
  const cancelMutation = useMutation({
    mutationFn: cancelTrainingJob,
    onMutate: () => {
      setMutationError(null);
      return { generation: mutationGenerationRef.current };
    },
    onSuccess: async (job, _jobId, context) => {
      if (context.generation !== mutationGenerationRef.current) {
        return;
      }
      const jobQueryKey = trainingQueryKeys.job(job.id);
      await queryClient.cancelQueries({ queryKey: jobQueryKey, exact: true });
      if (context.generation !== mutationGenerationRef.current) {
        return;
      }
      queryClient.setQueryData(jobQueryKey, job);
      setMutationError(null);
      setActiveJobId(job.id);
      publishJob(job);
      void refreshAfterMutation();
    },
    onError: (error, jobId, context) => {
      if (context?.generation !== mutationGenerationRef.current) {
        return;
      }
      setMutationError({
        action: "cancel",
        jobId,
        message: errorMessage(error),
      });
    },
  });

  const isRunning =
    activeTrainingJob?.status === "running" ||
    activeTrainingJob?.status === "queued";
  const hasPendingMutation = createMutation.isPending || cancelMutation.isPending;
  const canLaunchRunPlan = Boolean(enabled && !isRunning && !hasPendingMutation);
  const mutationErrorMessage =
    mutationError?.action === "create"
      ? mutationError.message
      : mutationError?.action === "cancel" &&
          mutationError.jobId === activeTrainingJob?.id
        ? mutationError.message
        : "";
  const trainingError =
    mutationErrorMessage || (jobQuery.isError ? errorMessage(jobQuery.error) : "");

  const launchRunPlan = useCallback(
    (request: TrainingJobCreateInput) => {
      if (!canLaunchRunPlan) {
        return;
      }
      setMutationError(null);
      createMutation.mutate(request);
    },
    [canLaunchRunPlan, createMutation],
  );
  const cancelTraining = useCallback(() => {
    if (enabled && activeTrainingJob?.id && !cancelMutation.isPending) {
      cancelMutation.mutate(activeTrainingJob.id);
    }
  }, [activeTrainingJob?.id, cancelMutation, enabled]);
  const resetTraining = useCallback(() => {
    if (hasPendingMutation) {
      return;
    }
    const jobId = activeJobRef.current?.id;
    if (jobId) {
      void queryClient.cancelQueries({
        queryKey: trainingQueryKeys.job(jobId),
        exact: true,
      });
    }
    setMutationError(null);
    createMutation.reset();
    cancelMutation.reset();
    setActiveJobId(null);
    activeJobRef.current = undefined;
    notifiedJobRef.current = "";
    publishJob(undefined);
  }, [
    cancelMutation,
    createMutation,
    hasPendingMutation,
    publishJob,
    queryClient,
  ]);
  const clearForConnectionChange = useCallback(() => {
    mutationGenerationRef.current += 1;
    const jobId = activeJobRef.current?.id;
    if (jobId) {
      void queryClient.cancelQueries({
        queryKey: trainingQueryKeys.job(jobId),
        exact: true,
      });
    }
    setMutationError(null);
    createMutation.reset();
    cancelMutation.reset();
    setActiveJobId(null);
    activeJobRef.current = undefined;
    notifiedJobRef.current = "";
    logRefreshSnapshotRef.current = emptyLogRefreshSnapshot;
    publishJob(undefined);
  }, [cancelMutation, createMutation, publishJob, queryClient]);

  return useMemo(
    () => ({
      job: activeTrainingJob,
      isRunning,
      canLaunchRunPlan,
      canResetTraining: Boolean(
        activeTrainingJob &&
          terminalStatuses.has(activeTrainingJob.status) &&
          !hasPendingMutation,
      ),
      isStarting: createMutation.isPending,
      isCancelling: cancelMutation.isPending,
      trainingError,
      launchRunPlan,
      cancelTraining,
      resetTraining,
      clearForConnectionChange,
    }),
    [
      activeTrainingJob,
      canLaunchRunPlan,
      cancelMutation.isPending,
      cancelTraining,
      clearForConnectionChange,
      createMutation.isPending,
      hasPendingMutation,
      isRunning,
      launchRunPlan,
      resetTraining,
      trainingError,
    ],
  );
}

export type TrainingJobLifecycle = ReturnType<typeof useTrainingJobLifecycle>;
