import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  cancelTrainingJob,
  createMutationRequestOptions,
  createTrainingJob,
  type MutationRequestOptions,
  type TrainingJobCreateInput,
} from "@/lib/api";
import { useLogQueryCache } from "@/features/workbench/state/logs/use-log-query-cache";
import {
  terminalTrainingStatuses,
  type TrainingJobPolling,
} from "@/features/workbench/state/training/use-training-job-polling";
import { trainingQueryKeys } from "@/lib/query-keys";
import { errorMessage } from "@/lib/utils";

type TrainingMutationError =
  | { action: "create"; message: string }
  | { action: "cancel"; jobId: string; message: string };

type TrainingLogRefreshSnapshot = {
  jobId: string | null;
  logDir: string | null;
  terminalStatus: string | null;
};

type CreateTrainingCommand = {
  request: TrainingJobCreateInput;
  fingerprint: string;
  mutation: MutationRequestOptions;
};

type CancelTrainingCommand = {
  jobId: string;
  mutation: MutationRequestOptions;
};

const emptyLogRefreshSnapshot: TrainingLogRefreshSnapshot = {
  jobId: null,
  logDir: null,
  terminalStatus: null,
};

export function useTrainingJobExecution({
  enabled,
  polling,
}: {
  enabled: boolean;
  polling: TrainingJobPolling;
}) {
  const [mutationError, setMutationError] =
    useState<TrainingMutationError | null>(null);
  const mutationGenerationRef = useRef(0);
  const createCommandRef = useRef<CreateTrainingCommand | null>(null);
  const cancelCommandRef = useRef<CancelTrainingCommand | null>(null);
  const logRefreshSnapshotRef = useRef(emptyLogRefreshSnapshot);
  const queryClient = useQueryClient();
  const { invalidateLogLists, refreshAfterMutation } = useLogQueryCache();

  useEffect(() => {
    const previous = logRefreshSnapshotRef.current;
    const job = polling.job;
    if (!job) {
      logRefreshSnapshotRef.current = emptyLogRefreshSnapshot;
      return;
    }
    const terminalStatus = terminalTrainingStatuses.has(job.status)
      ? job.status
      : null;
    const jobChanged = previous.jobId !== job.id;
    const reachedTerminal = Boolean(
      terminalStatus &&
        (jobChanged || previous.terminalStatus !== terminalStatus),
    );
    const gainedLogDir = Boolean(
      job.logDir && (jobChanged || previous.logDir !== job.logDir),
    );
    logRefreshSnapshotRef.current = {
      jobId: job.id,
      logDir: job.logDir ?? null,
      terminalStatus,
    };
    if (reachedTerminal) {
      void refreshAfterMutation();
    } else if (gainedLogDir) {
      void invalidateLogLists();
    }
  }, [invalidateLogLists, polling.job, refreshAfterMutation]);
  const createMutation = useMutation({
    mutationFn: ({ request, mutation }: CreateTrainingCommand) =>
      createTrainingJob(request, mutation),
    onMutate: () => {
      setMutationError(null);
      return { generation: mutationGenerationRef.current };
    },
    onSuccess: (job, _command, context) => {
      if (context.generation !== mutationGenerationRef.current) {
        return;
      }
      queryClient.setQueryData(trainingQueryKeys.job(job.id), job);
      createCommandRef.current = null;
      setMutationError(null);
      polling.activateJob(job);
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
    mutationFn: ({ jobId, mutation }: CancelTrainingCommand) =>
      cancelTrainingJob(jobId, mutation),
    onMutate: () => {
      setMutationError(null);
      return { generation: mutationGenerationRef.current };
    },
    onSuccess: async (job, _command, context) => {
      if (context.generation !== mutationGenerationRef.current) {
        return;
      }
      const jobQueryKey = trainingQueryKeys.job(job.id);
      await queryClient.cancelQueries({ queryKey: jobQueryKey, exact: true });
      if (context.generation !== mutationGenerationRef.current) {
        return;
      }
      queryClient.setQueryData(jobQueryKey, job);
      cancelCommandRef.current = null;
      setMutationError(null);
      polling.activateJob(job);
      void refreshAfterMutation();
    },
    onError: (error, command, context) => {
      if (context?.generation !== mutationGenerationRef.current) {
        return;
      }
      setMutationError({
        action: "cancel",
        jobId: command.jobId,
        message: errorMessage(error),
      });
    },
  });
  const hasPendingMutation = createMutation.isPending || cancelMutation.isPending;
  const canLaunchRunPlan = Boolean(
    enabled && !polling.isRunning && !hasPendingMutation,
  );
  const mutationErrorMessage =
    mutationError?.action === "create"
      ? mutationError.message
      : mutationError?.action === "cancel" &&
          mutationError.jobId === polling.job?.id
        ? mutationError.message
        : "";
  const trainingError = mutationErrorMessage || polling.pollingError;
  const launchRunPlan = useCallback(
    (request: TrainingJobCreateInput) => {
      if (canLaunchRunPlan) {
        const fingerprint = JSON.stringify(request);
        const previous = createCommandRef.current;
        const command =
          mutationError?.action === "create" &&
          previous?.fingerprint === fingerprint
            ? previous
            : {
                request,
                fingerprint,
                mutation: createMutationRequestOptions(),
              };
        createCommandRef.current = command;
        setMutationError(null);
        createMutation.mutate(command);
      }
    },
    [canLaunchRunPlan, createMutation, mutationError?.action],
  );
  const cancelTraining = useCallback(() => {
    if (enabled && polling.job?.id && !cancelMutation.isPending) {
      const previous = cancelCommandRef.current;
      const command =
        mutationError?.action === "cancel" &&
        previous?.jobId === polling.job.id
          ? previous
          : {
              jobId: polling.job.id,
              mutation: createMutationRequestOptions(),
            };
      cancelCommandRef.current = command;
      cancelMutation.mutate(command);
    }
  }, [cancelMutation, enabled, mutationError?.action, polling.job?.id]);
  const resetTraining = useCallback(() => {
    if (hasPendingMutation) {
      return;
    }
    setMutationError(null);
    createCommandRef.current = null;
    cancelCommandRef.current = null;
    createMutation.reset();
    cancelMutation.reset();
    polling.resetObservedJob();
  }, [cancelMutation, createMutation, hasPendingMutation, polling]);
  const clearForConnectionChange = useCallback(() => {
    mutationGenerationRef.current += 1;
    logRefreshSnapshotRef.current = emptyLogRefreshSnapshot;
    createCommandRef.current = null;
    cancelCommandRef.current = null;
    setMutationError(null);
    createMutation.reset();
    cancelMutation.reset();
  }, [cancelMutation, createMutation]);

  return useMemo(
    () => ({
      job: polling.job,
      isRunning: polling.isRunning,
      canLaunchRunPlan,
      canResetTraining: Boolean(
        polling.job &&
          terminalTrainingStatuses.has(polling.job.status) &&
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
      canLaunchRunPlan,
      cancelMutation.isPending,
      cancelTraining,
      clearForConnectionChange,
      createMutation.isPending,
      hasPendingMutation,
      launchRunPlan,
      polling.isRunning,
      polling.job,
      resetTraining,
      trainingError,
    ],
  );
}

export type TrainingJobExecution = ReturnType<typeof useTrainingJobExecution>;
