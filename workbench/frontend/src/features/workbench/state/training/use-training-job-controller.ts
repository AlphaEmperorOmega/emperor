import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  cancelTrainingJob,
  createTrainingJob,
  fetchTrainingJob,
  fetchTrainingRunPlan,
  type TrainingJob,
  type TrainingJobCreateInput,
  type TrainingRunPlan,
  type TrainingSearchCreateInput,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";
import {
  LARGE_GRID_RUN_THRESHOLD,
  type TrainingSearchState,
} from "@/lib/training-search";
import { useLogQueryCache } from "@/features/workbench/state/logs/use-log-query-cache";
import {
  trainingQueryKeys,
  type TrainingRunPlanQueryKeyInput,
} from "@/lib/query-keys";
import { errorMessage } from "@/lib/utils";
import {
  buildTrainingJobRequest,
  buildTrainingRunPlanRequest,
} from "@/features/workbench/state/training/training-request";

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

export function resolveTrainingLogRefresh(
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

type UseActiveTrainingJobProgressInput = {
  activeJobId: string | null;
  onJobChange: (job: TrainingJob | undefined) => void;
};

export function useActiveTrainingJobProgress({
  activeJobId,
  onJobChange,
}: UseActiveTrainingJobProgressInput) {
  const { invalidateLogLists, refreshAfterMutation } = useLogQueryCache();
  const logRefreshSnapshotRef = useRef<TrainingLogRefreshSnapshot>(
    emptyLogRefreshSnapshot,
  );
  const jobQuery = useQuery({
    queryKey: trainingQueryKeys.job(activeJobId),
    queryFn: () => fetchTrainingJob(activeJobId ?? ""),
    enabled: activeJobId !== null,
    refetchInterval: (query) => {
      const status = (query.state.data as TrainingJob | undefined)?.status;
      return status && terminalStatuses.has(status) ? false : 1000;
    },
  });
  const job = jobQuery.data;

  useEffect(() => {
    if (job || activeJobId === null) {
      onJobChange(job);
    }
  }, [activeJobId, job, onJobChange]);
  useEffect(() => {
    const { action, snapshot } = resolveTrainingLogRefresh(
      logRefreshSnapshotRef.current,
      job,
    );
    logRefreshSnapshotRef.current = snapshot;
    if (action === "details") {
      void refreshAfterMutation();
      return;
    }
    if (action === "lists") {
      void invalidateLogLists();
    }
  }, [job, invalidateLogLists, refreshAfterMutation]);

  return useMemo(
    () => ({
      isPolling: jobQuery.isFetching,
      progressError: jobQuery.isError ? errorMessage(jobQuery.error) : "",
    }),
    [jobQuery.error, jobQuery.isError, jobQuery.isFetching],
  );
}

export type ActiveTrainingJobProgress = ReturnType<
  typeof useActiveTrainingJobProgress
>;

type UseTrainingJobControllerInput = {
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedExperimentTask: string;
  selectedDatasets: string[];
  effectiveOverrides: OverrideValues;
  logFolder: string;
  selectedMonitors: string[];
  trainingSearch: TrainingSearchState;
  searchPayload?: TrainingSearchCreateInput;
  submittedRunPlan?: TrainingRunPlan;
  canPlan: boolean;
  hasValidLogFolder: boolean;
  plannedRunCount: number;
  activeTrainingJob: TrainingJob | undefined;
  progressError: string;
  onActiveJobIdChange: (jobId: string | null) => void;
  onJobChange: (job: TrainingJob | undefined) => void;
  onJobStarted: () => void;
};

export function useTrainingJobController({
  selectedModelType,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedExperimentTask,
  selectedDatasets,
  effectiveOverrides,
  logFolder,
  selectedMonitors,
  trainingSearch,
  searchPayload,
  submittedRunPlan,
  canPlan,
  hasValidLogFolder,
  plannedRunCount,
  activeTrainingJob,
  progressError,
  onActiveJobIdChange,
  onJobChange,
  onJobStarted,
}: UseTrainingJobControllerInput) {
  const [planNonce, setPlanNonce] = useState(0);
  const [pendingTrainingRequest, setPendingTrainingRequest] =
    useState<TrainingJobCreateInput | null>(null);
  const [mutationError, setMutationError] =
    useState<TrainingMutationError | null>(null);
  const queryClient = useQueryClient();
  const { refreshAfterMutation } = useLogQueryCache();
  const createMutation = useMutation({
    mutationFn: createTrainingJob,
    onMutate: () => {
      setMutationError(null);
    },
    onSuccess: (job) => {
      queryClient.setQueryData(trainingQueryKeys.job(job.id), job);
      setMutationError(null);
      onActiveJobIdChange(job.id);
      onJobChange(job);
      onJobStarted();
      void refreshAfterMutation();
    },
    onError: (error) => {
      setMutationError({ action: "create", message: errorMessage(error) });
    },
  });
  const cancelMutation = useMutation({
    mutationFn: cancelTrainingJob,
    onMutate: () => {
      setMutationError(null);
    },
    onSuccess: (job) => {
      queryClient.setQueryData(trainingQueryKeys.job(job.id), job);
      setMutationError(null);
      onActiveJobIdChange(job.id);
      onJobChange(job);
      void refreshAfterMutation();
    },
    onError: (error, jobId) => {
      setMutationError({
        action: "cancel",
        jobId,
        message: errorMessage(error),
      });
    },
  });
  const job = activeTrainingJob ?? createMutation.data ?? cancelMutation.data;

  const planRequest = useMemo(
    () =>
      buildTrainingRunPlanRequest({
        canPlan,
        selectedModelType,
        selectedModel,
        selectedPreset,
        selectedTrainingPresets,
        selectedExperimentTask,
        selectedDatasets,
        effectiveOverrides,
        logFolder,
        selectedMonitors,
        searchPayload,
        submittedRunPlan,
      }),
    [
      canPlan,
      effectiveOverrides,
      logFolder,
      searchPayload,
      selectedExperimentTask,
      selectedDatasets,
      selectedModelType,
      selectedModel,
      selectedMonitors,
      selectedPreset,
      selectedTrainingPresets,
      submittedRunPlan,
    ],
  );
  const planInputKey = useMemo<TrainingRunPlanQueryKeyInput>(
    () => ({
      modelType: selectedModelType,
      model: selectedModel,
      preset: selectedPreset,
      presets: selectedTrainingPresets,
      experimentTask: selectedExperimentTask,
      datasets: selectedDatasets,
      overrides: effectiveOverrides,
      logFolder,
      monitors: selectedMonitors,
      search: searchPayload,
      submittedRunPlan,
    }),
    [
      effectiveOverrides,
      logFolder,
      searchPayload,
      selectedExperimentTask,
      selectedDatasets,
      selectedModelType,
      selectedModel,
      selectedMonitors,
      selectedPreset,
      selectedTrainingPresets,
      submittedRunPlan,
    ],
  );
  const runPlanQuery = useQuery({
    queryKey: trainingQueryKeys.runPlan(planNonce, planInputKey),
    queryFn: () => {
      if (!planRequest) {
        throw new Error("Training run plan request is not ready.");
      }
      return fetchTrainingRunPlan(planRequest);
    },
    enabled: planRequest !== null,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const isRunning = job?.status === "running" || job?.status === "queued";
  const draftRunPlan = submittedRunPlan ?? runPlanQuery.data;
  const jobRunPlan = job?.runPlan ?? undefined;
  const progressRunPlan = jobRunPlan ?? draftRunPlan;
  const draftRunPlanSummary = draftRunPlan?.summary;
  const progressRunPlanSummary = progressRunPlan?.summary;
  const isPlanning =
    canPlan &&
    !submittedRunPlan &&
    (runPlanQuery.isLoading || runPlanQuery.isFetching);
  const planError =
    !submittedRunPlan && runPlanQuery.isError
      ? errorMessage(runPlanQuery.error)
      : "";
  const isProgressPlanning = jobRunPlan ? false : isPlanning;
  const progressPlanError = jobRunPlan ? "" : planError;
  const hasPendingMutation = createMutation.isPending || cancelMutation.isPending;
  const canStart = Boolean(
    canPlan &&
      hasValidLogFolder &&
      draftRunPlan &&
      !isPlanning &&
      !planError &&
      !isRunning &&
      !hasPendingMutation,
  );
  const displayedRunCount = draftRunPlanSummary?.totalRuns ?? plannedRunCount;
  const requiresLargeGridConfirmation =
    trainingSearch.mode === "grid" &&
    displayedRunCount > LARGE_GRID_RUN_THRESHOLD;
  const canResampleRunPlan = Boolean(
    trainingSearch.mode === "random" &&
      !submittedRunPlan &&
      !isRunning &&
      !jobRunPlan &&
      draftRunPlan,
  );
  const mutationErrorMessage =
    mutationError?.action === "create"
      ? mutationError.message
      : mutationError?.action === "cancel" && mutationError.jobId === job?.id
        ? mutationError.message
        : "";
  const trainingError = mutationErrorMessage || progressError;

  function trainingRequest(): TrainingJobCreateInput | null {
    return buildTrainingJobRequest({
      selectedModelType,
      selectedModel,
      selectedPreset,
      selectedTrainingPresets,
      selectedExperimentTask,
      selectedDatasets,
      effectiveOverrides,
      logFolder,
      runPlan: draftRunPlan,
      searchPayload,
      selectedMonitors,
    });
  }

  function submitTrainingRequest(request: TrainingJobCreateInput) {
    createMutation.mutate(request);
  }

  function startTraining() {
    if (!canStart) {
      return;
    }
    setMutationError(null);
    const request = trainingRequest();
    if (!request) {
      return;
    }
    if (requiresLargeGridConfirmation) {
      setPendingTrainingRequest(request);
      return;
    }
    submitTrainingRequest(request);
  }

  function confirmLargeGridSearch() {
    if (pendingTrainingRequest && canStart) {
      submitTrainingRequest(pendingTrainingRequest);
      setPendingTrainingRequest(null);
    }
  }

  function cancelLargeGridSearch() {
    setPendingTrainingRequest(null);
  }

  function cancelTraining() {
    if (job?.id && !cancelMutation.isPending) {
      cancelMutation.mutate(job.id);
    }
  }

  function resetTraining() {
    if (hasPendingMutation) {
      return;
    }
    setPendingTrainingRequest(null);
    setMutationError(null);
    createMutation.reset();
    cancelMutation.reset();
    onActiveJobIdChange(null);
    onJobChange(undefined);
  }

  function resampleRunPlan() {
    setPlanNonce((current) => current + 1);
  }

  return {
    job,
    draftRunPlan,
    progressRunPlan,
    progressRunPlanSummary,
    displayedRunCount,
    isPlanning,
    planError,
    isProgressPlanning,
    progressPlanError,
    isRunning,
    canResetTraining: Boolean(
      job && terminalStatuses.has(job.status) && !hasPendingMutation,
    ),
    canStart,
    canResampleRunPlan,
    isResampling: runPlanQuery.isFetching,
    isStarting: createMutation.isPending,
    isCancelling: cancelMutation.isPending,
    trainingError,
    requiresLargeGridConfirmation,
    showLargeGridConfirmation: pendingTrainingRequest !== null,
    startTraining,
    confirmLargeGridSearch,
    cancelLargeGridSearch,
    cancelTraining,
    resetTraining,
    resampleRunPlan,
  };
}
