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
  type TrainingRunPlanCreateInput,
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
} from "@/lib/query-keys";
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

type PendingTrainingRequest = {
  request: TrainingJobCreateInput;
  revision: string;
};

type TrainingDraftRequestInput = {
  canPlan: boolean;
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedExperimentTask?: string;
  selectedDatasets: string[];
  effectiveOverrides: OverrideValues;
  logFolder: string;
  selectedMonitors: string[];
  searchPayload?: TrainingSearchCreateInput;
};

function buildTrainingDraftRequest({
  canPlan,
  selectedModelType,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedExperimentTask = "",
  selectedDatasets,
  effectiveOverrides,
  logFolder,
  selectedMonitors,
  searchPayload,
}: TrainingDraftRequestInput): TrainingRunPlanCreateInput | null {
  if (!canPlan) {
    return null;
  }
  return {
    modelType: selectedModelType,
    model: selectedModel,
    preset: selectedPreset,
    presets: selectedTrainingPresets,
    ...(selectedExperimentTask ? { experimentTask: selectedExperimentTask } : {}),
    datasets: selectedDatasets,
    overrides: effectiveOverrides,
    logFolder,
    monitors: selectedMonitors,
    ...(searchPayload ? { search: searchPayload } : {}),
  };
}

function buildTrainingJobRequest({
  draftRequest,
  runPlan,
}: {
  draftRequest: TrainingRunPlanCreateInput | null;
  runPlan?: TrainingRunPlan;
}): TrainingJobCreateInput | null {
  if (!draftRequest || !runPlan) {
    return null;
  }
  const presets = runPlan.presets.length > 0
    ? runPlan.presets
    : draftRequest.presets;
  const experimentTask = runPlan.experimentTask || draftRequest.experimentTask;
  return {
    ...draftRequest,
    preset: runPlan.preset || draftRequest.preset,
    presets,
    ...(experimentTask ? { experimentTask } : {}),
    logFolder: draftRequest.logFolder ?? "",
    monitors: draftRequest.monitors ?? [],
    runPlan,
  };
}

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

type UseActiveTrainingJobProgressInput = {
  activeJobId: string | null;
  onJobChange: (job: TrainingJob | undefined) => void;
  enabled?: boolean;
};

export function useActiveTrainingJobProgress({
  activeJobId,
  onJobChange,
  enabled = true,
}: UseActiveTrainingJobProgressInput) {
  const { invalidateLogLists, refreshAfterMutation } = useLogQueryCache();
  const logRefreshSnapshotRef = useRef<TrainingLogRefreshSnapshot>(
    emptyLogRefreshSnapshot,
  );
  const jobQuery = useQuery({
    queryKey: trainingQueryKeys.job(activeJobId),
    queryFn: ({ signal }) =>
      fetchTrainingJob(activeJobId ?? "", { signal }),
    enabled: enabled && activeJobId !== null,
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
      progressError: jobQuery.isError ? errorMessage(jobQuery.error) : "",
    }),
    [jobQuery.error, jobQuery.isError],
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
  protectedReadsEnabled?: boolean;
  hasValidLogFolder: boolean;
  plannedRunCount: number;
  activeTrainingJob: TrainingJob | undefined;
  progressError: string;
  onActiveJobIdChange: (jobId: string | null) => void;
  onJobChange: (job: TrainingJob | undefined) => void;
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
  protectedReadsEnabled = true,
  hasValidLogFolder,
  plannedRunCount,
  activeTrainingJob,
  progressError,
  onActiveJobIdChange,
  onJobChange,
}: UseTrainingJobControllerInput) {
  const [planNonce, setPlanNonce] = useState(0);
  const [pendingTrainingRequest, setPendingTrainingRequest] =
    useState<PendingTrainingRequest | null>(null);
  const [mutationError, setMutationError] =
    useState<TrainingMutationError | null>(null);
  const mutationGenerationRef = useRef(0);
  const queryClient = useQueryClient();
  const { refreshAfterMutation } = useLogQueryCache();
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
      onActiveJobIdChange(job.id);
      onJobChange(job);
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
      await queryClient.cancelQueries({
        queryKey: jobQueryKey,
        exact: true,
      });
      if (context.generation !== mutationGenerationRef.current) {
        return;
      }
      queryClient.setQueryData(jobQueryKey, job);
      setMutationError(null);
      onActiveJobIdChange(job.id);
      onJobChange(job);
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
  const job = activeTrainingJob ?? createMutation.data ?? cancelMutation.data;

  const draftRequest = useMemo(
    () =>
      buildTrainingDraftRequest({
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
    ],
  );
  const planRequest = submittedRunPlan ? null : draftRequest;
  const runPlanQueryKey = useMemo(
    () =>
      planRequest
        ? trainingQueryKeys.runPlan(planNonce, planRequest)
        : (["training-run-plan", planNonce, null] as const),
    [planNonce, planRequest],
  );
  const planRevision = useMemo(
    () =>
      JSON.stringify({
        runPlanQueryKey,
        submittedRunPlan: submittedRunPlan ?? null,
        canPlan,
        hasValidLogFolder,
      }),
    [
      canPlan,
      hasValidLogFolder,
      runPlanQueryKey,
      submittedRunPlan,
    ],
  );
  const currentPlanRevisionRef = useRef(planRevision);
  currentPlanRevisionRef.current = planRevision;
  const runPlanQuery = useQuery({
    queryKey: runPlanQueryKey,
    queryFn: ({ signal }) => {
      if (!planRequest) {
        throw new Error("Training run plan request is not ready.");
      }
      return fetchTrainingRunPlan(planRequest, { signal });
    },
    enabled: protectedReadsEnabled && planRequest !== null,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const isRunning = job?.status === "running" || job?.status === "queued";
  const draftRunPlan = submittedRunPlan ?? runPlanQuery.data;
  const jobRunPlan = job?.runPlan ?? undefined;
  const progressRunPlan = jobRunPlan ?? draftRunPlan;
  const draftRunPlanSummary = draftRunPlan?.summary;
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
      protectedReadsEnabled &&
      hasValidLogFolder &&
      draftRunPlan &&
      !isPlanning &&
      !planError &&
      !isRunning &&
      !hasPendingMutation,
  );
  const canStartRef = useRef(canStart);
  canStartRef.current = canStart;
  const displayedRunCount = draftRunPlanSummary?.totalRuns ?? plannedRunCount;
  const requiresLargeGridConfirmation =
    trainingSearch.mode === "grid" &&
    displayedRunCount > LARGE_GRID_RUN_THRESHOLD;
  const canResampleRunPlan = Boolean(
    protectedReadsEnabled &&
      trainingSearch.mode === "random" &&
      !submittedRunPlan &&
      !isRunning &&
      !jobRunPlan &&
      draftRunPlan,
  );
  const canRetryRunPlan = Boolean(
    protectedReadsEnabled &&
      planRequest &&
      !submittedRunPlan &&
      runPlanQuery.isError &&
      !runPlanQuery.isFetching,
  );
  const mutationErrorMessage =
    mutationError?.action === "create"
      ? mutationError.message
      : mutationError?.action === "cancel" && mutationError.jobId === job?.id
        ? mutationError.message
        : "";
  const trainingError = mutationErrorMessage || progressError;

  useEffect(() => {
    setPendingTrainingRequest((pending) =>
      pending && pending.revision === planRevision && canStart
        ? pending
        : null,
    );
  }, [canStart, planRevision]);

  function trainingRequest(): TrainingJobCreateInput | null {
    return buildTrainingJobRequest({
      draftRequest,
      runPlan: draftRunPlan,
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
      setPendingTrainingRequest({ request, revision: planRevision });
      return;
    }
    submitTrainingRequest(request);
  }

  function confirmLargeGridSearch() {
    if (
      !pendingTrainingRequest ||
      pendingTrainingRequest.revision !== currentPlanRevisionRef.current ||
      !canStartRef.current
    ) {
      setPendingTrainingRequest(null);
      return;
    }
    const { request } = pendingTrainingRequest;
    setPendingTrainingRequest(null);
    submitTrainingRequest(request);
  }

  function cancelLargeGridSearch() {
    setPendingTrainingRequest(null);
  }

  function cancelTraining() {
    if (protectedReadsEnabled && job?.id && !cancelMutation.isPending) {
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
    if (!canResampleRunPlan || runPlanQuery.isFetching) {
      return;
    }
    setPendingTrainingRequest(null);
    setPlanNonce((current) => current + 1);
  }

  function retryRunPlan() {
    if (!protectedReadsEnabled || !canRetryRunPlan) {
      return;
    }
    void runPlanQuery.refetch();
  }

  function clearForConnectionChange() {
    mutationGenerationRef.current += 1;
    setPlanNonce(0);
    setPendingTrainingRequest(null);
    setMutationError(null);
    createMutation.reset();
    cancelMutation.reset();
  }

  return {
    job,
    progressRunPlan,
    displayedRunCount,
    isProgressPlanning,
    progressPlanError,
    isRunning,
    canResetTraining: Boolean(
      job && terminalStatuses.has(job.status) && !hasPendingMutation,
    ),
    canStart,
    canResampleRunPlan,
    canRetryRunPlan,
    isResampling: runPlanQuery.isFetching,
    isStarting: createMutation.isPending,
    isCancelling: cancelMutation.isPending,
    trainingError,
    requiresLargeGridConfirmation,
    showLargeGridConfirmation: Boolean(
      pendingTrainingRequest?.revision === planRevision && canStart,
    ),
    startTraining,
    confirmLargeGridSearch,
    cancelLargeGridSearch,
    cancelTraining,
    resetTraining,
    resampleRunPlan,
    retryRunPlan,
    clearForConnectionChange,
  };
}
