import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
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
import { useLogQueryCache } from "@/hooks/use-log-query-cache";
import { trainingQueryKeys } from "@/lib/query-keys";
import { errorMessage } from "@/lib/utils";

const terminalStatuses = new Set(["completed", "failed", "cancelled"]);

type UseTrainingJobControllerInput = {
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
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
  activeJobId: string | null;
  onActiveJobIdChange: (jobId: string | null) => void;
  onJobChange: (job: TrainingJob | undefined) => void;
  onJobStarted: () => void;
};

export function useTrainingJobController({
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
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
  activeJobId,
  onActiveJobIdChange,
  onJobChange,
  onJobStarted,
}: UseTrainingJobControllerInput) {
  const [planNonce, setPlanNonce] = useState(0);
  const [pendingTrainingRequest, setPendingTrainingRequest] =
    useState<TrainingJobCreateInput | null>(null);
  const { refreshAfterMutation } = useLogQueryCache();
  const createMutation = useMutation({
    mutationFn: createTrainingJob,
    onSuccess: (job) => {
      onActiveJobIdChange(job.id);
      onJobChange(job);
      onJobStarted();
      void refreshAfterMutation();
    },
  });
  const jobQuery = useQuery({
    queryKey: trainingQueryKeys.job(activeJobId),
    queryFn: () => fetchTrainingJob(activeJobId ?? ""),
    enabled: activeJobId !== null,
    refetchInterval: (query) => {
      const status = (query.state.data as TrainingJob | undefined)?.status;
      return status && terminalStatuses.has(status) ? false : 1000;
    },
  });
  const cancelMutation = useMutation({
    mutationFn: cancelTrainingJob,
    onSuccess: (job) => {
      onActiveJobIdChange(job.id);
      onJobChange(job);
      void refreshAfterMutation();
    },
  });
  const job = jobQuery.data ?? createMutation.data;

  useEffect(() => {
    onJobChange(job);
  }, [job, onJobChange]);
  useEffect(() => {
    if (!job) {
      return;
    }
    if (job.logDir || terminalStatuses.has(job.status)) {
      void refreshAfterMutation();
    }
  }, [job, refreshAfterMutation]);

  const planRequest = useMemo(
    () =>
      canPlan && !submittedRunPlan
        ? {
            model: selectedModel,
            preset: selectedPreset,
            presets: selectedTrainingPresets,
            datasets: selectedDatasets,
            overrides: effectiveOverrides,
            logFolder,
            ...(searchPayload ? { search: searchPayload } : {}),
          }
        : null,
    [
      canPlan,
      effectiveOverrides,
      logFolder,
      searchPayload,
      selectedDatasets,
      selectedModel,
      selectedPreset,
      selectedTrainingPresets,
      submittedRunPlan,
    ],
  );
  const planInputKey = useMemo(
    () =>
      JSON.stringify({
        model: selectedModel,
        preset: selectedPreset,
        presets: selectedTrainingPresets,
        datasets: selectedDatasets,
        overrides: effectiveOverrides,
        logFolder,
        search: searchPayload ?? null,
        submittedRunPlan: submittedRunPlan ?? null,
      }),
    [
      effectiveOverrides,
      logFolder,
      searchPayload,
      selectedDatasets,
      selectedModel,
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
  const canStart = Boolean(
    canPlan &&
      hasValidLogFolder &&
      draftRunPlan &&
      !isPlanning &&
      !planError &&
      !isRunning &&
      !createMutation.isPending,
  );
  const displayedRunCount = draftRunPlanSummary?.totalRuns ?? plannedRunCount;
  const canResampleRunPlan = Boolean(
    trainingSearch.mode === "random" &&
      !submittedRunPlan &&
      !isRunning &&
      !jobRunPlan &&
      draftRunPlan,
  );
  const trainingError =
    createMutation.isError || jobQuery.isError || cancelMutation.isError
      ? errorMessage(
          createMutation.error ?? jobQuery.error ?? cancelMutation.error,
        )
      : "";

  function trainingRequest(): TrainingJobCreateInput | null {
    if (!draftRunPlan) {
      return null;
    }
    return {
      model: selectedModel,
      preset: selectedPreset,
      presets: selectedTrainingPresets,
      datasets: selectedDatasets,
      overrides: effectiveOverrides,
      logFolder,
      monitors: selectedMonitors,
      ...(searchPayload ? { search: searchPayload } : {}),
      runPlan: draftRunPlan,
    };
  }

  function submitTrainingRequest(request: TrainingJobCreateInput) {
    createMutation.mutate(request);
  }

  function startTraining() {
    const request = trainingRequest();
    if (!request) {
      return;
    }
    if (
      trainingSearch.mode === "grid" &&
      displayedRunCount > LARGE_GRID_RUN_THRESHOLD
    ) {
      setPendingTrainingRequest(request);
      return;
    }
    submitTrainingRequest(request);
  }

  function confirmLargeGridSearch() {
    if (pendingTrainingRequest) {
      submitTrainingRequest(pendingTrainingRequest);
      setPendingTrainingRequest(null);
    }
  }

  function cancelLargeGridSearch() {
    setPendingTrainingRequest(null);
  }

  function cancelTraining() {
    if (job?.id) {
      cancelMutation.mutate(job.id);
    }
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
    canStart,
    canResampleRunPlan,
    isResampling: runPlanQuery.isFetching,
    isStarting: createMutation.isPending,
    isCancelling: cancelMutation.isPending,
    trainingError,
    showLargeGridConfirmation: pendingTrainingRequest !== null,
    startTraining,
    confirmLargeGridSearch,
    cancelLargeGridSearch,
    cancelTraining,
    resampleRunPlan,
  };
}
