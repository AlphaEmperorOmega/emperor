import { useCallback, useEffect, useMemo, useState } from "react";
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
  const [submittedPlanKey, setSubmittedPlanKey] = useState("");
  const [pendingTrainingRequest, setPendingTrainingRequest] =
    useState<TrainingJobCreateInput | null>(null);
  const queryClient = useQueryClient();
  const refreshLogWorkspaceQueries = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ["log-experiments"] });
    void queryClient.invalidateQueries({ queryKey: ["log-runs"] });
    queryClient.removeQueries({ queryKey: ["log-tags"] });
    queryClient.removeQueries({ queryKey: ["log-scalars"] });
  }, [queryClient]);
  const createMutation = useMutation({
    mutationFn: createTrainingJob,
    onSuccess: (job) => {
      onActiveJobIdChange(job.id);
      onJobChange(job);
      onJobStarted();
      refreshLogWorkspaceQueries();
    },
  });
  const jobQuery = useQuery({
    queryKey: ["training-job", activeJobId],
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
      refreshLogWorkspaceQueries();
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
      refreshLogWorkspaceQueries();
    }
  }, [job, refreshLogWorkspaceQueries]);

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
    queryKey: ["training-run-plan", planNonce, planInputKey],
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
  const activeJobRunPlan =
    job?.runPlan && (isRunning || submittedPlanKey === planInputKey)
      ? job.runPlan
      : undefined;
  const currentRunPlan = activeJobRunPlan ?? submittedRunPlan ?? runPlanQuery.data;
  const runPlanSummary = currentRunPlan?.summary;
  const isPlanning =
    canPlan &&
    !submittedRunPlan &&
    (runPlanQuery.isLoading || runPlanQuery.isFetching);
  const planError =
    !submittedRunPlan && runPlanQuery.isError
      ? errorMessage(runPlanQuery.error)
      : "";
  const canStart = Boolean(
    canPlan &&
      hasValidLogFolder &&
      currentRunPlan &&
      !isPlanning &&
      !planError &&
      !isRunning &&
      !createMutation.isPending,
  );
  const displayedRunCount = runPlanSummary?.totalRuns ?? plannedRunCount;
  const canResampleRunPlan = Boolean(
    trainingSearch.mode === "random" &&
      !submittedRunPlan &&
      !isRunning &&
      !activeJobRunPlan &&
      runPlanQuery.data,
  );
  const trainingError =
    createMutation.isError || jobQuery.isError || cancelMutation.isError
      ? errorMessage(
          createMutation.error ?? jobQuery.error ?? cancelMutation.error,
        )
      : "";

  function trainingRequest(): TrainingJobCreateInput | null {
    if (!currentRunPlan) {
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
      runPlan: currentRunPlan,
    };
  }

  function submitTrainingRequest(request: TrainingJobCreateInput) {
    setSubmittedPlanKey(planInputKey);
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
    currentRunPlan,
    runPlanSummary,
    displayedRunCount,
    isPlanning,
    planError,
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
