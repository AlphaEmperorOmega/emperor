import { useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchTrainingRunPlan,
  toTrainingRunPlanSubmitInput,
  type TrainingJobCreateInput,
  type TrainingRunPlan,
  type TrainingRunPlanCreateInput,
} from "@/lib/api/training-jobs";
import { type SearchAxis } from "@/lib/api/models";
import {
  configSectionsFields,
  type ConfigSection,
  type OverrideValues,
} from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { effectivePresetOverrides } from "@/features/workbench/state/runtime-defaults/runtime-defaults";
import { trainingQueryKeys } from "@/lib/query-keys";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  LARGE_GRID_RUN_THRESHOLD,
  buildEffectiveOverrides,
  buildTrainingSearchPayload,
  deriveTrainingSearchLockSummary,
  effectiveUnlockedTrainingSearch,
  estimateGridCombinations,
  estimatePlannedRuns,
  searchOverrideConflictKeys,
  selectedSearchAxisCount,
  trainingSearchModeLabel,
  unlockedSearchAxes,
  validateTrainingSearch,
  type TrainingSearchState,
} from "@/lib/training-search";
import { errorMessage } from "@/lib/utils";

type PendingTrainingRequest = {
  request: TrainingJobCreateInput;
  eligibility: object;
};

type TrainingRunPlanDraft = {
  modelPackage: {
    modelType: string;
    model: string;
    primaryPreset: string;
    selectedPresets: string[];
    selectedSnapshots: ConfigSnapshot[];
  };
  experiment: {
    task: string;
    datasets: string[];
    monitors: string[];
    logFolder: string;
    hasValidLogFolder: boolean;
  };
  runtimeDefaults: {
    sections: ConfigSection[];
    overrides: OverrideValues;
  };
  searchMetadata: {
    value: TrainingSearchState;
    axes: SearchAxis[];
    isLoading: boolean;
    update: (search: TrainingSearchState) => void;
  };
};

type TrainingRunPlanExecution = {
  activeRunPlan?: TrainingRunPlan | null;
  isJobRunning: boolean;
  canLaunch: boolean;
  launch: (request: TrainingJobCreateInput) => void;
};

type TrainingPlanStateInput = {
  draft: TrainingRunPlanDraft;
  availability: {
    trainingEnabled: boolean;
    protectedReadsEnabled: boolean;
  };
  execution: TrainingRunPlanExecution;
};

function buildTrainingDraftRequest({
  canPlan,
  draft,
  effectiveOverrides,
  searchPayload,
}: {
  canPlan: boolean;
  draft: TrainingRunPlanDraft;
  effectiveOverrides: OverrideValues;
  searchPayload: ReturnType<typeof buildTrainingSearchPayload>;
}): TrainingRunPlanCreateInput | null {
  if (!canPlan) {
    return null;
  }
  const { modelPackage, experiment } = draft;
  return {
    modelType: modelPackage.modelType,
    model: modelPackage.model,
    preset: modelPackage.primaryPreset,
    presets: modelPackage.selectedPresets,
    ...(experiment.task ? { experimentTask: experiment.task } : {}),
    datasets: experiment.datasets,
    overrides: effectiveOverrides,
    logFolder: experiment.logFolder,
    monitors: experiment.monitors,
    ...(modelPackage.selectedSnapshots.length > 0
      ? { snapshotIds: modelPackage.selectedSnapshots.map((snapshot) => snapshot.id) }
      : {}),
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
  const presets =
    runPlan.presets.length > 0 ? runPlan.presets : draftRequest.presets;
  const experimentTask = runPlan.experimentTask || draftRequest.experimentTask;
  const hasSnapshots = (draftRequest.snapshotIds?.length ?? 0) > 0;
  return {
    ...draftRequest,
    preset: runPlan.preset || draftRequest.preset,
    presets,
    ...(experimentTask ? { experimentTask } : {}),
    logFolder: draftRequest.logFolder ?? "",
    monitors: draftRequest.monitors ?? [],
    ...(hasSnapshots
      ? { snapshotRevisions: runPlan.snapshotRevisions ?? [] }
      : { runPlan: toTrainingRunPlanSubmitInput(runPlan) }),
  };
}

export function useTrainingPlanState({
  draft,
  availability,
  execution,
}: TrainingPlanStateInput) {
  const [planNonce, setPlanNonce] = useState(0);
  const [pendingTrainingRequest, setPendingTrainingRequest] =
    useState<PendingTrainingRequest | null>(null);
  const { modelPackage, experiment, runtimeDefaults, searchMetadata } = draft;
  const configFields = useMemo(
    () => configSectionsFields(runtimeDefaults.sections),
    [runtimeDefaults.sections],
  );
  const editablePresetOverrides = useMemo(
    () => effectivePresetOverrides(configFields, runtimeDefaults.overrides),
    [configFields, runtimeDefaults.overrides],
  );
  const activeConfigSnapshotCount = modelPackage.selectedSnapshots.length;
  const hasActiveConfigSnapshots = activeConfigSnapshotCount > 0;
  const baseTrainingSearch = hasActiveConfigSnapshots
    ? DEFAULT_TRAINING_SEARCH_STATE
    : searchMetadata.value;
  const searchLockSummary = useMemo(
    () => deriveTrainingSearchLockSummary(baseTrainingSearch, searchMetadata.axes),
    [baseTrainingSearch, searchMetadata.axes],
  );
  const effectiveTrainingSearch = useMemo(
    () => effectiveUnlockedTrainingSearch(baseTrainingSearch, searchMetadata.axes),
    [baseTrainingSearch, searchMetadata.axes],
  );
  const effectiveOverrides = useMemo<OverrideValues>(
    () =>
      hasActiveConfigSnapshots
        ? editablePresetOverrides
        : buildEffectiveOverrides(
            editablePresetOverrides,
            effectiveTrainingSearch,
          ),
    [
      editablePresetOverrides,
      effectiveTrainingSearch,
      hasActiveConfigSnapshots,
    ],
  );
  const searchConflictKeys = useMemo(
    () =>
      searchOverrideConflictKeys(
        editablePresetOverrides,
        effectiveTrainingSearch,
      ),
    [editablePresetOverrides, effectiveTrainingSearch],
  );
  const searchValidation = useMemo(
    () =>
      validateTrainingSearch(effectiveTrainingSearch, searchMetadata.axes, {
        allowEmptySelected: searchLockSummary.skippedSelectedAxisCount > 0,
      }),
    [
      effectiveTrainingSearch,
      searchLockSummary.skippedSelectedAxisCount,
      searchMetadata.axes,
    ],
  );
  const selectedPresetCount = modelPackage.selectedPresets.length;
  const activeSearchAxisCount = selectedSearchAxisCount(effectiveTrainingSearch);
  const searchPayload = useMemo(
    () =>
      hasActiveConfigSnapshots
        ? undefined
        : buildTrainingSearchPayload(effectiveTrainingSearch),
    [effectiveTrainingSearch, hasActiveConfigSnapshots],
  );
  const snapshotDraftReady = Boolean(
    availability.trainingEnabled &&
      hasActiveConfigSnapshots &&
      modelPackage.model &&
      (modelPackage.primaryPreset || modelPackage.selectedSnapshots.length > 0) &&
      experiment.datasets.length > 0,
  );
  const snapshotPlanInput = useMemo(
    () =>
      snapshotDraftReady
        ? {
            modelType: modelPackage.modelType,
            model: modelPackage.model,
            preset: modelPackage.primaryPreset,
            presets: modelPackage.selectedPresets,
            ...(experiment.task ? { experimentTask: experiment.task } : {}),
            datasets: experiment.datasets,
            overrides: editablePresetOverrides,
            logFolder: experiment.logFolder,
            monitors: experiment.monitors,
            snapshotIds: modelPackage.selectedSnapshots.map(
              (snapshot) => snapshot.id,
            ),
          }
        : null,
    [
      editablePresetOverrides,
      experiment.datasets,
      experiment.logFolder,
      experiment.monitors,
      experiment.task,
      modelPackage.model,
      modelPackage.modelType,
      modelPackage.primaryPreset,
      modelPackage.selectedPresets,
      modelPackage.selectedSnapshots,
      snapshotDraftReady,
    ],
  );
  const snapshotPlanQuery = useQuery({
    queryKey: ["training-config-snapshot-run-plan", snapshotPlanInput],
    queryFn: ({ signal }) => {
      if (!snapshotPlanInput) {
        throw new Error("Config Snapshot Run Plan input is not ready.");
      }
      return fetchTrainingRunPlan(snapshotPlanInput, { signal });
    },
    enabled:
      availability.protectedReadsEnabled && snapshotPlanInput !== null,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const snapshotRunPlan = snapshotPlanQuery.data;
  const estimatedRunCount = useMemo(
    () =>
      snapshotRunPlan?.summary.totalRuns ??
      (hasActiveConfigSnapshots
        ? (selectedPresetCount + activeConfigSnapshotCount) *
          experiment.datasets.length
        : estimatePlannedRuns(
            effectiveTrainingSearch,
            experiment.datasets.length,
            selectedPresetCount,
            {
              emptySearchRunsAsBase:
                searchLockSummary.skippedSelectedAxisCount > 0,
            },
          )),
    [
      activeConfigSnapshotCount,
      effectiveTrainingSearch,
      experiment.datasets.length,
      hasActiveConfigSnapshots,
      searchLockSummary.skippedSelectedAxisCount,
      selectedPresetCount,
      snapshotRunPlan,
    ],
  );
  const canPlan = Boolean(
    availability.trainingEnabled &&
      (hasActiveConfigSnapshots
        ? snapshotDraftReady && snapshotRunPlan
        : modelPackage.model &&
            modelPackage.primaryPreset &&
            selectedPresetCount > 0 &&
            experiment.datasets.length > 0 &&
            searchValidation.ready &&
            (effectiveTrainingSearch.mode === "off" ||
              !searchMetadata.isLoading)),
  );
  const draftRequest = useMemo(
    () =>
      buildTrainingDraftRequest({
        canPlan,
        draft,
        effectiveOverrides,
        searchPayload,
      }),
    [canPlan, draft, effectiveOverrides, searchPayload],
  );
  const planRequest = hasActiveConfigSnapshots ? null : draftRequest;
  const runPlanQueryKey = useMemo(
    () =>
      planRequest
        ? trainingQueryKeys.runPlan(planNonce, planRequest)
        : (["training-run-plan", planNonce, null] as const),
    [planNonce, planRequest],
  );
  const runPlanQuery = useQuery({
    queryKey: runPlanQueryKey,
    queryFn: ({ signal }) => {
      if (!planRequest) {
        throw new Error("Training run plan request is not ready.");
      }
      return fetchTrainingRunPlan(planRequest, { signal });
    },
    enabled: availability.protectedReadsEnabled && planRequest !== null,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const draftRunPlan = snapshotRunPlan ?? runPlanQuery.data;
  const activeRunPlan = execution.activeRunPlan ?? undefined;
  const displayRunPlan = activeRunPlan ?? draftRunPlan;
  const isPlanning = hasActiveConfigSnapshots
    ? snapshotDraftReady &&
      availability.protectedReadsEnabled &&
      (snapshotPlanQuery.isLoading || snapshotPlanQuery.isFetching)
    : canPlan && (runPlanQuery.isLoading || runPlanQuery.isFetching);
  const planError = hasActiveConfigSnapshots
    ? snapshotPlanQuery.isError
      ? errorMessage(snapshotPlanQuery.error)
      : ""
    : runPlanQuery.isError
      ? errorMessage(runPlanQuery.error)
      : "";
  const isDisplayPlanning = activeRunPlan ? false : isPlanning;
  const displayPlanError = activeRunPlan ? "" : planError;
  const canStart = Boolean(
    canPlan &&
      availability.protectedReadsEnabled &&
      experiment.hasValidLogFolder &&
      draftRunPlan &&
      !isPlanning &&
      !planError &&
      execution.canLaunch,
  );
  const displayedRunCount =
    draftRunPlan?.summary.totalRuns ?? estimatedRunCount;
  const requiresLargeGridConfirmation =
    effectiveTrainingSearch.mode === "grid" &&
    displayedRunCount > LARGE_GRID_RUN_THRESHOLD;
  const canResample = Boolean(
    availability.protectedReadsEnabled &&
      effectiveTrainingSearch.mode === "random" &&
      !snapshotRunPlan &&
      !execution.isJobRunning &&
      !activeRunPlan &&
      draftRunPlan,
  );
  const canRetry = Boolean(
    availability.protectedReadsEnabled &&
      (hasActiveConfigSnapshots
        ? snapshotPlanInput &&
          snapshotPlanQuery.isError &&
          !snapshotPlanQuery.isFetching
        : planRequest &&
          runPlanQuery.isError &&
          !runPlanQuery.isFetching),
  );
  const trainingRequest = useMemo(
    () => buildTrainingJobRequest({ draftRequest, runPlan: draftRunPlan }),
    [draftRequest, draftRunPlan],
  );
  const planRevision = useMemo(
    () =>
      JSON.stringify({
        runPlanQueryKey,
        snapshotRunPlan: snapshotRunPlan ?? null,
        canPlan,
        hasValidLogFolder: experiment.hasValidLogFolder,
      }),
    [
      canPlan,
      experiment.hasValidLogFolder,
      runPlanQueryKey,
      snapshotRunPlan,
    ],
  );
  const confirmationEligibility = useMemo(
    () => (canStart ? { planRevision } : null),
    [canStart, planRevision],
  );
  const activePendingTrainingRequest =
    pendingTrainingRequest?.eligibility === confirmationEligibility
      ? pendingTrainingRequest
      : null;

  const start = useCallback(() => {
    if (!canStart || !confirmationEligibility || !trainingRequest) {
      return;
    }
    if (requiresLargeGridConfirmation) {
      setPendingTrainingRequest({
        request: trainingRequest,
        eligibility: confirmationEligibility,
      });
      return;
    }
    execution.launch(trainingRequest);
  }, [
    canStart,
    confirmationEligibility,
    execution,
    requiresLargeGridConfirmation,
    trainingRequest,
  ]);
  const confirmLargeGridSearch = useCallback(() => {
    if (!activePendingTrainingRequest || !canStart) {
      setPendingTrainingRequest(null);
      return;
    }
    const { request } = activePendingTrainingRequest;
    setPendingTrainingRequest(null);
    execution.launch(request);
  }, [activePendingTrainingRequest, canStart, execution]);
  const cancelLargeGridSearch = useCallback(() => {
    setPendingTrainingRequest(null);
  }, []);
  const resample = useCallback(() => {
    if (!canResample || runPlanQuery.isFetching) {
      return;
    }
    setPendingTrainingRequest(null);
    setPlanNonce((current) => current + 1);
  }, [canResample, runPlanQuery.isFetching]);
  const retry = useCallback(() => {
    if (!canRetry) {
      return;
    }
    if (hasActiveConfigSnapshots) {
      void snapshotPlanQuery.refetch();
      return;
    }
    void runPlanQuery.refetch();
  }, [canRetry, hasActiveConfigSnapshots, runPlanQuery, snapshotPlanQuery]);
  const clearForConnectionChange = useCallback(() => {
    setPlanNonce(0);
    setPendingTrainingRequest(null);
  }, []);

  const unlockedAxes = useMemo(
    () => unlockedSearchAxes(searchMetadata.axes),
    [searchMetadata.axes],
  );
  const searchDisabledReason = execution.isJobRunning
    ? "Training setup is locked while the active job is running or queued."
    : hasActiveConfigSnapshots
      ? "Config snapshots train fixed variants; grid and random search are unavailable."
      : "";
  const lockWarning = [
    searchLockSummary.lockedAxesMessage,
    searchLockSummary.skippedSelectedAxisMessage,
  ]
    .filter(Boolean)
    .join(" ");

  return {
    activeConfigSnapshotCount,
    selectedPresetCount,
    datasetCount: experiment.datasets.length,
    displayRunPlan,
    displayedRunCount,
    isDisplayPlanning,
    displayPlanError,
    canStart,
    canResample,
    canRetry,
    isResampling: runPlanQuery.isFetching,
    presetCountLabel: `${selectedPresetCount} preset${
      selectedPresetCount === 1 ? "" : "s"
    }`,
    datasetCountLabel: `${experiment.datasets.length} dataset${
      experiment.datasets.length === 1 ? "" : "s"
    }`,
    search: {
      effective: effectiveTrainingSearch,
      axes: searchMetadata.axes,
      isLoading: searchMetadata.isLoading,
      conflictKeys: searchConflictKeys,
      validation: searchValidation,
      lockSummary: searchLockSummary,
      lockWarning,
      modeLabel: trainingSearchModeLabel(effectiveTrainingSearch.mode),
      activeAxisCount: activeSearchAxisCount,
      combinationCount: estimateGridCombinations(
        effectiveTrainingSearch.selectedValues,
      ),
      estimatedRunCount,
      unlockedAxisCount: unlockedAxes.length,
      unlockedAxes,
      disabledReason: searchDisabledReason,
      update: searchMetadata.update,
    },
    confirmation: {
      isRequired: requiresLargeGridConfirmation,
      isOpen: Boolean(activePendingTrainingRequest),
    },
    actions: {
      start,
      confirmLargeGridSearch,
      cancelLargeGridSearch,
      resample,
      retry,
    },
    clearForConnectionChange,
  };
}

export type TrainingRunPlanSearch = ReturnType<
  typeof useTrainingPlanState
>["search"];
