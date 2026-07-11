import { useCallback, useMemo, useRef, useState } from "react";
import { isCancelledError, useQueryClient } from "@tanstack/react-query";
import {
  inspectModel,
  type InspectResponse,
} from "@/lib/api";
import { overrideDigest, type OverrideValues } from "@/lib/config";
import { workbenchQueryKeys } from "@/lib/query-keys";
import type { HistoricalExperimentTarget } from "@/features/workbench/state/target/_inspection-target-state";

export type TargetMode = "preset" | "snapshot" | "experiment";

export type InspectionPreviewRequest = {
  modelType: string;
  model: string;
  preset: string;
  experimentTask?: string | null;
  dataset?: string;
  overrides: OverrideValues;
  targetMode?: TargetMode;
  targetId?: string;
  logRunId?: string;
};

export function inspectionTargetKey({
  modelType,
  model,
  preset,
  experimentTask,
  dataset,
  targetMode = "preset",
  targetId = preset,
  overrides,
}: InspectionPreviewRequest) {
  return [
    modelType,
    model,
    preset,
    experimentTask ?? "",
    dataset ?? "",
    targetMode,
    targetId,
    overrideDigest(overrides),
  ].join("\u0000");
}

export function resolveInspectionTarget({
  selectedTargetMode,
  selectedSnapshotId,
  selectedExperimentTarget,
  selectedPreset,
  selectedExperimentTask,
  selectedDatasets,
}: {
  selectedTargetMode: TargetMode;
  selectedSnapshotId: string;
  selectedExperimentTarget: HistoricalExperimentTarget | null;
  selectedPreset: string;
  selectedExperimentTask: string;
  selectedDatasets: string[];
}) {
  const catalogDataset = selectedDatasets[0] ?? "";
  if (selectedTargetMode === "snapshot" && selectedSnapshotId) {
    return {
      targetMode: "snapshot" as const,
      targetId: selectedSnapshotId,
      preset: selectedPreset,
      experimentTask: selectedExperimentTask,
      dataset: catalogDataset,
    };
  }
  if (selectedTargetMode === "experiment" && selectedExperimentTarget) {
    return {
      targetMode: "experiment" as const,
      targetId: selectedExperimentTarget.runId,
      preset: selectedExperimentTarget.preset,
      experimentTask: selectedExperimentTarget.experimentTask ?? "",
      dataset: selectedExperimentTarget.dataset,
    };
  }
  return {
    targetMode: "preset" as const,
    targetId: selectedPreset,
    preset: selectedPreset,
    experimentTask: selectedExperimentTask,
    dataset: catalogDataset,
  };
}

function previewInspectionPayload(request: InspectionPreviewRequest) {
  const payload = {
    modelType: request.modelType,
    model: request.model,
    preset: request.preset,
    overrides: request.overrides,
    ...(request.experimentTask
      ? { experimentTask: request.experimentTask }
      : {}),
    ...(request.dataset ? { dataset: request.dataset } : {}),
  };
  return request.logRunId ? { ...payload, logRunId: request.logRunId } : payload;
}

const PREVIEW_INSPECTION_STALE_TIME_MS = 5 * 60_000;

function presetIdentityKey(preset: string) {
  return preset.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === "AbortError";
}

function presetIdentityMatches(left: string, right: string) {
  return left === right || presetIdentityKey(left) === presetIdentityKey(right);
}

function assertPreviewIdentity<
  Response extends Pick<InspectResponse, "modelType" | "model" | "preset">,
>(request: InspectionPreviewRequest, response: Response) {
  if (
    response.modelType === request.modelType &&
    response.model === request.model &&
    presetIdentityMatches(response.preset, request.preset)
  ) {
    return response;
  }
  throw new Error(
    `Inspection response identity mismatch: requested ${request.modelType}/${request.model}/${request.preset}, received ${response.modelType}/${response.model}/${response.preset}.`,
  );
}

export function useInspectionPreviewState() {
  const queryClient = useQueryClient();
  const [response, setResponse] = useState<InspectResponse | undefined>();
  const [request, setRequest] = useState<InspectionPreviewRequest | null>(null);
  const [isBuilding, setIsBuilding] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const requestRevisionRef = useRef(0);
  const connectionGenerationRef = useRef(0);
  const inFlightQueryKeyRef = useRef<string | null>(null);

  const clear = useCallback(() => {
    requestRevisionRef.current += 1;
    void queryClient.cancelQueries({
      queryKey: workbenchQueryKeys.previewInspections(),
    });
    setResponse(undefined);
    setRequest(null);
    inFlightQueryKeyRef.current = null;
    setIsBuilding(false);
    setError(null);
  }, [queryClient]);

  const execute = useCallback(
    (
      nextRequest: InspectionPreviewRequest,
      { force = false } = {},
    ) => {
      const semanticKey = inspectionTargetKey(nextRequest);
      const queryKeyValue = `${connectionGenerationRef.current}\u0000${semanticKey}`;
      if (!force && inFlightQueryKeyRef.current === queryKeyValue) {
        return;
      }
      requestRevisionRef.current += 1;
      const requestRevision = requestRevisionRef.current;
      inFlightQueryKeyRef.current = queryKeyValue;
      const queryKey = workbenchQueryKeys.previewInspection(queryKeyValue);
      setRequest(nextRequest);
      setResponse(undefined);
      setIsBuilding(true);
      setError(null);

      const fetchLatest = async () => {
        await queryClient.cancelQueries({
          queryKey: workbenchQueryKeys.previewInspections(),
        });
        if (requestRevisionRef.current !== requestRevision) {
          return undefined;
        }
        return queryClient.fetchQuery({
          queryKey,
          queryFn: async ({ signal }) =>
            assertPreviewIdentity(
              nextRequest,
              await inspectModel(previewInspectionPayload(nextRequest), { signal }),
            ),
          staleTime: force ? 0 : PREVIEW_INSPECTION_STALE_TIME_MS,
        });
      };
      void fetchLatest()
        .then((nextResponse) => {
          if (
            nextResponse &&
            requestRevisionRef.current === requestRevision
          ) {
            setResponse(nextResponse);
          }
        })
        .catch((caught: unknown) => {
          if (
            requestRevisionRef.current === requestRevision &&
            !isCancelledError(caught) &&
            !isAbortError(caught)
          ) {
            setError(caught instanceof Error ? caught : new Error(String(caught)));
          }
        })
        .finally(() => {
          if (requestRevisionRef.current === requestRevision) {
            inFlightQueryKeyRef.current = null;
            setIsBuilding(false);
          }
        });
    },
    [queryClient],
  );

  const ensure = useCallback(
    (nextRequest: InspectionPreviewRequest) => execute(nextRequest),
    [execute],
  );
  const refresh = useCallback(
    (nextRequest: InspectionPreviewRequest) =>
      execute(nextRequest, { force: true }),
    [execute],
  );
  const clearForConnectionChange = useCallback(() => {
    connectionGenerationRef.current += 1;
    clear();
  }, [clear]);
  const status = useMemo(
    () => ({
      isBuilding,
      isError: Boolean(error),
      error,
    }),
    [error, isBuilding],
  );

  return useMemo(
    () => ({
      response,
      request,
      status,
      clear,
      clearForConnectionChange,
      ensure,
      refresh,
    }),
    [clear, clearForConnectionChange, ensure, refresh, request, response, status],
  );
}
