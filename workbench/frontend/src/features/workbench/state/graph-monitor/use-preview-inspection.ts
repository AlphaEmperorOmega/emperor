import { useCallback, useMemo, useRef, useState } from "react";
import { isCancelledError, useQueryClient } from "@tanstack/react-query";
import {
  inspectModel,
  type InspectResponse,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";
import { workbenchQueryKeys } from "@/lib/query-keys";

export type PreviewInspectionRequest = {
  modelType: string;
  model: string;
  preset: string;
  experimentTask?: string;
  dataset?: string;
  overrides: OverrideValues;
  targetMode?: "preset" | "snapshot" | "experiment";
  targetId?: string;
  logRunId?: string;
};

function previewInspectionRequestKey(request: PreviewInspectionRequest) {
  return JSON.stringify({
    modelType: request.modelType,
    model: request.model,
    preset: request.preset,
    experimentTask: request.experimentTask ?? null,
    dataset: request.dataset ?? null,
    overrides: Object.fromEntries(
      Object.entries(request.overrides).sort(([left], [right]) =>
        left.localeCompare(right),
      ),
    ),
    targetMode: request.targetMode ?? "preset",
    targetId: request.targetId ?? request.preset,
    logRunId: request.logRunId ?? null,
  });
}

function previewInspectionPayload(request: PreviewInspectionRequest) {
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

export function presetIdentityMatches(left: string, right: string) {
  return left === right || presetIdentityKey(left) === presetIdentityKey(right);
}

function assertPreviewIdentity<
  Response extends Pick<InspectResponse, "modelType" | "model" | "preset">,
>(request: PreviewInspectionRequest, response: Response) {
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

export function usePreviewInspectionState() {
  const queryClient = useQueryClient();
  const [graph, setGraph] = useState<InspectResponse | undefined>();
  const [previewRequest, setPreviewRequest] =
    useState<PreviewInspectionRequest | null>(null);
  const [previewRequestKey, setPreviewRequestKey] = useState<string | null>(null);
  const [isBuilding, setIsBuilding] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const requestIdRef = useRef(0);
  const inFlightRequestKeyRef = useRef<string | null>(null);

  const clearPreview = useCallback(() => {
    requestIdRef.current += 1;
    void queryClient.cancelQueries({
      queryKey: workbenchQueryKeys.previewInspections(),
    });
    setGraph(undefined);
    setPreviewRequest(null);
    setPreviewRequestKey(null);
    inFlightRequestKeyRef.current = null;
    setIsBuilding(false);
    setError(null);
  }, [queryClient]);

  const requestPreview = useCallback(
    (request: PreviewInspectionRequest) => {
      const requestKey = previewInspectionRequestKey(request);
      if (inFlightRequestKeyRef.current === requestKey) {
        return;
      }
      requestIdRef.current += 1;
      const requestId = requestIdRef.current;
      inFlightRequestKeyRef.current = requestKey;
      const queryKey = workbenchQueryKeys.previewInspection(requestKey);
      setPreviewRequest(request);
      setPreviewRequestKey(requestKey);
      setGraph(undefined);
      setIsBuilding(true);
      setError(null);
      const fetchLatestPreview = async () => {
        await queryClient.cancelQueries({
          queryKey: workbenchQueryKeys.previewInspections(),
        });
        if (requestIdRef.current !== requestId) {
          return undefined;
        }
        return queryClient.fetchQuery({
          queryKey,
          queryFn: async ({ signal }) =>
            assertPreviewIdentity(
              request,
              await inspectModel(previewInspectionPayload(request), { signal }),
            ),
          staleTime: PREVIEW_INSPECTION_STALE_TIME_MS,
        });
      };
      void fetchLatestPreview()
        .then((response) => {
          if (response && requestIdRef.current === requestId) {
            setGraph(response);
          }
        })
        .catch((caught: unknown) => {
          if (
            requestIdRef.current === requestId &&
            !isCancelledError(caught) &&
            !isAbortError(caught)
          ) {
            setError(caught instanceof Error ? caught : new Error(String(caught)));
          }
        })
        .finally(() => {
          if (requestIdRef.current === requestId) {
            inFlightRequestKeyRef.current = null;
            setIsBuilding(false);
          }
        });
    },
    [queryClient],
  );

  const previewInspection = useMemo(
    () => ({
      isBuilding,
      isError: Boolean(error),
      error,
    }),
    [error, isBuilding],
  );

  return {
    graph,
    previewRequest,
    previewRequestKey,
    clearPreview,
    requestPreview,
    previewInspection,
  };
}
