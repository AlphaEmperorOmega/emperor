import { useCallback, useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  inspectModel,
  type InspectResponse,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

export type PreviewInspectionRequest = {
  modelType: string;
  model: string;
  preset: string;
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
    dataset: request.dataset,
    overrides: request.overrides,
  };
  return request.logRunId ? { ...payload, logRunId: request.logRunId } : payload;
}

function assertPreviewIdentity<
  Response extends Pick<InspectResponse, "modelType" | "model" | "preset">,
>(request: PreviewInspectionRequest, response: Response) {
  if (
    response.modelType === request.modelType &&
    response.model === request.model &&
    response.preset === request.preset
  ) {
    return response;
  }
  throw new Error(
    `Inspection response identity mismatch: requested ${request.modelType}/${request.model}/${request.preset}, received ${response.modelType}/${response.model}/${response.preset}.`,
  );
}

export function usePreviewInspectionState() {
  const [graph, setGraph] = useState<InspectResponse | undefined>();
  const [previewRequest, setPreviewRequest] =
    useState<PreviewInspectionRequest | null>(null);
  const [previewRequestKey, setPreviewRequestKey] = useState<string | null>(null);
  const requestIdRef = useRef(0);
  const {
    mutate,
    reset,
    isPending,
    isError,
    error,
  } = useMutation({
    mutationFn: async (request: PreviewInspectionRequest) =>
      assertPreviewIdentity(request, await inspectModel(previewInspectionPayload(request))),
  });

  const clearPreview = useCallback(() => {
    requestIdRef.current += 1;
    setGraph(undefined);
    setPreviewRequest(null);
    setPreviewRequestKey(null);
    reset();
  }, [reset]);

  const requestPreview = useCallback(
    (request: PreviewInspectionRequest) => {
      requestIdRef.current += 1;
      const requestId = requestIdRef.current;
      const requestKey = previewInspectionRequestKey(request);
      setGraph(undefined);
      setPreviewRequest(request);
      setPreviewRequestKey(requestKey);
      mutate(request, {
        onSuccess: (response) => {
          if (requestIdRef.current === requestId) {
            setGraph(response);
          }
        },
      });
    },
    [mutate],
  );

  const previewInspection = useMemo(
    () => ({
      isBuilding: isPending,
      isError,
      error,
    }),
    [error, isError, isPending],
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
