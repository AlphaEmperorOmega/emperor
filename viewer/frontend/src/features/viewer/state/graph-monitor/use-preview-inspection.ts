import { useCallback, useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  inspectModel,
  inspectOperationGraph,
  type InspectResponse,
  type OperationGraphResponse,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

export type PreviewInspectionRequest = {
  modelType: string;
  model: string;
  preset: string;
  dataset?: string;
  overrides: OverrideValues;
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
  });
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
  const [operationGraph, setOperationGraph] =
    useState<OperationGraphResponse | undefined>();
  const [previewRequest, setPreviewRequest] =
    useState<PreviewInspectionRequest | null>(null);
  const [previewRequestKey, setPreviewRequestKey] = useState<string | null>(null);
  const [operationGraphRequestKey, setOperationGraphRequestKey] =
    useState<string | null>(null);
  const [operationGraphInFlightRequestKey, setOperationGraphInFlightRequestKey] =
    useState<string | null>(null);
  const [operationGraphFailedRequestKey, setOperationGraphFailedRequestKey] =
    useState<string | null>(null);
  const [operationGraphError, setOperationGraphError] = useState<unknown>(null);
  const previewRequestRef = useRef<PreviewInspectionRequest | null>(null);
  const previewRequestKeyRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);
  const operationRequestIdRef = useRef(0);
  const {
    mutate,
    reset,
    isPending,
    isError,
    error,
  } = useMutation({
    mutationFn: async (request: PreviewInspectionRequest) =>
      assertPreviewIdentity(request, await inspectModel(request)),
  });
  const {
    mutate: mutateOperationGraph,
    reset: resetOperationGraphMutation,
  } = useMutation({
    mutationFn: async (request: PreviewInspectionRequest) =>
      assertPreviewIdentity(request, await inspectOperationGraph(request)),
  });

  const clearPreview = useCallback(() => {
    requestIdRef.current += 1;
    operationRequestIdRef.current += 1;
    previewRequestRef.current = null;
    previewRequestKeyRef.current = null;
    setGraph(undefined);
    setOperationGraph(undefined);
    setPreviewRequest(null);
    setPreviewRequestKey(null);
    setOperationGraphRequestKey(null);
    setOperationGraphInFlightRequestKey(null);
    setOperationGraphFailedRequestKey(null);
    setOperationGraphError(null);
    reset();
    resetOperationGraphMutation();
  }, [reset, resetOperationGraphMutation]);

  const requestPreview = useCallback(
    (request: PreviewInspectionRequest) => {
      requestIdRef.current += 1;
      operationRequestIdRef.current += 1;
      const requestId = requestIdRef.current;
      const requestKey = previewInspectionRequestKey(request);
      previewRequestRef.current = request;
      previewRequestKeyRef.current = requestKey;
      setGraph(undefined);
      setOperationGraph(undefined);
      setPreviewRequest(request);
      setPreviewRequestKey(requestKey);
      setOperationGraphRequestKey(null);
      setOperationGraphInFlightRequestKey(null);
      setOperationGraphFailedRequestKey(null);
      setOperationGraphError(null);
      resetOperationGraphMutation();
      mutate(request, {
        onSuccess: (response) => {
          if (requestIdRef.current === requestId) {
            setGraph(response);
          }
        },
      });
    },
    [mutate, resetOperationGraphMutation],
  );

  const requestOperationGraph = useCallback(
    (request: PreviewInspectionRequest | null = previewRequestRef.current) => {
      if (!request) {
        return;
      }
      const requestKey = previewInspectionRequestKey(request);
      if (previewRequestKeyRef.current !== requestKey) {
        return;
      }
      operationRequestIdRef.current += 1;
      const requestId = operationRequestIdRef.current;
      setOperationGraph(undefined);
      setOperationGraphRequestKey(null);
      setOperationGraphInFlightRequestKey(requestKey);
      setOperationGraphFailedRequestKey(null);
      setOperationGraphError(null);
      resetOperationGraphMutation();
      mutateOperationGraph(request, {
        onSuccess: (response) => {
          if (
            operationRequestIdRef.current === requestId &&
            previewRequestKeyRef.current === requestKey
          ) {
            setOperationGraph(response);
            setOperationGraphRequestKey(requestKey);
            setOperationGraphInFlightRequestKey(null);
            setOperationGraphFailedRequestKey(null);
            setOperationGraphError(null);
          }
        },
        onError: (requestError) => {
          if (
            operationRequestIdRef.current === requestId &&
            previewRequestKeyRef.current === requestKey
          ) {
            setOperationGraphInFlightRequestKey(null);
            setOperationGraphFailedRequestKey(requestKey);
            setOperationGraphError(requestError);
          }
        },
      });
    },
    [mutateOperationGraph, resetOperationGraphMutation],
  );

  const resetOperationGraphFailure = useCallback(
    (requestKey: string | null = operationGraphFailedRequestKey) => {
      if (requestKey && operationGraphFailedRequestKey !== requestKey) {
        return;
      }
      setOperationGraphFailedRequestKey(null);
      setOperationGraphError(null);
      resetOperationGraphMutation();
    },
    [operationGraphFailedRequestKey, resetOperationGraphMutation],
  );

  const previewInspection = useMemo(
    () => ({
      isBuilding: isPending,
      isError,
      error,
    }),
    [error, isError, isPending],
  );
  const operationInspection = useMemo(
    () => ({
      isBuilding: operationGraphInFlightRequestKey !== null,
      isError: operationGraphFailedRequestKey !== null,
      error: operationGraphError,
    }),
    [
      operationGraphError,
      operationGraphFailedRequestKey,
      operationGraphInFlightRequestKey,
    ],
  );

  return {
    graph,
    operationGraph,
    previewRequest,
    previewRequestKey,
    operationGraphRequestKey,
    operationGraphInFlightRequestKey,
    operationGraphFailedRequestKey,
    clearPreview,
    requestPreview,
    requestOperationGraph,
    resetOperationGraphFailure,
    previewInspection,
    operationInspection,
  };
}
