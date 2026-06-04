import { useCallback, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { inspectModel, type InspectResponse } from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

export type PreviewInspectionRequest = {
  model: string;
  preset: string;
  dataset?: string;
  overrides: OverrideValues;
};

export function usePreviewInspectionState() {
  const [graph, setGraph] = useState<InspectResponse | undefined>();
  const requestIdRef = useRef(0);
  const { mutate, isPending, isError, error } = useMutation({
    mutationFn: inspectModel,
  });

  const requestPreview = useCallback(
    (request: PreviewInspectionRequest) => {
      requestIdRef.current += 1;
      const requestId = requestIdRef.current;
      setGraph(undefined);
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

  return {
    graph,
    requestPreview,
    previewInspection: {
      isBuilding: isPending,
      isError,
      error,
    },
  };
}
