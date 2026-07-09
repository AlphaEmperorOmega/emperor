import { useCallback, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getWorkbenchApiBaseUrl,
  normalizeWorkbenchApiBaseUrl,
  resetWorkbenchApiBaseUrl,
  setWorkbenchApiBaseUrl,
} from "@/lib/api";
import {
  type PreviewInspectionRequest,
} from "@/features/workbench/state/graph-monitor/use-preview-inspection";
import { workbenchQueryKeys } from "@/lib/query-keys";

type PreviewRefreshController = {
  previewRequest: PreviewInspectionRequest | null;
  clearPreview: () => void;
  requestPreview: (request: PreviewInspectionRequest) => void;
};

export function useWorkbenchApiConnectionSwitch({
  previewRequest,
  clearPreview,
  requestPreview,
}: PreviewRefreshController) {
  const queryClient = useQueryClient();
  const [apiBaseUrl, setApiBaseUrlState] = useState(() => getWorkbenchApiBaseUrl());
  const previewRequestRef = useRef<PreviewInspectionRequest | null>(null);
  previewRequestRef.current = previewRequest;

  const applyApiBaseUrlChange = useCallback(
    (changeApiBaseUrl: () => string) => {
      const previousPreviewRequest = previewRequestRef.current;
      clearPreview();
      const nextApiBaseUrl = changeApiBaseUrl();
      setApiBaseUrlState(nextApiBaseUrl);
      queryClient.removeQueries({ queryKey: workbenchQueryKeys.previewInspections() });
      void queryClient.invalidateQueries({ refetchType: "active" });
      if (previousPreviewRequest) {
        requestPreview(previousPreviewRequest);
      }
      return nextApiBaseUrl;
    },
    [clearPreview, queryClient, requestPreview],
  );

  const setApiBaseUrl = useCallback(
    (url: string) => {
      const normalizedUrl = normalizeWorkbenchApiBaseUrl(url);
      if (!normalizedUrl) {
        return setWorkbenchApiBaseUrl(url);
      }
      return applyApiBaseUrlChange(() => setWorkbenchApiBaseUrl(normalizedUrl));
    },
    [applyApiBaseUrlChange],
  );

  const resetApiBaseUrl = useCallback(
    () => applyApiBaseUrlChange(resetWorkbenchApiBaseUrl),
    [applyApiBaseUrlChange],
  );

  return useMemo(
    () => ({
      apiBaseUrl,
      setApiBaseUrl,
      resetApiBaseUrl,
    }),
    [apiBaseUrl, resetApiBaseUrl, setApiBaseUrl],
  );
}
