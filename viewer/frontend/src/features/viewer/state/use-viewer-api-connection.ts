import { useCallback, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getViewerApiBaseUrl,
  normalizeViewerApiBaseUrl,
  resetViewerApiBaseUrl,
  setViewerApiBaseUrl,
} from "@/lib/api";
import {
  type PreviewInspectionRequest,
} from "@/features/viewer/state/graph-monitor/use-preview-inspection";
import { viewerQueryKeys } from "@/lib/query-keys";

type PreviewRefreshController = {
  previewRequest: PreviewInspectionRequest | null;
  clearPreview: () => void;
  requestPreview: (request: PreviewInspectionRequest) => void;
};

export function useViewerApiConnectionSwitch({
  previewRequest,
  clearPreview,
  requestPreview,
}: PreviewRefreshController) {
  const queryClient = useQueryClient();
  const [apiBaseUrl, setApiBaseUrlState] = useState(() => getViewerApiBaseUrl());
  const previewRequestRef = useRef<PreviewInspectionRequest | null>(null);
  previewRequestRef.current = previewRequest;

  const applyApiBaseUrlChange = useCallback(
    (changeApiBaseUrl: () => string) => {
      const previousPreviewRequest = previewRequestRef.current;
      clearPreview();
      const nextApiBaseUrl = changeApiBaseUrl();
      setApiBaseUrlState(nextApiBaseUrl);
      queryClient.removeQueries({ queryKey: viewerQueryKeys.previewInspections() });
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
      const normalizedUrl = normalizeViewerApiBaseUrl(url);
      if (!normalizedUrl) {
        return setViewerApiBaseUrl(url);
      }
      return applyApiBaseUrlChange(() => setViewerApiBaseUrl(normalizedUrl));
    },
    [applyApiBaseUrlChange],
  );

  const resetApiBaseUrl = useCallback(
    () => applyApiBaseUrlChange(resetViewerApiBaseUrl),
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
