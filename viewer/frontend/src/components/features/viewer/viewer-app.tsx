"use client";
// Client boundary: owns Viewer workspace/dialog state and React Query providers.

import { useState } from "react";
import { ViewerProviders } from "@/components/features/viewer/providers/viewer-providers";
import { ViewerScreen } from "@/components/features/viewer/viewer-screen";
import { useLogsWorkspaceState } from "@/components/features/viewer/state/use-logs-workspace-state";
import {
  LOCAL_DEFAULT_CAPABILITIES,
  useCapabilitiesQuery,
} from "@/components/features/viewer/state/use-viewer-queries";
import { type ViewerWorkspace } from "@/types/viewer";

export function ViewerApp() {
  const [activeWorkspace, setActiveWorkspace] = useState<ViewerWorkspace>("model");
  const [isFullConfigOpen, setIsFullConfigOpen] = useState(false);
  const [isFeatureListOpen, setIsFeatureListOpen] = useState(false);
  const capabilitiesQuery = useCapabilitiesQuery();
  const capabilities = capabilitiesQuery.data ?? LOCAL_DEFAULT_CAPABILITIES;
  const logsState = useLogsWorkspaceState({
    enabled: activeWorkspace === "logs",
    logDeletionEnabled: capabilities.logDeletionEnabled,
  });

  const changeWorkspace = (workspace: ViewerWorkspace) => {
    setActiveWorkspace(workspace);
    if (workspace !== "model") {
      setIsFullConfigOpen(false);
    }
  };

  return (
    <ViewerProviders onJobStarted={logsState.includeStartedExperiment}>
      <ViewerScreen
        activeWorkspace={activeWorkspace}
        onChangeWorkspace={changeWorkspace}
        isFullConfigOpen={isFullConfigOpen}
        onOpenFullConfig={() => setIsFullConfigOpen(true)}
        onCloseFullConfig={() => setIsFullConfigOpen(false)}
        isFeatureListOpen={isFeatureListOpen}
        onOpenFeatureList={() => setIsFeatureListOpen(true)}
        onCloseFeatureList={() => setIsFeatureListOpen(false)}
        logsState={logsState}
      />
    </ViewerProviders>
  );
}
