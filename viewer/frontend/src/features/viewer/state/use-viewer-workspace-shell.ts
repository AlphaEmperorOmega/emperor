import { useCallback, useState } from "react";
import { useLogsWorkspaceState } from "@/features/viewer/state/logs/use-logs-workspace-state";
import {
  LOCAL_DEFAULT_CAPABILITIES,
  useCapabilitiesQuery,
} from "@/features/viewer/state/use-viewer-queries";
import { type ViewerWorkspace } from "@/types/viewer";

export type ViewerDialogControls = {
  isOpen: boolean;
  open: () => void;
  close: () => void;
};

export type FullConfigDialogMode = "default" | "snapshotDraft";

export type FullConfigDialogControls = {
  isOpen: boolean;
  mode: FullConfigDialogMode;
  open: (mode?: FullConfigDialogMode) => void;
  close: () => void;
};

export type ViewerScreenShell = {
  activeWorkspace: ViewerWorkspace;
  onChangeWorkspace: (workspace: ViewerWorkspace) => void;
  fullConfigDialog: FullConfigDialogControls;
  featureListDialog: ViewerDialogControls;
};

export function useViewerWorkspaceShell() {
  const [activeWorkspace, setActiveWorkspace] = useState<ViewerWorkspace>("model");
  const [isFullConfigOpen, setIsFullConfigOpen] = useState(false);
  const [fullConfigMode, setFullConfigMode] =
    useState<FullConfigDialogMode>("default");
  const [isFeatureListOpen, setIsFeatureListOpen] = useState(false);
  const capabilitiesQuery = useCapabilitiesQuery();
  const capabilities = capabilitiesQuery.data ?? LOCAL_DEFAULT_CAPABILITIES;
  const logsWorkspaceState = useLogsWorkspaceState({
    enabled: activeWorkspace === "logs",
    logDeletionEnabled: capabilities.logDeletionEnabled,
  });

  const changeWorkspace = useCallback((workspace: ViewerWorkspace) => {
    setActiveWorkspace(workspace);
    if (workspace !== "model") {
      setIsFullConfigOpen(false);
    }
  }, []);
  const openFullConfig = useCallback((mode: FullConfigDialogMode = "default") => {
    setFullConfigMode(mode);
    setIsFullConfigOpen(true);
  }, []);
  const closeFullConfig = useCallback(() => setIsFullConfigOpen(false), []);
  const openFeatureList = useCallback(() => setIsFeatureListOpen(true), []);
  const closeFeatureList = useCallback(() => setIsFeatureListOpen(false), []);

  const screen: ViewerScreenShell = {
    activeWorkspace,
    onChangeWorkspace: changeWorkspace,
    fullConfigDialog: {
      isOpen: isFullConfigOpen,
      mode: fullConfigMode,
      open: openFullConfig,
      close: closeFullConfig,
    },
    featureListDialog: {
      isOpen: isFeatureListOpen,
      open: openFeatureList,
      close: closeFeatureList,
    },
  };

  return {
    screen,
    logsWorkspaceState,
    onJobStarted: logsWorkspaceState.includeStartedExperiment,
  };
}
