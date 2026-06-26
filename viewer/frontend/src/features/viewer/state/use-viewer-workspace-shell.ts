import { useCallback, useState } from "react";
import { type ViewerWorkspace } from "@/types/viewer";

export type ViewerDialogControls = {
  isOpen: boolean;
  open: () => void;
  close: () => void;
};

export type FullConfigDialogMode = "default" | "snapshotDraft" | "snapshotEdit";
export type FullConfigDialogScope = "model" | "training";

export type FullConfigDialogControls = {
  isOpen: boolean;
  mode: FullConfigDialogMode;
  scope: FullConfigDialogScope;
  open: (mode?: FullConfigDialogMode, scope?: FullConfigDialogScope) => void;
  close: () => void;
};

export type ViewerScreenShell = {
  activeWorkspace: ViewerWorkspace;
  onChangeWorkspace: (workspace: ViewerWorkspace) => void;
  fullConfigDialog: FullConfigDialogControls;
  featureListDialog: ViewerDialogControls;
  apiConnectionDialog: ViewerDialogControls;
  importLogsDialog: ViewerDialogControls;
};

export function useViewerWorkspaceShell() {
  const [activeWorkspace, setActiveWorkspace] = useState<ViewerWorkspace>("model");
  const [isFullConfigOpen, setIsFullConfigOpen] = useState(false);
  const [fullConfigMode, setFullConfigMode] =
    useState<FullConfigDialogMode>("default");
  const [fullConfigScope, setFullConfigScope] =
    useState<FullConfigDialogScope>("model");
  const [isFeatureListOpen, setIsFeatureListOpen] = useState(false);
  const [isApiConnectionOpen, setIsApiConnectionOpen] = useState(false);
  const [isImportLogsOpen, setIsImportLogsOpen] = useState(false);

  const changeWorkspace = useCallback((workspace: ViewerWorkspace) => {
    setActiveWorkspace(workspace);
    if (workspace !== "model") {
      setIsFullConfigOpen(false);
    }
  }, []);
  const openFullConfig = useCallback(
    (
      mode: FullConfigDialogMode = "default",
      scope: FullConfigDialogScope = "model",
    ) => {
      setFullConfigMode(mode);
      setFullConfigScope(scope);
      setIsFullConfigOpen(true);
    },
    [],
  );
  const closeFullConfig = useCallback(() => setIsFullConfigOpen(false), []);
  const openFeatureList = useCallback(() => setIsFeatureListOpen(true), []);
  const closeFeatureList = useCallback(() => setIsFeatureListOpen(false), []);
  const openApiConnection = useCallback(() => setIsApiConnectionOpen(true), []);
  const closeApiConnection = useCallback(() => setIsApiConnectionOpen(false), []);
  const openImportLogs = useCallback(() => setIsImportLogsOpen(true), []);
  const closeImportLogs = useCallback(() => setIsImportLogsOpen(false), []);

  const screen: ViewerScreenShell = {
    activeWorkspace,
    onChangeWorkspace: changeWorkspace,
    fullConfigDialog: {
      isOpen: isFullConfigOpen,
      mode: fullConfigMode,
      scope: fullConfigScope,
      open: openFullConfig,
      close: closeFullConfig,
    },
    featureListDialog: {
      isOpen: isFeatureListOpen,
      open: openFeatureList,
      close: closeFeatureList,
    },
    apiConnectionDialog: {
      isOpen: isApiConnectionOpen,
      open: openApiConnection,
      close: closeApiConnection,
    },
    importLogsDialog: {
      isOpen: isImportLogsOpen,
      open: openImportLogs,
      close: closeImportLogs,
    },
  };

  return {
    screen,
  };
}
