import { useCallback, useEffect, useState } from "react";
import {
  parseWorkbenchWorkspace,
  type WorkbenchWorkspace,
} from "@/types/workbench";

export type WorkbenchDialogControls = {
  isOpen: boolean;
  open: () => void;
  close: () => void;
};

export type FullConfigDialogMode = "default" | "snapshotDraft" | "snapshotEdit";
export type FullConfigDialogScope = "model" | "training";
export type DeferredWorkbenchWorkspace = Extract<
  WorkbenchWorkspace,
  "logs" | "training"
>;

export type FullConfigDialogControls = {
  isOpen: boolean;
  mode: FullConfigDialogMode;
  scope: FullConfigDialogScope;
  open: (mode?: FullConfigDialogMode, scope?: FullConfigDialogScope) => void;
  close: () => void;
};

export type WorkbenchScreenShell = {
  activeWorkspace: WorkbenchWorkspace;
  onChangeWorkspace: (workspace: WorkbenchWorkspace) => void;
  fullConfigDialog: FullConfigDialogControls;
  featureListDialog: WorkbenchDialogControls;
  apiConnectionDialog: WorkbenchDialogControls;
  importLogsDialog: WorkbenchDialogControls;
};

export function useWorkbenchWorkspaceShell(
  initialWorkspace: WorkbenchWorkspace = "model",
) {
  const [activeWorkspace, setActiveWorkspace] =
    useState<WorkbenchWorkspace>(initialWorkspace);
  const [deferredWorkspaceOrder, setDeferredWorkspaceOrder] = useState<
    DeferredWorkbenchWorkspace[]
  >(() =>
    initialWorkspace === "logs" || initialWorkspace === "training"
      ? [initialWorkspace]
      : [],
  );
  const [isFullConfigOpen, setIsFullConfigOpen] = useState(false);
  const [fullConfigMode, setFullConfigMode] =
    useState<FullConfigDialogMode>("default");
  const [fullConfigScope, setFullConfigScope] =
    useState<FullConfigDialogScope>("model");
  const [isFeatureListOpen, setIsFeatureListOpen] = useState(false);
  const [isApiConnectionOpen, setIsApiConnectionOpen] = useState(false);
  const [isImportLogsOpen, setIsImportLogsOpen] = useState(false);

  const applyWorkspace = useCallback((workspace: WorkbenchWorkspace) => {
    setActiveWorkspace(workspace);
    if (workspace === "logs" || workspace === "training") {
      setDeferredWorkspaceOrder((current) =>
        current.includes(workspace) ? current : [...current, workspace],
      );
    }
    setIsFullConfigOpen(false);
    setIsFeatureListOpen(false);
    setIsApiConnectionOpen(false);
    setIsImportLogsOpen(false);
  }, []);
  const changeWorkspace = useCallback(
    (workspace: WorkbenchWorkspace) => {
      if (workspace === activeWorkspace) return;
      applyWorkspace(workspace);
      const url = new URL(location.href);
      url.searchParams.set("workspace", workspace);
      history.pushState(history.state, "", url);
    },
    [activeWorkspace, applyWorkspace],
  );
  useEffect(() => {
    const handlePopState = () => {
      const workspace = parseWorkbenchWorkspace(
        new URL(location.href).searchParams.get("workspace"),
      );
      applyWorkspace(workspace);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [applyWorkspace]);
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

  const screen: WorkbenchScreenShell = {
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
    deferredWorkspaceOrder,
    logsWorkspaceActivated: deferredWorkspaceOrder.includes("logs"),
    trainingWorkspaceActivated: deferredWorkspaceOrder.includes("training"),
    screen,
  };
}
