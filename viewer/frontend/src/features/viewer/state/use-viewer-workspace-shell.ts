import { useCallback, useState } from "react";
import { type ViewerWorkspace } from "@/types/viewer";

export type ViewerDialogControls = {
  isOpen: boolean;
  open: () => void;
  close: () => void;
};

export type FullConfigDialogMode = "default" | "snapshotDraft" | "snapshotEdit";

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
  apiConnectionDialog: ViewerDialogControls;
};

export function useViewerWorkspaceShell() {
  const [activeWorkspace, setActiveWorkspace] = useState<ViewerWorkspace>("model");
  const [isFullConfigOpen, setIsFullConfigOpen] = useState(false);
  const [fullConfigMode, setFullConfigMode] =
    useState<FullConfigDialogMode>("default");
  const [isFeatureListOpen, setIsFeatureListOpen] = useState(false);
  const [isApiConnectionOpen, setIsApiConnectionOpen] = useState(false);

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
  const openApiConnection = useCallback(() => setIsApiConnectionOpen(true), []);
  const closeApiConnection = useCallback(() => setIsApiConnectionOpen(false), []);

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
    apiConnectionDialog: {
      isOpen: isApiConnectionOpen,
      open: openApiConnection,
      close: closeApiConnection,
    },
  };

  return {
    screen,
  };
}
