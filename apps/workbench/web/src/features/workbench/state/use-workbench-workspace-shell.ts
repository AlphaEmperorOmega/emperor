import {
  startTransition,
  useCallback,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";
import {
  createWorkbenchLocationServerSnapshot,
  getWorkbenchLocationSnapshot,
  navigateToWorkbenchWorkspace,
  subscribeWorkbenchLocation,
  workspaceHref as buildWorkspaceHref,
  type DeferredWorkbenchWorkspace,
} from "@/features/workbench/state/workbench-location-store";
import {
  type WorkbenchWorkspace,
} from "@/types/workbench";

export type { DeferredWorkbenchWorkspace } from "@/features/workbench/state/workbench-location-store";

export type WorkbenchDialogControls = {
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

export type WorkbenchScreenShell = {
  activeWorkspace: WorkbenchWorkspace;
  onChangeWorkspace: (workspace: WorkbenchWorkspace) => void;
  workspaceHref: (workspace: WorkbenchWorkspace) => string;
  fullConfigDialog: FullConfigDialogControls;
  featureListDialog: WorkbenchDialogControls;
  apiConnectionDialog: WorkbenchDialogControls;
  importLogsDialog: WorkbenchDialogControls;
};

export function useWorkbenchWorkspaceShell(
  initialWorkspace: WorkbenchWorkspace = "model",
) {
  const serverSnapshot = useMemo(
    () => createWorkbenchLocationServerSnapshot(initialWorkspace),
    [initialWorkspace],
  );
  const initialClientSnapshot = useMemo(
    () => ({
      ...serverSnapshot,
      href:
        typeof window === "undefined"
          ? serverSnapshot.href
          : window.location.href,
    }),
    [serverSnapshot],
  );
  const subscribe = useCallback(
    (listener: () => void) =>
      subscribeWorkbenchLocation(
        () => startTransition(listener),
        initialWorkspace,
      ),
    [initialWorkspace],
  );
  const getSnapshot = useCallback(() => {
    const current = getWorkbenchLocationSnapshot();
    const currentUrl = new URL(current.href);
    if (
      current.revision === 0 &&
      !currentUrl.searchParams.has("workspace") &&
      initialWorkspace !== "model"
    ) {
      return initialClientSnapshot;
    }
    return current;
  }, [initialClientSnapshot, initialWorkspace]);
  const locationSnapshot = useSyncExternalStore(
    subscribe,
    getSnapshot,
    () => serverSnapshot,
  );
  const activeWorkspace = locationSnapshot.workspace;
  const deferredWorkspaceOrder =
    locationSnapshot.deferredWorkspaceOrder as readonly DeferredWorkbenchWorkspace[];
  const [fullConfigDialog, setFullConfigDialog] = useState<{
    revision: number;
    mode: FullConfigDialogMode;
    scope: FullConfigDialogScope;
  } | null>(null);
  const [featureListRevision, setFeatureListRevision] = useState<number | null>(
    null,
  );
  const [apiConnectionRevision, setApiConnectionRevision] = useState<
    number | null
  >(null);
  const [importLogsRevision, setImportLogsRevision] = useState<number | null>(
    null,
  );

  const changeWorkspace = useCallback(
    (workspace: WorkbenchWorkspace) => {
      if (workspace === activeWorkspace) return;
      navigateToWorkbenchWorkspace(workspace);
    },
    [activeWorkspace],
  );
  const workspaceHref = useCallback(
    (workspace: WorkbenchWorkspace) =>
      buildWorkspaceHref(locationSnapshot.href, workspace),
    [locationSnapshot.href],
  );
  const openFullConfig = useCallback(
    (
      mode: FullConfigDialogMode = "default",
      scope: FullConfigDialogScope = "model",
    ) => {
      setFullConfigDialog({
        revision: locationSnapshot.revision,
        mode,
        scope,
      });
    },
    [locationSnapshot.revision],
  );
  const closeFullConfig = useCallback(() => setFullConfigDialog(null), []);
  const openFeatureList = useCallback(
    () => setFeatureListRevision(locationSnapshot.revision),
    [locationSnapshot.revision],
  );
  const closeFeatureList = useCallback(() => setFeatureListRevision(null), []);
  const openApiConnection = useCallback(
    () => setApiConnectionRevision(locationSnapshot.revision),
    [locationSnapshot.revision],
  );
  const closeApiConnection = useCallback(
    () => setApiConnectionRevision(null),
    [],
  );
  const openImportLogs = useCallback(
    () => setImportLogsRevision(locationSnapshot.revision),
    [locationSnapshot.revision],
  );
  const closeImportLogs = useCallback(() => setImportLogsRevision(null), []);
  const fullConfigIsOpen =
    fullConfigDialog?.revision === locationSnapshot.revision;

  const screen: WorkbenchScreenShell = {
    activeWorkspace,
    onChangeWorkspace: changeWorkspace,
    workspaceHref,
    fullConfigDialog: {
      isOpen: fullConfigIsOpen,
      mode: fullConfigDialog?.mode ?? "default",
      scope: fullConfigDialog?.scope ?? "model",
      open: openFullConfig,
      close: closeFullConfig,
    },
    featureListDialog: {
      isOpen: featureListRevision === locationSnapshot.revision,
      open: openFeatureList,
      close: closeFeatureList,
    },
    apiConnectionDialog: {
      isOpen: apiConnectionRevision === locationSnapshot.revision,
      open: openApiConnection,
      close: closeApiConnection,
    },
    importLogsDialog: {
      isOpen: importLogsRevision === locationSnapshot.revision,
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
