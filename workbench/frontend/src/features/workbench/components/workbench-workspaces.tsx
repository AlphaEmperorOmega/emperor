import dynamic from "next/dynamic";
import { Activity, type ReactNode } from "react";
import {
  WorkbenchThreeRegionLayout,
  WorkbenchWideWorkspaceRegion,
  WorkbenchWorkspaceLoadingStatus,
} from "@/features/workbench/components/workbench-workspace-layout";
import {
  type DeferredWorkbenchWorkspace,
  type FullConfigDialogControls,
} from "@/features/workbench/state/use-workbench-workspace-shell";
import { type WorkbenchWorkspace } from "@/types/workbench";

function TrainingWorkspaceLoadingFallback() {
  return <WorkbenchWorkspaceLoadingStatus label="Loading training workspace…" />;
}

function LogsWorkspaceProviderLoadingFallback() {
  return (
    <WorkbenchWideWorkspaceRegion>
      <WorkbenchWorkspaceLoadingStatus label="Loading logs workspace…" />
    </WorkbenchWideWorkspaceRegion>
  );
}

function ModelWorkspaceLoadingFallback() {
  return (
    <WorkbenchWideWorkspaceRegion>
      <WorkbenchWorkspaceLoadingStatus label="Loading model workspace…" />
    </WorkbenchWideWorkspaceRegion>
  );
}

const WorkbenchModelWorkspace = dynamic(
  () =>
    import("@/features/workbench/components/workbench-model-workspace").then(
      (module) => module.WorkbenchModelWorkspace,
    ),
  { ssr: false, loading: ModelWorkspaceLoadingFallback },
);

const DeferredTrainingExecutionProvider = dynamic(
  () =>
    import(
      "@/features/workbench/providers/training-execution-provider"
    ).then((module) => module.TrainingExecutionProvider),
  { ssr: false },
);

const DeferredLogsWorkspaceProvider = dynamic(
  () =>
    import("@/features/workbench/providers/logs-workspace-provider").then(
      (module) => module.LogsWorkspaceProvider,
    ),
  {
    ssr: false,
    loading: LogsWorkspaceProviderLoadingFallback,
  },
);

const TrainingPanel = dynamic(
  () =>
    import("@/features/workbench/components/training-panel").then(
      (module) => module.TrainingPanel,
    ),
  {
    ssr: false,
    loading: TrainingWorkspaceLoadingFallback,
  },
);
const FullConfigDialog = dynamic(
  () =>
    import("@/features/workbench/components/config/full-config-dialog").then(
      (module) => module.FullConfigDialog,
    ),
  { ssr: false },
);
const ConnectedLogsSidebarPanel = dynamic(
  () =>
    import("@/features/workbench/components/logs/logs-workspace").then(
      (module) => module.ConnectedLogsSidebarPanel,
    ),
  { ssr: false },
);
const ConnectedLogsGraphPreviewPanel = dynamic(
  () =>
    import("@/features/workbench/components/logs/logs-workspace").then(
      (module) => module.ConnectedLogsGraphPreviewPanel,
    ),
  { ssr: false },
);
const ConnectedLogRunDetailsPanel = dynamic(
  () =>
    import("@/features/workbench/components/logs/log-run-details-panel").then(
      (module) => module.ConnectedLogRunDetailsPanel,
    ),
  { ssr: false },
);

export function WorkbenchWorkspaceActivities({
  activeWorkspace,
  deferredWorkspaceOrder,
  model,
  logs,
  training,
  trainingBoundary,
}: {
  activeWorkspace: WorkbenchWorkspace;
  deferredWorkspaceOrder: readonly DeferredWorkbenchWorkspace[];
  model: ReactNode;
  logs: ReactNode;
  training: ReactNode;
  trainingBoundary?: (activity: ReactNode) => ReactNode;
}) {
  const logsActivity = deferredWorkspaceOrder.includes("logs") ? (
    <Activity
      name="Workbench logs workspace"
      mode={activeWorkspace === "logs" ? "visible" : "hidden"}
    >
      {logs}
    </Activity>
  ) : null;
  const trainingActivity = deferredWorkspaceOrder.includes("training") ? (
    <Activity
      name="Workbench training workspace"
      mode={activeWorkspace === "training" ? "visible" : "hidden"}
    >
      {training}
    </Activity>
  ) : null;

  return (
    <>
      <Activity
        name="Workbench Model workspace"
        mode={activeWorkspace === "model" ? "visible" : "hidden"}
      >
        {model}
      </Activity>
      {logsActivity}
      {trainingBoundary
        ? trainingBoundary(trainingActivity)
        : trainingActivity}
    </>
  );
}

export function WorkbenchWorkspaceRegions({
  activeWorkspace,
  deferredWorkspaceOrder =
    activeWorkspace === "model" ? [] : [activeWorkspace],
  fullConfigDialog,
  onOpenFullConfig,
  startedLogFolders = [],
  trainingRuntimeActivated = false,
}: {
  activeWorkspace: WorkbenchWorkspace;
  deferredWorkspaceOrder?: readonly DeferredWorkbenchWorkspace[];
  fullConfigDialog?: FullConfigDialogControls;
  onOpenFullConfig: FullConfigDialogControls["open"];
  startedLogFolders?: readonly string[];
  trainingRuntimeActivated?: boolean;
}) {
  const model = (
    <WorkbenchModelWorkspace onOpenFullConfig={onOpenFullConfig} />
  );
  const logs = (
    <DeferredLogsWorkspaceProvider
      enabled={activeWorkspace === "logs"}
      startedExperiments={startedLogFolders}
    >
      <WorkbenchThreeRegionLayout
        sidebar={<ConnectedLogsSidebarPanel />}
        primary={<ConnectedLogsGraphPreviewPanel />}
        details={<ConnectedLogRunDetailsPanel />}
      />
    </DeferredLogsWorkspaceProvider>
  );
  const training = (
    <WorkbenchWideWorkspaceRegion>
      <TrainingPanel />
    </WorkbenchWideWorkspaceRegion>
  );
  const trainingBoundary = trainingRuntimeActivated
    ? (activity: ReactNode) => (
        <DeferredTrainingExecutionProvider
          activeWorkspace={activeWorkspace}
          onOpenFullConfig={onOpenFullConfig}
        >
          {activity}
          {fullConfigDialog?.isOpen && (
            <FullConfigDialog
              mode={fullConfigDialog.mode}
              scope={fullConfigDialog.scope}
              onClose={fullConfigDialog.close}
            />
          )}
        </DeferredTrainingExecutionProvider>
      )
    : undefined;

  return (
    <WorkbenchWorkspaceActivities
      activeWorkspace={activeWorkspace}
      deferredWorkspaceOrder={deferredWorkspaceOrder}
      model={model}
      logs={logs}
      training={training}
      trainingBoundary={trainingBoundary}
    />
  );
}
