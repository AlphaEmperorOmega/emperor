import { type ReactNode } from "react";
import {
  useWorkbenchState,
  type GraphViewContextValue,
  type GraphMonitorContextValue,
  type HistoricalRunsContextValue,
} from "@/features/workbench/state/use-workbench-state";
import {
  WorkbenchConnectionProvider,
  isWorkbenchProtectedAccessReady,
  useRegisterWorkbenchConnectionReset,
  useWorkbenchCapabilities,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  TrainingLifecycleProvider,
  useActiveTrainingJob,
} from "@/features/workbench/providers/training-provider";
import type { useModelPackageInspectionState } from "@/features/workbench/state/target/use-model-package-inspection-state";
import {
  type ConfigSnapshotEditorSessionState,
  useConfigSnapshotEditorSessionState,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-editor-session";
import { type WorkbenchWorkspace } from "@/types/workbench";

const noop = () => undefined;

type TargetContexts = ReturnType<
  typeof useModelPackageInspectionState
>["contexts"];
type ModelPackageCatalogContextValue = TargetContexts["catalog"];
type ModelPackageInspectionContextValue = TargetContexts["model"];
type ConfigSnapshotRecordsContextValue = TargetContexts["snapshots"];

const [ModelPackageCatalogProvider, useModelPackageCatalog] =
  createWorkbenchContext<ModelPackageCatalogContextValue>(
    "ModelPackageCatalogContext",
  );
const [ModelPackageInspectionProvider, useModelPackageInspection] =
  createWorkbenchContext<ModelPackageInspectionContextValue>(
    "ModelPackageInspectionContext",
  );
const [ConfigSnapshotRecordsProvider, useConfigSnapshotRecords] =
  createWorkbenchContext<ConfigSnapshotRecordsContextValue>(
    "ConfigSnapshotRecordsContext",
  );
const [ConfigSnapshotEditorProvider, useConfigSnapshotEditor] =
  createWorkbenchContext<ConfigSnapshotEditorSessionState>(
    "ConfigSnapshotEditorContext",
  );
const [GraphViewProvider, useGraphView] =
  createWorkbenchContext<GraphViewContextValue>("GraphViewContext");
const [HistoricalRunsProvider, useHistoricalRuns] =
  createWorkbenchContext<HistoricalRunsContextValue>("HistoricalRunsContext");
const [GraphMonitorProvider, useGraphMonitor] =
  createWorkbenchContext<GraphMonitorContextValue>("GraphMonitorContext");
export {
  useModelPackageCatalog,
  useModelPackageInspection,
  useConfigSnapshotRecords,
  useConfigSnapshotEditor,
  useGraphView,
  useHistoricalRuns,
  useGraphMonitor,
  useWorkbenchCapabilities,
  useWorkbenchConnection,
};

export type WorkbenchProvidersProps = {
  /** Wired to the logs workspace so a new job's folder appears in its run list. */
  onJobStarted?: (logFolder: string) => void;
  activeWorkspace?: WorkbenchWorkspace;
  clearShellForConnectionChange?: () => void;
  children: ReactNode;
};

function WorkbenchCompositionProviders({
  activeWorkspace,
  clearShellForConnectionChange,
  children,
}: Omit<WorkbenchProvidersProps, "onJobStarted">) {
  const activeJob = useActiveTrainingJob();
  const workbenchConnection = useWorkbenchConnection();
  const protectedReadsEnabled = isWorkbenchProtectedAccessReady(
    workbenchConnection,
  );
  const configSnapshotEditor = useConfigSnapshotEditorSessionState();
  const {
    targetContexts,
    graph,
    history,
    graphMonitor,
    clearForConnectionChange,
  } = useWorkbenchState({
    activeWorkspace,
    activeTrainingJob: activeJob.activeTrainingJob,
    protectedReadsEnabled,
  });
  useRegisterWorkbenchConnectionReset(clearForConnectionChange);
  useRegisterWorkbenchConnectionReset(
    configSnapshotEditor.actions.close,
  );
  useRegisterWorkbenchConnectionReset(
    clearShellForConnectionChange ?? noop,
  );

  return (
    <ConfigSnapshotEditorProvider value={configSnapshotEditor}>
      <ModelPackageCatalogProvider value={targetContexts.catalog}>
        <ModelPackageInspectionProvider value={targetContexts.model}>
          <ConfigSnapshotRecordsProvider value={targetContexts.snapshots}>
            <GraphViewProvider value={graph}>
              <HistoricalRunsProvider value={history}>
                <GraphMonitorProvider value={graphMonitor}>
                  {children}
                </GraphMonitorProvider>
              </HistoricalRunsProvider>
            </GraphViewProvider>
          </ConfigSnapshotRecordsProvider>
        </ModelPackageInspectionProvider>
      </ModelPackageCatalogProvider>
    </ConfigSnapshotEditorProvider>
  );
}

/**
 * Runs the workbench orchestration engine once and distributes focused domain
 * projections through nested contexts, so panels read exactly the slice they need
 * instead of receiving it drilled down through props.
 */
export function WorkbenchProviders({
  activeWorkspace,
  onJobStarted,
  clearShellForConnectionChange,
  children,
}: WorkbenchProvidersProps) {
  return (
    <WorkbenchConnectionProvider>
      <TrainingLifecycleProvider onJobStarted={onJobStarted}>
        <WorkbenchCompositionProviders
          activeWorkspace={activeWorkspace}
          clearShellForConnectionChange={clearShellForConnectionChange}
        >
          {children}
        </WorkbenchCompositionProviders>
      </TrainingLifecycleProvider>
    </WorkbenchConnectionProvider>
  );
}
