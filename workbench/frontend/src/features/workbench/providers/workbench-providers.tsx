import { type ReactNode, useEffect } from "react";
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
  TrainingConfigurationContextProvider,
  TrainingWorkspaceContextProvider,
  useActiveTrainingJob,
  useTrainingLifecycleImplementation,
} from "@/features/workbench/providers/training-provider";
import type { useModelPackageInspectionState } from "@/features/workbench/state/target/use-model-package-inspection-state";
import { useTrainingWorkspaceState } from "@/features/workbench/state/training/use-training-workspace-state";
import {
  type ConfigSnapshotEditorSessionState,
  useConfigSnapshotEditorSessionState,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-editor-session";
import { type FullConfigDialogControls } from "@/features/workbench/state/use-workbench-workspace-shell";
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

function TrainingWorkspaceController({
  activeWorkspace,
  onOpenFullConfig,
  children,
}: {
  activeWorkspace?: WorkbenchWorkspace;
  onOpenFullConfig?: FullConfigDialogControls["open"];
  children: ReactNode;
}) {
  const catalog = useModelPackageCatalog();
  const { capabilities } = useWorkbenchCapabilities();
  const workbenchConnection = useWorkbenchConnection();
  const modelTarget = useModelPackageInspection();
  const snapshotEditor = useConfigSnapshotEditor();
  const lifecycle = useTrainingLifecycleImplementation();
  const protectedReadsEnabled =
    activeWorkspace === "training" &&
    isWorkbenchProtectedAccessReady(workbenchConnection);
  const training = useTrainingWorkspaceState({
    activeWorkspace: activeWorkspace ?? "model",
    models: catalog.modelPackages.records,
    seed: {
      modelType: modelTarget.browser.selectedModelType,
      model: modelTarget.browser.selectedModel,
      preset: modelTarget.browser.selectedPreset,
    },
    trainingEnabled: capabilities.trainingEnabled && protectedReadsEnabled,
    protectedReadsEnabled,
    onOpenFullConfig: () => onOpenFullConfig?.("default", "training"),
    onCreatePresetSnapshot: (target) => {
      if (snapshotEditor.actions.beginDraft(target)) {
        onOpenFullConfig?.("snapshotDraft");
      }
    },
    onEditConfigSnapshot: (snapshot) => {
      if (snapshotEditor.actions.beginEdit(snapshot)) {
        onOpenFullConfig?.("snapshotEdit");
      }
    },
    onDuplicateConfigSnapshot: (snapshot) => {
      if (snapshotEditor.actions.beginDuplicate(snapshot)) {
        onOpenFullConfig?.("snapshotDraft");
      }
    },
    activeTrainingJob: lifecycle.activeTrainingJob,
    progressError: lifecycle.progressError,
    onActiveJobIdChange: lifecycle.setActiveJobId,
    onJobChange: lifecycle.onJobChange,
  });
  const registerDraftConnectionClear = lifecycle.registerDraftConnectionClear;
  const clearForConnectionChange = training.clearForConnectionChange;

  useEffect(
    () => registerDraftConnectionClear(clearForConnectionChange),
    [clearForConnectionChange, registerDraftConnectionClear],
  );

  return (
    <TrainingConfigurationContextProvider value={training.configuration}>
      <TrainingWorkspaceContextProvider value={training.workspace}>
        {children}
      </TrainingWorkspaceContextProvider>
    </TrainingConfigurationContextProvider>
  );
}

export type WorkbenchProvidersProps = {
  /** Wired to the logs workspace so a new job's folder appears in its run list. */
  onJobStarted?: (logFolder: string) => void;
  activeWorkspace?: WorkbenchWorkspace;
  onOpenFullConfig?: FullConfigDialogControls["open"];
  clearShellForConnectionChange?: () => void;
  children: ReactNode;
};

function WorkbenchCompositionProviders({
  activeWorkspace,
  onOpenFullConfig,
  clearShellForConnectionChange,
  children,
}: Omit<WorkbenchProvidersProps, "onJobStarted">) {
  const lifecycle = useTrainingLifecycleImplementation();
  const activeJob = useActiveTrainingJob();
  const workbenchConnection = useWorkbenchConnection();
  const protectedReadsEnabled = isWorkbenchProtectedAccessReady(
    workbenchConnection,
  );
  const configSnapshotEditor = useConfigSnapshotEditorSessionState();
  const { targetContexts, graph, history, graphMonitor, clearForConnectionChange } =
    useWorkbenchState({
    activeWorkspace,
    activeTrainingJob: activeJob.activeTrainingJob,
    protectedReadsEnabled,
  });
  useRegisterWorkbenchConnectionReset(clearForConnectionChange);
  useRegisterWorkbenchConnectionReset(lifecycle.clearForConnectionChange);
  useRegisterWorkbenchConnectionReset(configSnapshotEditor.actions.close);
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
                <TrainingWorkspaceController
                  activeWorkspace={activeWorkspace}
                  onOpenFullConfig={onOpenFullConfig}
                >
                  <GraphMonitorProvider value={graphMonitor}>
                    {children}
                  </GraphMonitorProvider>
                </TrainingWorkspaceController>
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
  onOpenFullConfig,
  clearShellForConnectionChange,
  children,
}: WorkbenchProvidersProps) {
  return (
    <WorkbenchConnectionProvider>
      <TrainingLifecycleProvider onJobStarted={onJobStarted}>
        <WorkbenchCompositionProviders
          activeWorkspace={activeWorkspace}
          onOpenFullConfig={onOpenFullConfig}
          clearShellForConnectionChange={clearShellForConnectionChange}
        >
          {children}
        </WorkbenchCompositionProviders>
      </TrainingLifecycleProvider>
    </WorkbenchConnectionProvider>
  );
}
