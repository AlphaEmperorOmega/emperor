import dynamic from "next/dynamic";
import {
  useGraphMonitor,
  useGraphView,
} from "@/features/workbench/providers/workbench-providers";
import {
  type FullConfigDialogControls,
  type WorkbenchDialogControls,
} from "@/features/workbench/state/use-workbench-workspace-shell";
import { type WorkbenchWorkspace } from "@/types/workbench";

const FullConfigDialog = dynamic(
  () =>
    import("@/features/workbench/components/config/full-config-dialog").then(
      (module) => module.FullConfigDialog,
    ),
  { ssr: false },
);
const ApiConnectionDialog = dynamic(
  () =>
    import(
      "@/features/workbench/components/screen/api-connection-dialog"
    ).then((module) => module.ApiConnectionDialog),
  { ssr: false },
);
const ImportLogsDialog = dynamic(
  () =>
    import("@/features/workbench/components/screen/import-logs-dialog").then(
      (module) => module.ImportLogsDialog,
    ),
  { ssr: false },
);
const FeatureListDialog = dynamic(
  () =>
    import("@/features/workbench/components/feature-list-dialog").then(
      (module) => module.FeatureListDialog,
    ),
  { ssr: false },
);
const ConnectedMonitorChartsModal = dynamic(
  () =>
    import(
      "@/features/workbench/components/monitor/connected-monitor-charts-modal"
    ).then((module) => module.ConnectedMonitorChartsModal),
  { ssr: false },
);
const ConnectedNeuronCluster3DPopup = dynamic(
  () =>
    import("@/features/workbench/components/graph/neuron-cluster-3d-popup").then(
      (module) => module.ConnectedNeuronCluster3DPopup,
    ),
  { ssr: false },
);

export function WorkbenchWorkspaceOverlays({
  activeWorkspace,
  fullConfigDialog,
  featureListDialog,
  apiConnectionDialog,
  importLogsDialog,
  fullConfigManagedByTrainingRuntime = false,
}: {
  activeWorkspace: WorkbenchWorkspace;
  fullConfigDialog: FullConfigDialogControls;
  featureListDialog: WorkbenchDialogControls;
  apiConnectionDialog: WorkbenchDialogControls;
  importLogsDialog: WorkbenchDialogControls;
  fullConfigManagedByTrainingRuntime?: boolean;
}) {
  const isModelWorkspace = activeWorkspace === "model";
  const { cluster3dNodeId } = useGraphView();
  const { graphMonitorNode, graphMonitorSource } = useGraphMonitor();

  return (
    <>
      {fullConfigDialog.isOpen && !fullConfigManagedByTrainingRuntime && (
        <FullConfigDialog
          mode={fullConfigDialog.mode}
          scope={fullConfigDialog.scope}
          onClose={fullConfigDialog.close}
        />
      )}
      {featureListDialog.isOpen && (
        <FeatureListDialog onClose={featureListDialog.close} />
      )}
      {apiConnectionDialog.isOpen && (
        <ApiConnectionDialog onClose={apiConnectionDialog.close} />
      )}
      {importLogsDialog.isOpen && (
        <ImportLogsDialog onClose={importLogsDialog.close} />
      )}
      {isModelWorkspace && graphMonitorNode && graphMonitorSource && (
        <ConnectedMonitorChartsModal />
      )}
      {isModelWorkspace && cluster3dNodeId && <ConnectedNeuronCluster3DPopup />}
    </>
  );
}
