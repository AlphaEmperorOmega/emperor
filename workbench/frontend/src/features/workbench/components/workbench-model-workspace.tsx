import { NodeDetailsPanel } from "@/features/workbench/components/screen/node-details-panel";
import { PreviewPanel } from "@/features/workbench/components/screen/preview-panel";
import { PreviewToolbar } from "@/features/workbench/components/screen/preview-toolbar";
import { WorkbenchModelSidebar } from "@/features/workbench/components/workbench-model-sidebar";
import { WorkbenchThreeRegionLayout } from "@/features/workbench/components/workbench-workspace-layout";
import { type FullConfigDialogControls } from "@/features/workbench/state/use-workbench-workspace-shell";

export function WorkbenchModelWorkspace({
  onOpenFullConfig,
}: {
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  return (
    <WorkbenchThreeRegionLayout
      sidebar={<WorkbenchModelSidebar onOpenFullConfig={onOpenFullConfig} />}
      primary={
        <div className="grid h-full min-h-0 grid-rows-[56px_minmax(0,1fr)] bg-transparent">
          <PreviewToolbar />
          <PreviewPanel />
        </div>
      }
      details={<NodeDetailsPanel />}
    />
  );
}
