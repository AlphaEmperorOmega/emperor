import { ErrorPanel } from "@/features/workbench/components/error-panel";
import { ConfigSummaryPanel } from "@/features/workbench/components/config/config-summary-panel";
import { TargetPresetPanel } from "@/features/workbench/components/screen/target-preset-panel";
import {
  useTargetQueryStatusState,
} from "@/features/workbench/providers/workbench-providers";
import { type FullConfigDialogControls } from "@/features/workbench/state/use-workbench-workspace-shell";
import { isUnauthorizedApiError } from "@/lib/api";
import { errorMessage } from "@/lib/utils";

function errorPanelTitle(defaultTitle: string, error: unknown) {
  return isUnauthorizedApiError(error) ? "Authentication required" : defaultTitle;
}

export function WorkbenchModelSidebar({
  onOpenFullConfig,
}: {
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  const { modelsQuery, presetsQuery, datasetsQuery, schemaQuery } =
    useTargetQueryStatusState();

  return (
    <>
      {modelsQuery.isError && (
        <ErrorPanel
          title={errorPanelTitle("Backend unavailable", modelsQuery.error)}
          message={errorMessage(modelsQuery.error)}
        />
      )}

      <TargetPresetPanel onOpenFullConfig={onOpenFullConfig} />
      <ConfigSummaryPanel onOpenFullConfig={() => onOpenFullConfig()} />

      {presetsQuery.isError && (
        <ErrorPanel
          title={errorPanelTitle("Model import failed", presetsQuery.error)}
          message={errorMessage(presetsQuery.error)}
        />
      )}
      {datasetsQuery.isError && (
        <ErrorPanel
          title={errorPanelTitle("Dataset discovery failed", datasetsQuery.error)}
          message={errorMessage(datasetsQuery.error)}
        />
      )}
      {schemaQuery.isError && (
        <ErrorPanel
          title={errorPanelTitle("Config schema failed", schemaQuery.error)}
          message={errorMessage(schemaQuery.error)}
        />
      )}
    </>
  );
}
