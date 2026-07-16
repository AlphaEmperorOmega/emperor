import { Activity } from "lucide-react";
import { ErrorPanel } from "@/features/workbench/components/error-panel";
import { ConfigSummaryPanel } from "@/features/workbench/components/config/config-summary-panel";
import { TargetPresetPanel } from "@/features/workbench/components/screen/target-preset-panel";
import { WorkbenchSidebarHeader } from "@/features/workbench/components/shared/workbench-sidebar";
import {
  useModelPackageCatalog,
  useModelPackageInspection,
} from "@/features/workbench/providers/workbench-providers";
import { useWorkbenchConnection } from "@/features/workbench/providers/workbench-connection-provider";
import { type FullConfigDialogControls } from "@/features/workbench/state/use-workbench-workspace-shell";
import { errorMessage } from "@/lib/utils";

function errorPanelTitle(
  defaultTitle: string,
  authenticationState: ReturnType<typeof useWorkbenchConnection>["authentication"]["state"],
) {
  return authenticationState === "rejected"
    ? "Authentication required"
    : defaultTitle;
}

export function WorkbenchModelSidebar({
  onOpenFullConfig,
}: {
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  const { modelPackages } = useModelPackageCatalog();
  const { authentication } = useWorkbenchConnection();
  const { status } = useModelPackageInspection();
  const modelsQuery = modelPackages;
  const presetsQuery = status.presets;
  const datasetsQuery = status.datasets;
  const schemaQuery = status.schema;
  const authenticationRequired =
    authentication.state === "unauthenticated" ||
    authentication.state === "rejected";

  return (
    <>
      <WorkbenchSidebarHeader icon={<Activity aria-hidden />} title="Setup" />
      {authenticationRequired && (
        <ErrorPanel
          title="Authentication required"
          message={
            authentication.state === "rejected"
              ? "The session bearer token was rejected. Replace it or log out from API Connection."
              : "Enter the bearer token supplied by the API operator in API Connection."
          }
        />
      )}
      {modelsQuery.isError && !authenticationRequired && (
        <ErrorPanel
          title={errorPanelTitle("Backend unavailable", authentication.state)}
          message={errorMessage(modelsQuery.error)}
        />
      )}

      <TargetPresetPanel onOpenFullConfig={onOpenFullConfig} />
      <ConfigSummaryPanel onOpenFullConfig={() => onOpenFullConfig()} />

      {presetsQuery.isError && (
        <ErrorPanel
          title={errorPanelTitle("Model import failed", authentication.state)}
          message={errorMessage(presetsQuery.error)}
        />
      )}
      {datasetsQuery.isError && (
        <ErrorPanel
          title={errorPanelTitle("Dataset discovery failed", authentication.state)}
          message={errorMessage(datasetsQuery.error)}
        />
      )}
      {schemaQuery.isError && (
        <ErrorPanel
          title={errorPanelTitle("Config schema failed", authentication.state)}
          message={errorMessage(schemaQuery.error)}
        />
      )}
    </>
  );
}
