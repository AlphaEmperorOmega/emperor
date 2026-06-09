import { ConfigSummaryPanel } from "@/features/viewer/components/config/config-summary-panel";
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import { ModelExperimentsPanel } from "@/features/viewer/components/model-experiments-panel";
import { TargetPresetPanel } from "@/features/viewer/components/screen/target-preset-panel";
import {
  useTargetQueryStatusState,
} from "@/features/viewer/providers/viewer-providers";
import { isUnauthorizedApiError } from "@/lib/api";
import { errorMessage } from "@/lib/utils";

function errorPanelTitle(defaultTitle: string, error: unknown) {
  return isUnauthorizedApiError(error) ? "Authentication required" : defaultTitle;
}

export function ViewerModelSidebar({ onOpenFullConfig }: { onOpenFullConfig: () => void }) {
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

      <TargetPresetPanel />

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

      <ConfigSummaryPanel onOpenFullConfig={onOpenFullConfig} />

      <ModelExperimentsPanel />
    </>
  );
}
