import { ConfigSummaryPanel } from "@/components/features/viewer/config/config-summary-panel";
import { ErrorPanel } from "@/components/features/viewer/error-panel";
import { ModelExperimentsPanel } from "@/components/features/viewer/model-experiments-panel";
import { TargetPresetPanel } from "@/components/features/viewer/screen/target-preset-panel";
import { useTargetConfig } from "@/components/features/viewer/providers/viewer-providers";
import { errorMessage } from "@/lib/utils";

export function ViewerModelSidebar({ onOpenFullConfig }: { onOpenFullConfig: () => void }) {
  const { modelsQuery, presetsQuery, datasetsQuery, schemaQuery } = useTargetConfig();

  return (
    <>
      {modelsQuery.isError && (
        <ErrorPanel title="Backend unavailable" message={errorMessage(modelsQuery.error)} />
      )}

      <TargetPresetPanel />

      {presetsQuery.isError && (
        <ErrorPanel title="Model import failed" message={errorMessage(presetsQuery.error)} />
      )}
      {datasetsQuery.isError && (
        <ErrorPanel title="Dataset discovery failed" message={errorMessage(datasetsQuery.error)} />
      )}
      {schemaQuery.isError && (
        <ErrorPanel title="Config schema failed" message={errorMessage(schemaQuery.error)} />
      )}

      <ConfigSummaryPanel onOpenFullConfig={onOpenFullConfig} />

      <ModelExperimentsPanel />
    </>
  );
}
