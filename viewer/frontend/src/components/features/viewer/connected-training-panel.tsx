import { TrainingPanel } from "@/components/features/viewer/training-panel";
import {
  useTargetConfig,
  useTraining,
} from "@/components/features/viewer/providers/viewer-providers";

export function ConnectedTrainingPanel({
  onOpenFullConfig,
}: {
  onOpenFullConfig: () => void;
}) {
  const target = useTargetConfig();
  const training = useTraining();
  const canOpenFullConfig = Boolean(
    target.selectedModel && target.selectedPreset && target.schemaQuery.isSuccess,
  );

  return (
    <TrainingPanel
      models={target.modelsQuery.data?.models ?? []}
      presets={target.presetsQuery.data?.presets ?? []}
      datasetOptions={target.datasetsQuery.data?.datasets ?? []}
      configSections={target.configSections}
      selectedModel={target.selectedModel}
      selectedPreset={target.selectedPreset}
      selectedTrainingPresets={target.selectedTrainingPresets}
      selectedDatasets={target.selectedDatasets}
      overrides={target.overrides}
      configSnapshots={target.configSnapshots}
      configSnapshotCount={target.configSnapshotCount}
      monitorOptions={target.monitorsQuery.data?.monitors ?? []}
      selectedMonitors={target.selectedMonitors}
      monitorsLoading={target.monitorsQuery.isLoading}
      searchAxes={target.searchSpaceQuery.data?.axes ?? []}
      searchLoading={target.searchSpaceQuery.isLoading}
      trainingSearch={target.trainingSearch}
      trainingEnabled={target.capabilities.trainingEnabled}
      onSelectModel={target.selectModel}
      onSelectPreset={target.selectPreset}
      onSetTrainingPresets={target.setTrainingPresetSelection}
      onToggleTrainingPreset={target.toggleTrainingPreset}
      onMakeTrainingPresetPrimary={target.makeTrainingPresetPrimary}
      onSelectAllTrainingPresets={target.selectAllTrainingPresets}
      onSelectPrimaryTrainingPreset={target.selectPrimaryTrainingPreset}
      onSetDatasets={target.setDatasetSelection}
      onToggleDataset={target.toggleDataset}
      onSelectAllDatasets={target.selectAllDatasets}
      onSelectFirstDataset={target.selectFirstDataset}
      onResetOverrides={target.resetOverrides}
      onOpenFullConfig={onOpenFullConfig}
      canOpenFullConfig={canOpenFullConfig}
      onRemoveConfigSnapshot={target.removeConfigSnapshot}
      onToggleMonitor={target.toggleMonitor}
      onTrainingSearchChange={target.setTrainingSearch}
      activeJobId={training.activeJobId}
      onActiveJobIdChange={training.setActiveJobId}
      onJobChange={training.onJobChange}
    />
  );
}
