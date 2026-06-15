import { TrainingPanel } from "@/features/viewer/components/training-panel";
import {
  useHistoricalRuns,
  useTargetConfig,
  useTraining,
} from "@/features/viewer/providers/viewer-providers";
import {
  useTrainingPanelViewModel,
} from "@/features/viewer/state/training/use-training-panel-view-model";
import {
  type FullConfigDialogMode,
} from "@/features/viewer/state/use-viewer-workspace-shell";

export function ConnectedTrainingPanel({
  onOpenFullConfig,
}: {
  onOpenFullConfig: (mode?: FullConfigDialogMode) => void;
}) {
  const target = useTargetConfig();
  const training = useTraining();
  const history = useHistoricalRuns();
  const canOpenFullConfig = Boolean(
    target.selectedModel && target.selectedPreset && target.schemaQuery.isSuccess,
  );
  const viewModel = useTrainingPanelViewModel({
    models: target.modelsQuery.data?.models ?? [],
    presets: target.presetsQuery.data?.presets ?? [],
    datasetOptions: target.datasetsQuery.data?.datasets ?? [],
    configSections: target.configSections,
    selectedModelType: target.selectedModelType,
    selectedModel: target.selectedModel,
    selectedPreset: target.selectedPreset,
    selectedTrainingPresets: target.selectedTrainingPresets,
    selectedTrainingSnapshotIds: target.selectedTrainingSnapshotIds,
    selectedDatasets: target.selectedDatasets,
    overrides: target.overrides,
    allConfigSnapshots: target.allConfigSnapshots,
    configSnapshotCount: target.allConfigSnapshotCount,
    monitorOptions: target.monitorsQuery.data?.monitors ?? [],
    selectedMonitors: target.selectedMonitors,
    monitorsLoading: target.monitorsQuery.isLoading,
    searchAxes: target.searchSpaceQuery.data?.axes ?? [],
    searchLoading: target.searchSpaceQuery.isLoading,
    trainingSearch: target.trainingSearch,
    trainingEnabled: target.capabilities.trainingEnabled,
    trainingLockedByHistoricalSelection: history.selectedLogRunId !== null,
    historicalTrainingLockExperiment: history.selectedHistoricalExperiment,
    canOpenFullConfig,
    onSelectModelType: target.selectModelType,
    onSelectModel: target.selectModel,
    onSelectPreset: target.selectPreset,
    onSetTrainingPresets: target.setTrainingPresetSelection,
    onSetTrainingSnapshotSelection: target.setTrainingSnapshotSelection,
    onToggleTrainingPreset: target.toggleTrainingPreset,
    onToggleDraftTrainingPreset: target.toggleDraftTrainingPreset,
    onExcludeDraftTrainingPreset: target.excludeDraftTrainingPreset,
    onMakeTrainingPresetPrimary: target.makeTrainingPresetPrimary,
    onSelectAllTrainingPresets: target.selectAllTrainingPresets,
    onSelectPrimaryTrainingPreset: target.selectPrimaryTrainingPreset,
    onSetDatasets: target.setDatasetSelection,
    onToggleDataset: target.toggleDataset,
    onSelectAllDatasets: target.selectAllDatasets,
    onSelectFirstDataset: target.selectFirstDataset,
    onResetOverrides: target.resetOverrides,
    onOpenFullConfig,
    onRemoveConfigSnapshot: target.removeConfigSnapshot,
    onIncludeConfigSnapshot: target.includeConfigSnapshot,
    onExcludeConfigSnapshot: target.excludeConfigSnapshot,
    onEditPresetAsSnapshot: (preset) => {
      if (target.preparePresetSnapshotDraft(preset)) {
        onOpenFullConfig("snapshotDraft");
      }
    },
    onEditConfigSnapshotCopy: (snapshotId) => {
      if (target.loadConfigSnapshot(snapshotId)) {
        onOpenFullConfig("snapshotDraft");
      }
    },
    onToggleMonitor: target.toggleMonitor,
    onTrainingSearchChange: target.setTrainingSearch,
    activeJobId: training.activeJobId,
    onActiveJobIdChange: training.setActiveJobId,
    onJobChange: training.onJobChange,
  });

  return <TrainingPanel viewModel={viewModel} />;
}
