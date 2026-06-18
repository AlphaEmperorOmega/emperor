import { TrainingPanel } from "@/features/viewer/components/training-panel";
import {
  useActiveTrainingJob,
  useActiveTrainingJobProgressState,
  useHistoricalRuns,
  useTargetConfig,
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
  const activeJob = useActiveTrainingJob();
  const activeJobProgress = useActiveTrainingJobProgressState();
  const history = useHistoricalRuns();
  const viewModel = useTrainingPanelViewModel({
    models: target.models,
    presets: target.presets,
    datasetOptions: target.datasets,
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
    monitorOptions: target.monitors,
    selectedMonitors: target.selectedMonitors,
    monitorsLoading: target.monitorsLoading,
    searchAxes: target.searchAxes,
    searchLoading: target.searchAxesLoading,
    trainingSearch: target.trainingSearch,
    trainingEnabled: target.capabilities.trainingEnabled,
    trainingLockedByHistoricalSelection: history.selectedLogRunId !== null,
    historicalTrainingLockExperiment: history.selectedHistoricalExperiment,
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
    onRemoveConfigSnapshot: target.removeConfigSnapshot,
    onIncludeConfigSnapshot: target.includeConfigSnapshot,
    onExcludeConfigSnapshot: target.excludeConfigSnapshot,
    onCreatePresetSnapshot: (preset) => {
      if (
        target.preparePresetSnapshotDraft(preset, {
          includeTrainingPreset: false,
        })
      ) {
        onOpenFullConfig("snapshotDraft");
      }
    },
    onEditConfigSnapshot: (snapshotId) => {
      if (
        target.prepareSelectedSnapshotEdit(snapshotId, {
          includeTrainingSnapshot: false,
        })
      ) {
        onOpenFullConfig("snapshotEdit");
      }
    },
    onDuplicateConfigSnapshot: (snapshotId) => {
      if (
        target.prepareSelectedSnapshotEdit(snapshotId, {
          includeTrainingSnapshot: false,
        })
      ) {
        onOpenFullConfig("snapshotDraft");
      }
    },
    onSetMonitors: target.setMonitorSelection,
    onSelectAllMonitors: target.selectAllMonitors,
    onClearMonitors: target.clearMonitors,
    onTrainingSearchChange: target.setTrainingSearch,
    activeTrainingJob: activeJob.activeTrainingJob,
    progressError: activeJobProgress.progressError,
    onActiveJobIdChange: activeJob.setActiveJobId,
    onJobChange: activeJob.onJobChange,
  });

  return <TrainingPanel viewModel={viewModel} />;
}
