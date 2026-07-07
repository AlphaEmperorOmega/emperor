import { TrainingPanel } from "@/features/viewer/components/training-panel";
import {
  useActiveTrainingJob,
  useActiveTrainingJobProgressState,
  useTargetConfig,
} from "@/features/viewer/providers/viewer-providers";
import {
  useTrainingPanelViewModel,
} from "@/features/viewer/state/training/use-training-panel-view-model";
import {
  type FullConfigDialogControls,
} from "@/features/viewer/state/use-viewer-workspace-shell";

export function ConnectedTrainingWorkspace({
  onOpenFullConfig,
}: {
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  const target = useTargetConfig();
  const activeJob = useActiveTrainingJob();
  const activeJobProgress = useActiveTrainingJobProgressState();
  const viewModel = useTrainingPanelViewModel({
    models: target.models,
    presets: target.trainingPresets,
    datasetOptions: target.trainingDatasets,
    experimentTaskOptions: target.trainingExperimentTaskOptions,
    configSections: target.trainingConfigSections,
    selectedModelType: target.selectedTrainingModelType,
    selectedModel: target.selectedTrainingModel,
    selectedPreset: target.selectedTrainingPrimaryPreset,
    selectedTrainingPresets: target.selectedTrainingPresets,
    selectedExperimentTask: target.selectedTrainingExperimentTask,
    selectedTrainingSnapshotIds: target.selectedTrainingSnapshotIds,
    selectedDatasets: target.selectedTrainingDatasets,
    overrides: target.trainingOverrides,
    snapshotOverrideWarning: target.snapshotOverrideWarning,
    allConfigSnapshots: target.allTrainingConfigSnapshots,
    configSnapshotCount: target.allTrainingConfigSnapshotCount,
    monitorOptions: target.monitors,
    selectedMonitors: target.selectedTrainingMonitors,
    monitorsLoading: target.monitorsLoading,
    searchAxes: target.searchAxes,
    searchLoading: target.searchAxesLoading,
    trainingSearch: target.trainingSearch,
    trainingEnabled: target.capabilities.trainingEnabled,
    canOpenFullConfig: Boolean(
      target.selectedTrainingModel &&
        target.selectedTrainingPrimaryPreset &&
        target.isTrainingSchemaReady,
    ),
    onOpenFullConfig: () => onOpenFullConfig("default", "training"),
    onSelectModelType: target.selectTrainingModelType,
    onSelectModel: target.selectTrainingModel,
    onSelectPreset: target.selectTrainingPrimaryPreset,
    onSetTrainingPresets: target.setTrainingPresetSelection,
    onSetTrainingSnapshotSelection: target.setTrainingSnapshotSelection,
    onToggleTrainingPreset: target.toggleTrainingPreset,
    onToggleDraftTrainingPreset: target.toggleDraftTrainingPreset,
    onExcludeDraftTrainingPreset: target.excludeDraftTrainingPreset,
    onMakeTrainingPresetPrimary: target.makeTrainingPresetPrimary,
    onSelectAllTrainingPresets: target.selectAllTrainingPresets,
    onSelectPrimaryTrainingPreset: target.selectPrimaryTrainingPreset,
    onSelectExperimentTask: target.selectTrainingExperimentTask,
    onSetDatasets: target.setTrainingDatasetSelection,
    onToggleDataset: target.toggleTrainingDataset,
    onSelectAllDatasets: target.selectAllTrainingDatasets,
    onSelectFirstDataset: target.selectFirstTrainingDataset,
    onRemoveConfigSnapshot: target.removeConfigSnapshot,
    onIncludeConfigSnapshot: target.includeConfigSnapshot,
    onExcludeConfigSnapshot: target.excludeConfigSnapshot,
    onCreatePresetSnapshot: (preset) => {
      if (
        target.prepareTrainingPresetSnapshotDraft(preset)
      ) {
        onOpenFullConfig("snapshotDraft");
      }
    },
    onEditConfigSnapshot: (snapshotId) => {
      if (
        target.prepareTrainingSelectedSnapshotEdit(snapshotId)
      ) {
        onOpenFullConfig("snapshotEdit");
      }
    },
    onDuplicateConfigSnapshot: (snapshotId) => {
      if (
        target.prepareTrainingSelectedSnapshotEdit(snapshotId)
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

export const ConnectedTrainingPanel = ConnectedTrainingWorkspace;
