import { useMemo, useState } from "react";
import { Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TrainingCommandDialog } from "@/features/workbench/components/config/training-command-dialog";
import {
  TargetSelectorSection,
  type TargetSelectorCommands,
  type TargetSelectorView,
} from "@/features/workbench/components/screen/target-selector-section";
import {
  useConfigSnapshotRecords,
  useConfigSnapshotEditor,
  useHistoricalRuns,
  useModelPackageCatalog,
  useModelPackageInspection,
} from "@/features/workbench/providers/workbench-providers";
import { useWorkbenchCapabilities } from "@/features/workbench/providers/workbench-connection-provider";
import { type FullConfigDialogControls } from "@/features/workbench/state/use-workbench-workspace-shell";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import {
  modelNameForId,
  modelsForType,
  modelTypeOptions as createModelTypeOptions,
} from "@/lib/selection";
import { buildTrainingCommand } from "@/lib/training-command";

function historicalFilterLabel(option: { label: string; count: number }) {
  return option.count > 1 ? `${option.label} (${option.count})` : option.label;
}

type TrainingCommandMode = "preset" | "snapshot";

export function TargetPresetPanel({
  onOpenFullConfig,
}: {
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  const { modelPackages } = useModelPackageCatalog();
  const { capabilities } = useWorkbenchCapabilities();
  const snapshotRecords = useConfigSnapshotRecords();
  const { target, browser, options, runtimeDefaults, status, actions } =
    useModelPackageInspection();
  const selectedModelType = browser.selectedModelType;
  const selectedModel = browser.selectedModel;
  const selectedTargetMode = browser.mode;
  const activateTargetPresetMode = actions.showPresetTarget;
  const activateTargetSnapshotMode = actions.showSnapshotTarget;
  const activateTargetExperimentMode = actions.browseHistoricalRuns;
  const selectedPreset = browser.selectedPreset;
  const selectedSnapshotId = browser.selectedSnapshotId;
  const selectedConfigSnapshot =
    target.kind === "snapshot" ? target.snapshot : undefined;
  const selectedExperimentTask = browser.selectedExperimentTask;
  const experimentTaskOptions = options.experimentTasks;
  const configSnapshotsEnabled = capabilities.configSnapshotsEnabled;
  const isSchemaReady = status.schema.isReady;
  const selectedDatasets = browser.selectedDatasets;
  const activeOverrides = runtimeDefaults.active;
  const effectivePresetOverrides = runtimeDefaults.effectivePreset;
  const configSections = options.configSections;
  const targetMonitors = options.monitorMetadata;
  const targetMonitorsLoading = status.monitors.isLoading;
  const onSelectModelType = actions.selectModelType;
  const onSelectModel = actions.selectModelPackage;
  const onSelectPreset = actions.selectPresetTarget;
  const onSelectSnapshot = actions.selectSnapshotTarget;
  const onSelectExperimentTask = actions.selectExperimentTask;
  const models = modelPackages.records;
  const presets = options.presets;
  const snapshots = snapshotRecords.records.all;
  const {
    historicalExperimentOptions,
    historicalDatasetOptions,
    historicalPresetOptions,
    selectedHistoricalExperimentFilter,
    setSelectedHistoricalExperimentFilter,
    selectedHistoricalDatasetFilter,
    setSelectedHistoricalDatasetFilter,
    selectedHistoricalPreset,
    setSelectedHistoricalPreset,
  } = useHistoricalRuns();
  const snapshotEditor = useConfigSnapshotEditor();
  const [trainingCommandMode, setTrainingCommandMode] =
    useState<TrainingCommandMode | null>(null);
  const [includeAllMonitors, setIncludeAllMonitors] = useState(false);
  const modelTypeOptions = createModelTypeOptions(models);
  const modelOptions = modelsForType(models, selectedModelType).map((model) => ({
    value: model.model,
    label: modelNameForId(model),
  }));
  const presetOptions = presets.map((preset) => ({
    value: preset.name,
    label: preset.name,
  }));
  const presetControlValue =
    target.kind === "preset" ? selectedPreset : "";
  const snapshotOptions = snapshots.map((snapshot) => ({
    value: snapshot.id,
    label: snapshot.name,
  }));
  const experimentOptions = historicalExperimentOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
    description: option.description,
  }));
  const experimentDatasetOptions = historicalDatasetOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
    description: option.description,
  }));
  const experimentPresetOptions = historicalPresetOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
    description: option.description,
  }));
  const selectedSnapshotName = selectedConfigSnapshot?.name ?? "";
  const hasSnapshots = snapshotOptions.length > 0;
  const canActivateHistoricalMode =
    Boolean(selectedModel) || experimentOptions.length > 0;
  const activeTargetMode =
    selectedTargetMode === "snapshot" && hasSnapshots
      ? "snapshot"
      : selectedTargetMode === "experiment" && canActivateHistoricalMode
        ? "experiment"
        : "preset";
  const snapshotValue =
    activeTargetMode === "snapshot" &&
    snapshotOptions.some((option) => option.value === selectedSnapshotId)
      ? selectedSnapshotId
      : "";
  const historicalExperimentValue =
    activeTargetMode === "experiment" &&
    experimentOptions.some(
      (option) => option.value === selectedHistoricalExperimentFilter,
    )
      ? selectedHistoricalExperimentFilter
      : "";
  const historicalDatasetValue =
    activeTargetMode === "experiment" &&
    experimentDatasetOptions.some(
      (option) => option.value === selectedHistoricalDatasetFilter,
    )
      ? selectedHistoricalDatasetFilter
      : "";
  const historicalPresetValue =
    activeTargetMode === "experiment" &&
    experimentPresetOptions.some(
      (option) => option.value === selectedHistoricalPreset,
    )
      ? selectedHistoricalPreset
      : "";
  const targetMonitorNames = useMemo(
    () => targetMonitors.map((monitor) => monitor.name),
    [targetMonitors],
  );
  const canOpenPresetTrainingCommand = Boolean(
    selectedModelType &&
      selectedModel &&
      selectedPreset &&
      isSchemaReady &&
      selectedDatasets.length > 0,
  );
  const canOpenSnapshotTrainingCommand = Boolean(
    canOpenPresetTrainingCommand &&
      selectedTargetMode === "snapshot" &&
      selectedSnapshotId &&
      selectedConfigSnapshot,
  );
  const snapshotActionsDisabled = Boolean(
    !configSnapshotsEnabled ||
      !isSchemaReady ||
      !selectedModel ||
      !selectedPreset,
  );
  const targetSelectorView = {
    modelPackage: {
      modelType: selectedModelType,
      model: selectedModel,
      modelTypes: modelTypeOptions,
      models: modelOptions,
    },
    experimentTask: {
      visible: experimentTaskOptions.length > 0,
      value: selectedExperimentTask,
      options: experimentTaskOptions,
    },
    source: {
      activeMode: activeTargetMode,
      snapshotAvailable: hasSnapshots,
      historicalAvailable: canActivateHistoricalMode,
      preset: {
        value: presetControlValue,
        options: presetOptions,
        trainingCommandDisabled: !canOpenPresetTrainingCommand,
        createSnapshotDisabled: snapshotActionsDisabled,
      },
      snapshot: {
        value: snapshotValue,
        name: selectedSnapshotName,
        options: snapshotOptions,
        trainingCommandDisabled: !canOpenSnapshotTrainingCommand,
        actionsDisabled: snapshotActionsDisabled,
      },
      historical: {
        experiment: {
          value: historicalExperimentValue,
          options: experimentOptions,
        },
        dataset: {
          value: historicalDatasetValue,
          options: experimentDatasetOptions,
        },
        preset: {
          value: historicalPresetValue,
          options: experimentPresetOptions,
        },
      },
    },
  } satisfies TargetSelectorView;
  const commandPreset =
    trainingCommandMode === "snapshot"
      ? selectedConfigSnapshot?.preset ?? selectedPreset
      : selectedPreset;
  const commandOverrides =
    trainingCommandMode === "snapshot" ? activeOverrides : effectivePresetOverrides;
  const trainingCommand = useMemo(
    () =>
      trainingCommandMode
        ? buildTrainingCommand({
            modelType: selectedModelType,
            model: selectedModel,
            preset: commandPreset,
            experimentTask: selectedExperimentTask,
            datasets: selectedDatasets,
            monitors: includeAllMonitors ? targetMonitorNames : undefined,
            sections: configSections,
            overrides: commandOverrides,
          })
        : "",
    [
      commandOverrides,
      commandPreset,
      configSections,
      includeAllMonitors,
      selectedDatasets,
      selectedExperimentTask,
      selectedModel,
      selectedModelType,
      targetMonitorNames,
      trainingCommandMode,
    ],
  );
  const { status: copyStatus, copy: copyTrainingCommand } =
    useCopyToClipboard(trainingCommand);

  function openTrainingCommand(mode: TrainingCommandMode) {
    if (
      (mode === "preset" && !canOpenPresetTrainingCommand) ||
      (mode === "snapshot" && !canOpenSnapshotTrainingCommand)
    ) {
      return;
    }
    setIncludeAllMonitors(false);
    setTrainingCommandMode(mode);
  }

  function closeTrainingCommand() {
    setTrainingCommandMode(null);
    setIncludeAllMonitors(false);
  }

  const createPresetSnapshot = () => {
    if (
      snapshotEditor.actions.beginDraft({
        modelType: selectedModelType,
        model: selectedModel,
        preset: selectedPreset,
      })
    ) {
      onOpenFullConfig("snapshotDraft");
    }
  };

  const editSelectedSnapshot = () => {
    if (
      selectedConfigSnapshot &&
      snapshotEditor.actions.beginEdit(selectedConfigSnapshot)
    ) {
      onOpenFullConfig("snapshotEdit");
    }
  };

  const duplicateSelectedSnapshot = () => {
    if (
      selectedConfigSnapshot &&
      snapshotEditor.actions.beginDuplicate(selectedConfigSnapshot)
    ) {
      onOpenFullConfig("snapshotDraft");
    }
  };
  const targetSelectorCommands = {
    selectModelType: onSelectModelType,
    selectModel: onSelectModel,
    selectExperimentTask: onSelectExperimentTask,
    showPreset: activateTargetPresetMode,
    showSnapshot: activateTargetSnapshotMode,
    showHistorical: activateTargetExperimentMode,
    selectPreset: onSelectPreset,
    selectSnapshot: onSelectSnapshot,
    selectHistoricalExperiment: setSelectedHistoricalExperimentFilter,
    selectHistoricalDataset: setSelectedHistoricalDatasetFilter,
    selectHistoricalPreset: setSelectedHistoricalPreset,
    createSnapshot: createPresetSnapshot,
    editSnapshot: editSelectedSnapshot,
    duplicateSnapshot: duplicateSelectedSnapshot,
    openPresetTrainingCommand: () => openTrainingCommand("preset"),
    openSnapshotTrainingCommand: () => openTrainingCommand("snapshot"),
  } satisfies TargetSelectorCommands;

  return (
    <>
      <TargetSelectorSection
        view={targetSelectorView}
        commands={targetSelectorCommands}
      />

      {trainingCommandMode && (
        <TrainingCommandDialog
          model={selectedModel}
          preset={
            trainingCommandMode === "snapshot"
              ? selectedSnapshotName || commandPreset
              : commandPreset
          }
          trainingCommand={trainingCommand}
          copyStatus={copyStatus}
          footerStart={
            <Button
              variant={includeAllMonitors ? "primary" : "secondary"}
              aria-label="Include all monitors"
              aria-pressed={includeAllMonitors}
              disabled={targetMonitorsLoading || targetMonitorNames.length === 0}
              onClick={() => setIncludeAllMonitors((current) => !current)}
              className="min-w-[11rem] justify-between px-3"
              title={
                targetMonitorsLoading
                  ? "Loading monitors…"
                  : targetMonitorNames.length > 0
                    ? `${targetMonitorNames.length} optional monitors`
                    : "No optional monitors"
              }
            >
              <Activity className="h-4 w-4" aria-hidden />
              <span>Monitors</span>
              <span className="rounded-full border border-current/25 px-1.5 py-0.5 type-caption font-bold uppercase leading-none opacity-90">
                {includeAllMonitors ? "On" : "Off"}
              </span>
            </Button>
          }
          onCopy={copyTrainingCommand}
          onClose={closeTrainingCommand}
        />
      )}
    </>
  );
}
