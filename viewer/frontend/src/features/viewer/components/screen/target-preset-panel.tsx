import { useMemo, useState } from "react";
import { Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TrainingCommandDialog } from "@/features/viewer/components/config/training-command-dialog";
import { TargetSelectorSection } from "@/features/viewer/components/screen/target-selector-section";
import {
  useHistoricalRuns,
  useTargetSelectorState,
} from "@/features/viewer/providers/viewer-providers";
import { type FullConfigDialogControls } from "@/features/viewer/state/use-viewer-workspace-shell";
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
  const {
    selectedModelType,
    selectedModel,
    selectedTargetMode,
    activateTargetPresetMode,
    activateTargetSnapshotMode,
    activateTargetExperimentMode,
    selectedPreset,
    selectedSnapshotId,
    selectedConfigSnapshot,
    configSnapshotsEnabled,
    isSchemaReady,
    selectedDatasets,
    activeOverrides,
    effectivePresetOverrides,
    configSections,
    targetMonitors,
    targetMonitorsLoading,
    selectModelType: onSelectModelType,
    selectModel: onSelectModel,
    selectPreset: onSelectPreset,
    selectSnapshot: onSelectSnapshot,
    preparePresetSnapshotDraft,
    prepareSelectedSnapshotEdit,
    models,
    presets,
    snapshots,
  } = useTargetSelectorState();
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
  const [trainingCommandMode, setTrainingCommandMode] =
    useState<TrainingCommandMode | null>(null);
  const [includeAllMonitors, setIncludeAllMonitors] = useState(false);
  const presetSelectId = "target-preset-select";
  const snapshotSelectId = "target-snapshot-select";
  const experimentSelectId = "target-experiment-select";
  const experimentDatasetSelectId = "target-experiment-dataset-select";
  const experimentPresetSelectId = "target-experiment-preset-select";
  const modelTypeOptions = createModelTypeOptions(models);
  const modelOptions = modelsForType(models, selectedModelType).map((model) => ({
    value: model.model,
    label: modelNameForId(model),
  }));
  const presetOptions = presets.map((preset) => ({
    value: preset.name,
    label: preset.name,
  }));
  const snapshotOptions = snapshots.map((snapshot) => ({
    value: snapshot.id,
    label: snapshot.name,
  }));
  const experimentOptions = historicalExperimentOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
  }));
  const experimentDatasetOptions = historicalDatasetOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
  }));
  const experimentPresetOptions = historicalPresetOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
  }));
  const selectedSnapshotName = selectedConfigSnapshot?.name ?? "";
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
    if (preparePresetSnapshotDraft(selectedPreset)) {
      onOpenFullConfig("snapshotDraft");
    }
  };

  const editSelectedSnapshot = () => {
    if (selectedSnapshotId && prepareSelectedSnapshotEdit(selectedSnapshotId)) {
      onOpenFullConfig("snapshotEdit");
    }
  };

  const duplicateSelectedSnapshot = () => {
    if (selectedSnapshotId && onSelectSnapshot(selectedSnapshotId)) {
      onOpenFullConfig("snapshotDraft");
    }
  };

  return (
    <>
      <TargetSelectorSection
        selectedModelType={selectedModelType}
        selectedModel={selectedModel}
        selectedTargetMode={selectedTargetMode}
        selectedPreset={selectedPreset}
        selectedSnapshotId={selectedSnapshotId}
        selectedSnapshotName={selectedSnapshotName}
        selectedHistoricalExperimentFilter={selectedHistoricalExperimentFilter}
        selectedHistoricalDatasetFilter={selectedHistoricalDatasetFilter}
        selectedHistoricalPreset={selectedHistoricalPreset}
        configSnapshotsEnabled={configSnapshotsEnabled}
        isSchemaReady={isSchemaReady}
        modelTypeOptions={modelTypeOptions}
        modelOptions={modelOptions}
        presetOptions={presetOptions}
        snapshotOptions={snapshotOptions}
        experimentOptions={experimentOptions}
        experimentDatasetOptions={experimentDatasetOptions}
        experimentPresetOptions={experimentPresetOptions}
        presetSelectId={presetSelectId}
        snapshotSelectId={snapshotSelectId}
        experimentSelectId={experimentSelectId}
        experimentDatasetSelectId={experimentDatasetSelectId}
        experimentPresetSelectId={experimentPresetSelectId}
        presetTrainingCommandDisabled={!canOpenPresetTrainingCommand}
        snapshotTrainingCommandDisabled={!canOpenSnapshotTrainingCommand}
        onSelectModelType={onSelectModelType}
        onSelectModel={onSelectModel}
        onActivatePresetMode={activateTargetPresetMode}
        onActivateSnapshotMode={activateTargetSnapshotMode}
        onActivateExperimentMode={activateTargetExperimentMode}
        onSelectPreset={onSelectPreset}
        onSelectSnapshot={onSelectSnapshot}
        onSelectHistoricalExperimentFilter={setSelectedHistoricalExperimentFilter}
        onSelectHistoricalDatasetFilter={setSelectedHistoricalDatasetFilter}
        onSelectHistoricalPreset={setSelectedHistoricalPreset}
        onCreateSnapshot={createPresetSnapshot}
        onEditSnapshot={editSelectedSnapshot}
        onDuplicateSnapshot={duplicateSelectedSnapshot}
        onOpenPresetTrainingCommand={() => openTrainingCommand("preset")}
        onOpenSnapshotTrainingCommand={() => openTrainingCommand("snapshot")}
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
                  ? "Loading monitors"
                  : targetMonitorNames.length > 0
                    ? `${targetMonitorNames.length} optional monitors`
                    : "No optional monitors"
              }
            >
              <Activity className="h-4 w-4" aria-hidden />
              <span>Monitors</span>
              <span className="rounded-full border border-current/25 px-1.5 py-0.5 text-[0.65rem] font-bold uppercase leading-none opacity-90">
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
