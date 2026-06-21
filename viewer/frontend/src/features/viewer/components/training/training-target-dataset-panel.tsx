import { useId, useState } from "react";
import {
  Activity,
  Camera,
  Copy,
  Cpu,
  Database,
  FilePlus2,
  Layers,
  Pencil,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { MultiSelectDropdown } from "@/features/viewer/components/screen/multi-select-dropdown";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { StatChip } from "@/features/viewer/components/shared/stat-chip";
import { SurfacePanel } from "@/features/viewer/components/shared/surface-panel";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import { type Dataset, type MonitorOption } from "@/lib/api";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

type SelectOption = {
  value: string;
  label: string;
};

type TrainingConfigTab = "presets" | "snapshots";

const footerIconClass = "h-[15px] w-[15px] text-violet";
const defaultFieldLabelClass =
  "text-xs font-semibold tracking-[0.02em] text-ink-dim";
const fieldLabelWithIconClass = `${defaultFieldLabelClass} inline-flex items-center gap-1.5`;
const inlineFieldIconClass = "h-3.5 w-3.5 text-violet";

function overrideCountLabel(count: number) {
  return `${count} override${count === 1 ? "" : "s"}`;
}

export function TrainingTargetDatasetPanel({
  modelTypeOptions = [],
  modelOptions,
  selectedModelType = "",
  presetOptions,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets = selectedPreset ? [selectedPreset] : [],
  configSnapshots = [],
  selectedTrainingSnapshotIds = [],
  datasetOptions,
  selectedDatasets,
  monitorOptions = [],
  selectedMonitors = [],
  monitorsLoading = false,
  onSelectModelType,
  onSelectModel,
  onSelectPreset,
  onSetTrainingPresets,
  onSetTrainingSnapshotSelection,
  onToggleTrainingPreset,
  onMakeTrainingPresetPrimary,
  onSelectAllTrainingPresets,
  onSelectPrimaryTrainingPreset,
  onSetDatasets,
  onToggleDataset,
  onSelectAllDatasets,
  onSelectFirstDataset,
  onSetMonitors,
  onSelectAllMonitors,
  onClearMonitors,
  onCreatePresetSnapshot,
  onEditConfigSnapshot,
  onDuplicateConfigSnapshot,
  onDeleteConfigSnapshot,
  disabled = false,
  presentation = "default",
}: {
  modelTypeOptions?: SelectOption[];
  modelOptions: SelectOption[];
  selectedModelType?: string;
  presetOptions: SelectOption[];
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets?: string[];
  configSnapshots?: ConfigSnapshot[];
  selectedTrainingSnapshotIds?: string[];
  datasetOptions: Dataset[];
  selectedDatasets: string[];
  monitorOptions?: MonitorOption[];
  selectedMonitors?: string[];
  monitorsLoading?: boolean;
  onSelectModelType?: (modelType: string) => void;
  onSelectModel: (model: string) => void;
  onSelectPreset: (preset: string) => void;
  onSetTrainingPresets?: (presets: string[]) => void;
  onSetTrainingSnapshotSelection?: (snapshotIds: string[]) => void;
  onToggleTrainingPreset?: (preset: string) => void;
  onMakeTrainingPresetPrimary?: (preset: string) => void;
  onSelectAllTrainingPresets?: () => void;
  onSelectPrimaryTrainingPreset?: () => void;
  onSetDatasets?: (datasets: string[]) => void;
  onToggleDataset?: (dataset: string) => void;
  onSelectAllDatasets: () => void;
  onSelectFirstDataset: () => void;
  onSetMonitors?: (monitors: string[]) => void;
  onSelectAllMonitors?: () => void;
  onClearMonitors?: () => void;
  onCreatePresetSnapshot?: (preset: string) => void;
  onEditConfigSnapshot?: (snapshotId: string) => void;
  onDuplicateConfigSnapshot?: (snapshotId: string) => void;
  onDeleteConfigSnapshot?: (snapshotId: string) => void;
  disabled?: boolean;
  presentation?: "default" | "footer" | "setup";
}) {
  const isFooterPresentation = presentation === "footer";
  const isSetupPresentation = presentation === "setup";
  const [activeTrainingConfigTab, setActiveTrainingConfigTab] =
    useState<TrainingConfigTab>("presets");
  const trainingConfigTabsId = useId();
  const presetsTabId = `${trainingConfigTabsId}-presets-tab`;
  const snapshotsTabId = `${trainingConfigTabsId}-snapshots-tab`;
  const presetsPanelId = `${trainingConfigTabsId}-presets-panel`;
  const snapshotsPanelId = `${trainingConfigTabsId}-snapshots-panel`;
  const datasetCount = `${selectedDatasets.length} / ${datasetOptions.length}`;
  const monitorCount = `${selectedMonitors.length} / ${monitorOptions.length}`;
  const trainingPresetCount = `${selectedTrainingPresets.length} / ${presetOptions.length}`;
  const trainingSnapshotCount = `${selectedTrainingSnapshotIds.length} / ${configSnapshots.length}`;
  const trainingPresetDisabledValues =
    selectedTrainingPresets.length === 1 && selectedTrainingSnapshotIds.length === 0
      ? selectedTrainingPresets
      : [];
  const datasetDisabledValues =
    selectedDatasets.length === 1 ? selectedDatasets : [];
  const trainingPresetOptions = presetOptions.map((preset) => ({
    value: preset.value,
    label: preset.label,
    description: preset.value,
    actions: onCreatePresetSnapshot && !disabled
      ? [
          {
            label: `Create snapshot from ${preset.value}`,
            tooltip: `Create a Config Snapshot from ${preset.value} defaults`,
            icon: <FilePlus2 className="h-3.5 w-3.5" aria-hidden />,
            onAction: onCreatePresetSnapshot,
          },
        ]
      : undefined,
  }));
  const trainingSnapshotOptions = configSnapshots.map((snapshot) => {
    const overrideCount = Object.keys(snapshot.overrides).length;
    return {
      value: snapshot.id,
      label: snapshot.name,
      description: `${snapshot.preset} · ${overrideCountLabel(overrideCount)}`,
      meta: <span>{snapshot.preset}</span>,
      actions: disabled
        ? undefined
        : [
            ...(onEditConfigSnapshot
              ? [
                  {
                    label: `Edit snapshot ${snapshot.name}`,
                    tooltip: "Edit this Config Snapshot",
                    icon: <Pencil className="h-3.5 w-3.5" aria-hidden />,
                    onAction: onEditConfigSnapshot,
                  },
                ]
              : []),
            ...(onDuplicateConfigSnapshot
              ? [
                  {
                    label: `Duplicate snapshot ${snapshot.name}`,
                    tooltip: "Duplicate this Config Snapshot",
                    icon: <Copy className="h-3.5 w-3.5" aria-hidden />,
                    onAction: onDuplicateConfigSnapshot,
                  },
                ]
              : []),
            ...(onDeleteConfigSnapshot
              ? [
                  {
                    label: `Delete snapshot ${snapshot.name}`,
                    tooltip: "Delete this Config Snapshot",
                    icon: <Trash2 className="h-3.5 w-3.5" aria-hidden />,
                    onAction: onDeleteConfigSnapshot,
                  },
                ]
              : []),
          ],
    };
  });
  const trainingDatasetOptions = datasetOptions.map((dataset) => ({
    value: dataset.name,
    label: dataset.label,
    description: dataset.name,
    meta: (
      <span>
        {dataset.inputDim} {"->"} {dataset.outputDim}
      </span>
    ),
  }));
  const trainingMonitorOptions = monitorOptions.map((monitor) => ({
    value: monitor.name,
    label: monitor.label,
    description: monitor.description,
    meta:
      monitor.kinds.length > 0 ? (
        <span>{monitor.kinds.join(" / ")}</span>
      ) : undefined,
  }));
  const showMonitorField =
    Boolean(onSetMonitors) || monitorOptions.length > 0 || monitorsLoading;
  function changeTrainingPresets(nextPresets: string[]) {
    if (onSetTrainingPresets) {
      onSetTrainingPresets(nextPresets);
      return;
    }
    const changedPreset = presetOptions.find(
      (preset) =>
        selectedTrainingPresets.includes(preset.value) !==
        nextPresets.includes(preset.value),
    );
    if (changedPreset) {
      onToggleTrainingPreset?.(changedPreset.value);
    }
  }

  function changeTrainingSnapshots(nextSnapshotIds: string[]) {
    onSetTrainingSnapshotSelection?.(nextSnapshotIds);
  }

  function makeTrainingPresetPrimary(preset: string) {
    if (onMakeTrainingPresetPrimary) {
      onMakeTrainingPresetPrimary(preset);
      return;
    }
    onSelectPreset(preset);
  }

  function changeDatasets(nextDatasets: string[]) {
    if (onSetDatasets) {
      onSetDatasets(nextDatasets);
      return;
    }
    const changedDataset = datasetOptions.find(
      (dataset) =>
        selectedDatasets.includes(dataset.name) !==
        nextDatasets.includes(dataset.name),
    );
    if (changedDataset) {
      onToggleDataset?.(changedDataset.name);
    }
  }

  function changeMonitors(nextMonitors: string[]) {
    onSetMonitors?.(nextMonitors);
  }

  const modelTypeControl =
    modelTypeOptions.length > 0 && onSelectModelType ? (
      <SelectOnlyDropdown
        label="training model type"
        value={selectedModelType}
        options={modelTypeOptions}
        onChange={onSelectModelType}
        placeholder="Select type"
        disabled={disabled}
      />
    ) : null;

  const modelControl = (
    <SelectOnlyDropdown
      label="training model"
      value={selectedModel}
      options={modelOptions}
      onChange={onSelectModel}
      placeholder="Select model"
      disabled={disabled}
    />
  );
  const modelSelectorGridClass = modelTypeControl
    ? isSetupPresentation
      ? "grid min-w-0 gap-2"
      : "grid min-w-0 grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)] gap-2"
    : "grid min-w-0 gap-2";
  const modelTypeLabel = (
    <span className={fieldLabelWithIconClass}>
      <Layers className={inlineFieldIconClass} aria-hidden />
      Model Type
    </span>
  );
  const modelNameLabel = (
    <span className={fieldLabelWithIconClass}>
      <Cpu className={inlineFieldIconClass} aria-hidden />
      Model Name
    </span>
  );
  const modelLabel = (
    <span className={fieldLabelWithIconClass}>
      <Cpu className={inlineFieldIconClass} aria-hidden />
      Model
    </span>
  );

  const modelField = isFooterPresentation ? (
    <SurfacePanel
      className="min-w-0"
      icon={<Layers className={footerIconClass} aria-hidden />}
      title="Model"
    >
      <div className={modelSelectorGridClass}>
        {modelTypeControl && (
          <div className="grid min-w-0 gap-1.5">
            {modelTypeLabel}
            {modelTypeControl}
          </div>
        )}
        <div className="grid min-w-0 gap-1.5">
          {modelTypeControl && modelNameLabel}
          {modelControl}
        </div>
      </div>
    </SurfacePanel>
  ) : isSetupPresentation ? (
    <div className={modelSelectorGridClass}>
      {modelTypeControl && (
        <div className="grid min-w-0 gap-2">
          <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
            <SectionHeading
              icon={<Layers className={footerIconClass} aria-hidden />}
              title="Model Type"
            />
          </div>
          {modelTypeControl}
        </div>
      )}
      <div className="grid min-w-0 gap-2">
        <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
          <SectionHeading
            icon={<Cpu className={footerIconClass} aria-hidden />}
            title={modelTypeControl ? "Model Name" : "Model"}
          />
        </div>
        {modelControl}
      </div>
    </div>
  ) : (
    <div className={modelSelectorGridClass}>
      {modelTypeControl && (
        <div className="grid min-w-0 gap-1.5">
          {modelTypeLabel}
          {modelTypeControl}
        </div>
      )}
      <div className="grid min-w-0 gap-1.5">
        {modelTypeControl ? modelNameLabel : modelLabel}
        {modelControl}
      </div>
    </div>
  );

  const presetsControls = (
    <>
      <MultiSelectDropdown
        label="Presets"
        values={selectedTrainingPresets}
        options={trainingPresetOptions}
        onChange={changeTrainingPresets}
        disabledValues={trainingPresetDisabledValues}
        disabled={disabled}
        primaryValue={selectedPreset}
        onPrimaryChange={makeTrainingPresetPrimary}
        placeholder="Select presets"
        emptyMessage="No presets for this model"
      />
      {presetOptions.length === 0 && (
        <InlineStatus compact>
          No presets for this model
        </InlineStatus>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="secondary"
          onClick={onSelectAllTrainingPresets}
          disabled={disabled || presetOptions.length === 0 || !onSelectAllTrainingPresets}
          className="h-9 text-[13px]"
        >
          All
        </Button>
        <Button
          variant="ghost"
          onClick={onSelectPrimaryTrainingPreset}
          disabled={disabled || !selectedPreset || !onSelectPrimaryTrainingPreset}
          className="h-9 border border-line bg-white/[0.025] text-[13px]"
        >
          Primary only
        </Button>
      </div>
    </>
  );

  const snapshotControls = (
    <>
      <MultiSelectDropdown
        label="Config snapshots"
        values={selectedTrainingSnapshotIds}
        options={trainingSnapshotOptions}
        onChange={changeTrainingSnapshots}
        disabled={disabled}
        placeholder="Select snapshots"
        emptyMessage="No config snapshots for this model"
      />
      {configSnapshots.length === 0 && (
        <InlineStatus compact>
          No config snapshots for this model
        </InlineStatus>
      )}
    </>
  );

  const trainingConfigTabs = (
    <SegmentedControl
      aria-label="Training config selector"
      variant="tablist"
      className="grid w-full grid-cols-2"
    >
      <ViewModeButton
        variant="tab"
        id={presetsTabId}
        controls={presetsPanelId}
        active={activeTrainingConfigTab === "presets"}
        onClick={() => setActiveTrainingConfigTab("presets")}
      >
        <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
        Presets
      </ViewModeButton>
      <ViewModeButton
        variant="tab"
        id={snapshotsTabId}
        controls={snapshotsPanelId}
        active={activeTrainingConfigTab === "snapshots"}
        onClick={() => setActiveTrainingConfigTab("snapshots")}
      >
        <Camera className="h-3.5 w-3.5" aria-hidden />
        Snapshots
      </ViewModeButton>
    </SegmentedControl>
  );

  const presetsField = isFooterPresentation ? (
    <SurfacePanel
      className="min-w-0"
      icon={<SlidersHorizontal className={footerIconClass} aria-hidden />}
      title="Presets"
      detail={
        <StatChip>
          {activeTrainingConfigTab === "snapshots"
            ? trainingSnapshotCount
            : trainingPresetCount}
        </StatChip>
      }
    >
      {trainingConfigTabs}
      <div
        id={presetsPanelId}
        role="tabpanel"
        aria-labelledby={presetsTabId}
        aria-label="Presets"
        hidden={activeTrainingConfigTab !== "presets"}
        className="grid gap-2"
      >
        {activeTrainingConfigTab === "presets" ? presetsControls : null}
      </div>
      <div
        id={snapshotsPanelId}
        role="tabpanel"
        aria-labelledby={snapshotsTabId}
        aria-label="Snapshots"
        hidden={activeTrainingConfigTab !== "snapshots"}
        className="grid gap-2"
      >
        {activeTrainingConfigTab === "snapshots" ? snapshotControls : null}
      </div>
    </SurfacePanel>
  ) : isSetupPresentation ? (
    <div className="grid min-w-0 gap-2">
      <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
        <SectionHeading
          icon={<SlidersHorizontal className={footerIconClass} aria-hidden />}
          title="Variants"
        />
        <StatChip>
          {activeTrainingConfigTab === "snapshots"
            ? trainingSnapshotCount
            : trainingPresetCount}
        </StatChip>
      </div>
      {trainingConfigTabs}
      <div
        id={presetsPanelId}
        role="tabpanel"
        aria-labelledby={presetsTabId}
        aria-label="Presets"
        hidden={activeTrainingConfigTab !== "presets"}
        className="grid gap-2"
      >
        {activeTrainingConfigTab === "presets" ? presetsControls : null}
      </div>
      <div
        id={snapshotsPanelId}
        role="tabpanel"
        aria-labelledby={snapshotsTabId}
        aria-label="Snapshots"
        hidden={activeTrainingConfigTab !== "snapshots"}
        className="grid gap-2"
      >
        {activeTrainingConfigTab === "snapshots" ? snapshotControls : null}
      </div>
    </div>
  ) : (
    <div className="grid min-w-0 gap-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className={defaultFieldLabelClass}>Presets</div>
        <StatChip>{trainingPresetCount}</StatChip>
      </div>
      {presetsControls}
    </div>
  );

  const datasetsControls = (
    <>
      <MultiSelectDropdown
        label="Training datasets"
        values={selectedDatasets}
        options={trainingDatasetOptions}
        onChange={changeDatasets}
        disabledValues={datasetDisabledValues}
        disabled={disabled}
        placeholder="Select datasets"
        emptyMessage="No datasets for this model"
      />
      {datasetOptions.length === 0 && (
        <InlineStatus compact>
          No datasets for this model
        </InlineStatus>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="secondary"
          onClick={onSelectAllDatasets}
          disabled={disabled || datasetOptions.length === 0}
          className="h-9 text-[13px]"
        >
          All
        </Button>
        <Button
          variant="ghost"
          onClick={onSelectFirstDataset}
          disabled={disabled || datasetOptions.length === 0}
          className="h-9 border border-line bg-white/[0.025] text-[13px]"
        >
          First
        </Button>
      </div>
    </>
  );

  const datasetsField = isFooterPresentation ? (
    <SurfacePanel
      className="xl:min-h-0"
      icon={<Database className={footerIconClass} aria-hidden />}
      title="Datasets"
      detail={<StatChip>{datasetCount}</StatChip>}
    >
      {datasetsControls}
    </SurfacePanel>
  ) : isSetupPresentation ? (
    <div className="grid min-w-0 gap-2">
      <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
        <SectionHeading
          icon={<Database className={footerIconClass} aria-hidden />}
          title="Datasets"
        />
        <StatChip>{datasetCount}</StatChip>
      </div>
      {datasetsControls}
    </div>
  ) : (
    <div className="xl:min-h-0 grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionHeading
          icon={<Database className="h-[15px] w-[15px] text-violet" aria-hidden />}
          title="Datasets"
        />
        <StatChip>{datasetCount}</StatChip>
      </div>
      {datasetsControls}
    </div>
  );

  const monitorsControls = (
    <>
      <MultiSelectDropdown
        label="Training monitors"
        values={selectedMonitors}
        options={trainingMonitorOptions}
        onChange={changeMonitors}
        disabled={disabled}
        placeholder={`${monitorCount} selected`}
        emptyMessage={viewerStatusCopy.empty.optionalMonitors}
      />
      {monitorsLoading && (
        <InlineStatus compact>
          {viewerStatusCopy.loading.monitorOptions}
        </InlineStatus>
      )}
      {!monitorsLoading && monitorOptions.length === 0 && (
        <InlineStatus compact>
          {viewerStatusCopy.empty.optionalMonitors}
        </InlineStatus>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="secondary"
          onClick={onSelectAllMonitors}
          disabled={disabled || monitorOptions.length === 0 || !onSelectAllMonitors}
          className="h-9 text-[13px]"
        >
          All
        </Button>
        <Button
          variant="ghost"
          onClick={onClearMonitors}
          disabled={disabled || selectedMonitors.length === 0 || !onClearMonitors}
          className="h-9 border border-line bg-white/[0.025] text-[13px]"
        >
          None
        </Button>
      </div>
    </>
  );

  const monitorsField = !showMonitorField ? null : isFooterPresentation ? (
    <SurfacePanel
      className="xl:min-h-0"
      icon={<Activity className={footerIconClass} aria-hidden />}
      title="Signals"
      detail={<StatChip>{monitorCount}</StatChip>}
    >
      {monitorsControls}
    </SurfacePanel>
  ) : isSetupPresentation ? (
    <div className="grid min-w-0 gap-2">
      <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
        <SectionHeading
          icon={<Activity className={footerIconClass} aria-hidden />}
          title="Signals"
        />
        <StatChip>{monitorCount}</StatChip>
      </div>
      {monitorsControls}
    </div>
  ) : (
    <div className="xl:min-h-0 grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionHeading
          icon={<Activity className="h-[15px] w-[15px] text-violet" aria-hidden />}
          title="Signals"
        />
        <StatChip>{monitorCount}</StatChip>
      </div>
      {monitorsControls}
    </div>
  );

  if (isFooterPresentation) {
    return (
      <>
        {modelField}
        {presetsField}
        {datasetsField}
        {monitorsField}
      </>
    );
  }

  if (isSetupPresentation) {
    return (
      <div className="grid min-w-0 gap-3">
        {modelField}
        {presetsField}
        {datasetsField}
        {monitorsField}
      </div>
    );
  }

  return (
    <div className="grid content-start gap-3 xl:h-full xl:grid-rows-[auto_minmax(0,1fr)] xl:content-stretch">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
        {modelField}
        {presetsField}
      </div>

      {datasetsField}
      {monitorsField}
    </div>
  );
}
