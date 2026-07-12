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
import { SectionHeading } from "@/components/ui/section-heading";
import { MultiSelectDropdown } from "@/features/workbench/components/screen/multi-select-dropdown";
import { SelectOnlyDropdown } from "@/features/workbench/components/screen/select-only-dropdown";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";
import { workbenchStatusCopy } from "@/features/workbench/components/shared/status-copy";
import { type TrainingWorkspace } from "@/features/workbench/state/training/use-training-workspace-state";
import {
  configSnapshotOverrideCount,
  configSnapshotOverrideCountLabel,
} from "@/lib/config-snapshots";

type TrainingSetup = TrainingWorkspace["draft"]["setup"];
type TrainingConfigTab = "presets" | "snapshots";

const setupIconClass = "h-[15px] w-[15px] text-violet";

/** Setup-only rendering Adapter for the grouped Training draft Interface. */
export function TrainingTargetDatasetPanel({
  setup,
  disabled = false,
}: {
  setup: TrainingSetup;
  disabled?: boolean;
}) {
  const { model, variants, experimentTask, datasets, monitors } = setup;
  const [activeConfigTab, setActiveConfigTab] =
    useState<TrainingConfigTab>("presets");
  const tabsId = useId();
  const presetsTabId = `${tabsId}-presets-tab`;
  const snapshotsTabId = `${tabsId}-snapshots-tab`;
  const presetsPanelId = `${tabsId}-presets-panel`;
  const snapshotsPanelId = `${tabsId}-snapshots-panel`;
  const snapshotMutationPending =
    variants.snapshotMutation.phase === "pending";
  const presetDisabledValues =
    variants.selectedPresets.length === 1 &&
    variants.selectedSnapshotIds.length === 0
      ? variants.selectedPresets
      : [];
  const datasetDisabledValues =
    datasets.selected.length === 1 ? datasets.selected : [];
  const presetOptions = variants.presetOptions.map((preset) => ({
    value: preset.value,
    label: preset.label,
    description: preset.value,
    actions: disabled
      ? undefined
      : [
          {
            label: `Create snapshot from ${preset.value}`,
            tooltip: `Create a Config Snapshot from ${preset.value} defaults`,
            icon: <FilePlus2 className="h-3.5 w-3.5" aria-hidden />,
            onAction: variants.createPresetSnapshot,
          },
        ],
  }));
  const snapshotOptions = variants.snapshots.map((snapshot) => ({
    value: snapshot.id,
    label: snapshot.name,
    description: `${snapshot.preset} · ${configSnapshotOverrideCountLabel(
      configSnapshotOverrideCount(snapshot),
    )}`,
    meta: <span>{snapshot.preset}</span>,
    actions:
      disabled || snapshotMutationPending
        ? undefined
        : [
            {
              label: `Edit snapshot ${snapshot.name}`,
              tooltip: "Edit this Config Snapshot",
              icon: <Pencil className="h-3.5 w-3.5" aria-hidden />,
              onAction: variants.editSnapshot,
            },
            {
              label: `Duplicate snapshot ${snapshot.name}`,
              tooltip: "Duplicate this Config Snapshot",
              icon: <Copy className="h-3.5 w-3.5" aria-hidden />,
              onAction: variants.duplicateSnapshot,
            },
            {
              label: `Delete snapshot ${snapshot.name}`,
              tooltip: "Delete this Config Snapshot",
              icon: <Trash2 className="h-3.5 w-3.5" aria-hidden />,
              onAction: variants.removeSnapshot,
            },
          ],
  }));
  const datasetOptions = datasets.options.map((dataset) => ({
    value: dataset.name,
    label: dataset.label,
    description: dataset.name,
    meta: (
      <span>
        {dataset.inputDim} {"->"} {dataset.outputDim}
      </span>
    ),
  }));
  const monitorOptions = monitors.options.map((monitor) => ({
    value: monitor.name,
    label: monitor.label,
    description: monitor.description,
    meta:
      monitor.kinds.length > 0 ? (
        <span>{monitor.kinds.join(" / ")}</span>
      ) : undefined,
  }));
  const modelTypeControl = model.typeOptions.length > 0 && (
    <SelectOnlyDropdown
      label="training model type"
      value={model.selectedType}
      options={model.typeOptions}
      onChange={model.selectType}
      placeholder="Select type"
      disabled={disabled}
    />
  );

  const presetsControls = (
    <>
      <MultiSelectDropdown
        label="Presets"
        values={variants.selectedPresets}
        options={presetOptions}
        onChange={variants.selectPresets}
        disabledValues={presetDisabledValues}
        disabled={disabled}
        primaryValue={variants.primaryPreset}
        onPrimaryChange={variants.makePresetPrimary}
        placeholder="Select presets"
        emptyMessage="No presets for this model"
      />
      {presetOptions.length === 0 && (
        <InlineStatus compact>No presets for this model</InlineStatus>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="secondary"
          onClick={variants.selectAllPresets}
          disabled={disabled || presetOptions.length === 0}
          className="h-9 text-[13px]"
        >
          All
        </Button>
        <Button
          variant="ghost"
          onClick={variants.selectOnlyPrimaryPreset}
          disabled={disabled || !variants.primaryPreset}
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
        values={variants.selectedSnapshotIds}
        options={snapshotOptions}
        onChange={variants.selectSnapshots}
        disabled={disabled || snapshotMutationPending}
        placeholder="Select snapshots"
        emptyMessage="No config snapshots for this model"
      />
      {variants.snapshots.length === 0 && (
        <InlineStatus compact>No config snapshots for this model</InlineStatus>
      )}
      {variants.snapshotMutation.phase === "pending" && (
        <InlineStatus busy compact>
          {variants.snapshotMutation.kind === "remove"
            ? "Removing Config Snapshot…"
            : "Updating Config Snapshot…"}
        </InlineStatus>
      )}
      {variants.snapshotMutation.phase === "failed" && (
        <InlineStatus tone="danger" role="alert" compact>
          <div className="grid gap-2">
            <span>{variants.snapshotMutation.error}</span>
            <div className="flex flex-wrap gap-2">
              {variants.snapshotMutation.canRetry && (
                <Button
                  variant="secondary"
                  onClick={() => void variants.retrySnapshotMutation()}
                  className="h-8 text-xs"
                >
                  Retry change
                </Button>
              )}
              <Button
                variant="ghost"
                onClick={variants.dismissSnapshotMutation}
                className="h-8 text-xs"
              >
                Dismiss
              </Button>
            </div>
          </div>
        </InlineStatus>
      )}
    </>
  );
  const configTabs = (
    <SegmentedControl
      aria-label="Training config selector"
      variant="tablist"
      className="grid w-full grid-cols-2"
    >
      <ViewModeButton
        variant="tab"
        id={presetsTabId}
        controls={presetsPanelId}
        active={activeConfigTab === "presets"}
        onClick={() => setActiveConfigTab("presets")}
      >
        <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
        Presets
      </ViewModeButton>
      <ViewModeButton
        variant="tab"
        id={snapshotsTabId}
        controls={snapshotsPanelId}
        active={activeConfigTab === "snapshots"}
        onClick={() => setActiveConfigTab("snapshots")}
      >
        <Camera className="h-3.5 w-3.5" aria-hidden />
        Snapshots
      </ViewModeButton>
    </SegmentedControl>
  );

  return (
    <div className="grid min-w-0 gap-3">
      {experimentTask.options.length > 0 && (
        <div className="grid min-w-0 gap-2">
          <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
            <SectionHeading
              icon={<Activity className={setupIconClass} aria-hidden />}
              title="Experiment Task"
            />
          </div>
          <SelectOnlyDropdown
            label="Experiment task"
            value={experimentTask.selected}
            options={experimentTask.options}
            onChange={experimentTask.select}
            placeholder="Select task"
            disabled={disabled}
          />
        </div>
      )}

      <div className="grid min-w-0 gap-2">
        {modelTypeControl && (
          <div className="grid min-w-0 gap-2">
            <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
              <SectionHeading
                icon={<Layers className={setupIconClass} aria-hidden />}
                title="Model Type"
              />
            </div>
            {modelTypeControl}
          </div>
        )}
        <div className="grid min-w-0 gap-2">
          <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
            <SectionHeading
              icon={<Cpu className={setupIconClass} aria-hidden />}
              title={modelTypeControl ? "Model Name" : "Model"}
            />
          </div>
          <SelectOnlyDropdown
            label="training model"
            value={model.selected}
            options={model.options}
            onChange={model.select}
            placeholder="Select model"
            disabled={disabled}
          />
        </div>
      </div>

      <div className="grid min-w-0 gap-2">
        <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
          <SectionHeading
            icon={<SlidersHorizontal className={setupIconClass} aria-hidden />}
            title="Variants"
          />
          <StatChip>
            {activeConfigTab === "snapshots"
              ? `${variants.selectedSnapshotIds.length} / ${variants.snapshots.length}`
              : `${variants.selectedPresets.length} / ${variants.presetOptions.length}`}
          </StatChip>
        </div>
        {configTabs}
        <div
          id={presetsPanelId}
          role="tabpanel"
          aria-labelledby={presetsTabId}
          aria-label="Presets"
          hidden={activeConfigTab !== "presets"}
          className="grid gap-2"
        >
          {activeConfigTab === "presets" ? presetsControls : null}
        </div>
        <div
          id={snapshotsPanelId}
          role="tabpanel"
          aria-labelledby={snapshotsTabId}
          aria-label="Snapshots"
          hidden={activeConfigTab !== "snapshots"}
          className="grid gap-2"
        >
          {activeConfigTab === "snapshots" ? snapshotControls : null}
        </div>
      </div>

      <div className="grid min-w-0 gap-2">
        <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
          <SectionHeading
            icon={<Database className={setupIconClass} aria-hidden />}
            title="Datasets"
          />
          <StatChip>
            {datasets.selected.length} / {datasets.options.length}
          </StatChip>
        </div>
        <MultiSelectDropdown
          label="Training datasets"
          values={datasets.selected}
          options={datasetOptions}
          onChange={datasets.select}
          disabledValues={datasetDisabledValues}
          disabled={disabled}
          placeholder="Select datasets"
          emptyMessage="No datasets for this model"
        />
        {datasets.options.length === 0 && (
          <InlineStatus compact>No datasets for this model</InlineStatus>
        )}
        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="secondary"
            onClick={datasets.selectAll}
            disabled={disabled || datasets.options.length === 0}
            className="h-9 text-[13px]"
          >
            All
          </Button>
          <Button
            variant="ghost"
            onClick={datasets.selectFirst}
            disabled={disabled || datasets.options.length === 0}
            className="h-9 border border-line bg-white/[0.025] text-[13px]"
          >
            First
          </Button>
        </div>
      </div>

      <div className="grid min-w-0 gap-2">
        <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
          <SectionHeading
            icon={<Activity className={setupIconClass} aria-hidden />}
            title="Signals"
          />
          <StatChip>
            {monitors.selected.length} / {monitors.options.length}
          </StatChip>
        </div>
        <MultiSelectDropdown
          label="Training monitors"
          values={monitors.selected}
          options={monitorOptions}
          onChange={monitors.select}
          disabled={disabled}
          placeholder={`${monitors.selected.length} / ${monitors.options.length} selected`}
          emptyMessage={workbenchStatusCopy.empty.optionalMonitors}
        />
        {monitors.isLoading && (
          <InlineStatus compact>
            {workbenchStatusCopy.loading.monitorOptions}
          </InlineStatus>
        )}
        {!monitors.isLoading && monitors.options.length === 0 && (
          <InlineStatus compact>
            {workbenchStatusCopy.empty.optionalMonitors}
          </InlineStatus>
        )}
        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="secondary"
            onClick={monitors.selectAll}
            disabled={disabled || monitors.options.length === 0}
            className="h-9 text-[13px]"
          >
            All
          </Button>
          <Button
            variant="ghost"
            onClick={monitors.clear}
            disabled={disabled || monitors.selected.length === 0}
            className="h-9 border border-line bg-white/[0.025] text-[13px]"
          >
            None
          </Button>
        </div>
      </div>
    </div>
  );
}
