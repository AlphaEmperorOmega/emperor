import {
  Activity,
  ChevronDown,
  ChevronUp,
  CircleStop,
  FolderOpen,
  FolderPlus,
  ListChecks,
  Loader2,
  Maximize2,
  Play,
  RotateCcw,
  SlidersHorizontal,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { Select } from "@/components/ui/select";
import { TrainingSearchSetup } from "@/features/viewer/components/training-search-setup";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { MultiSelectDropdown } from "@/features/viewer/components/screen/multi-select-dropdown";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { StatChip } from "@/features/viewer/components/shared/stat-chip";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import { TrainingTargetDatasetPanel } from "@/features/viewer/components/training/training-target-dataset-panel";
import { TrainingFooterField } from "@/features/viewer/components/training/training-footer-field";
import { TrainingProgressDialog } from "@/features/viewer/components/training/training-progress-dialog";
import {
  type TrainingPanelViewModel,
} from "@/features/viewer/state/training/use-training-panel-view-model";

type TrainingPanelProps = {
  viewModel: TrainingPanelViewModel;
};

const footerFieldLabelClass =
  "flex items-center gap-2 text-xs font-bold uppercase tracking-[0.08em] text-ink-faint";
const footerIconClass = "h-[15px] w-[15px] text-violet";

export function TrainingPanel({ viewModel }: TrainingPanelProps) {
  const { actions, input, logFolder, options, request, status, training, ui } =
    viewModel;
  const {
    datasetOptions,
    configSections,
    selectedModel,
    selectedPreset,
    selectedTrainingPresets,
    selectedDatasets,
    overrides,
    configSnapshotCount,
    monitorOptions,
    selectedMonitors,
    monitorsLoading,
    searchAxes,
    searchLoading,
    trainingEnabled,
    canOpenFullConfig,
    onSelectModel,
    onSelectPreset,
    onSetTrainingPresets,
    onToggleTrainingPreset,
    onMakeTrainingPresetPrimary,
    onSelectAllTrainingPresets,
    onSelectPrimaryTrainingPreset,
    onSetDatasets,
    onToggleDataset,
    onSelectAllDatasets,
    onSelectFirstDataset,
    onResetOverrides,
    onOpenFullConfig,
    onRemoveConfigSnapshot,
    onTrainingSearchChange,
  } = input;
  const { changeMonitors } = actions;
  const {
    existingHelp,
    existingValue: selectedExistingLogFolder,
    isLoading: logFoldersLoading,
    mode: logFolderMode,
    newError: newLogFolderError,
    newValid: newLogFolderValid,
    newValue: newLogFolder,
    options: logFolderOptions,
    setExistingValue: setSelectedExistingLogFolder,
    setNewValue: setNewLogFolder,
  } = logFolder;
  const { modelOptions, presetOptions, trainingMonitorOptions } = options;
  const {
    activeSearchAxisCount,
    canRequestTraining,
    effectiveTrainingSearch,
    fieldCount,
    hasConfigSnapshots,
    overrideCount,
    searchConflictKeys,
    searchModeLabel,
    selectedFieldSummary,
    selectedTrainingPresetCount,
    trainingSearchValidation,
  } = request;
  const {
    job,
    progressRunPlan,
    displayedRunCount,
    isProgressPlanning,
    progressPlanError,
    isRunning,
    canStart,
    canResampleRunPlan,
    isResampling,
    isStarting,
    isCancelling,
    trainingError,
    showLargeGridConfirmation,
    startTraining,
    confirmLargeGridSearch,
    cancelLargeGridSearch,
    cancelTraining,
    resampleRunPlan,
  } = training;
  const {
    activeSearchLabel,
    clusterGrowth,
    currentDataset,
    currentPreset,
    datasetCountLabel,
    epochStep,
    historicalTrainingLockMessage,
    jobStatus,
    logFolderLabel,
    metricLabel: jobMetricLabel,
    monitorCount,
    plannedRunLabel,
    presetCountLabel,
    progressButtonLabel,
  } = status;
  const logFolderModeControl = (
    <SegmentedControl aria-label="Log folder mode">
      <ViewModeButton
        active={logFolder.mode === "existing"}
        onClick={() => logFolder.setMode("existing")}
      >
        <FolderOpen className="h-3.5 w-3.5" aria-hidden />
        Existing folder
      </ViewModeButton>
      <ViewModeButton
        active={logFolder.mode === "new"}
        onClick={() => logFolder.setMode("new")}
      >
        <FolderPlus className="h-3.5 w-3.5" aria-hidden />
        New folder
      </ViewModeButton>
    </SegmentedControl>
  );

  return (
    <section className="border-t border-line bg-[linear-gradient(0deg,rgba(14,12,24,0.7),rgba(8,8,14,0.4))] backdrop-blur-xl">
      <div className="grid h-16 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-[22px]">
        <button
          type="button"
          onClick={ui.toggleExpanded}
          className="flex min-w-0 items-center gap-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          aria-expanded={ui.isExpanded}
          aria-controls="training-panel-details"
          aria-label={`Training ${jobStatus} ${currentDataset} ${epochStep} · ${jobMetricLabel}`}
        >
          <span className="relative grid h-[38px] w-[38px] shrink-0 place-items-center rounded-[11px] border border-violet/35 bg-[linear-gradient(135deg,#2a2740,#16142a)] text-[15px] font-extrabold text-white">
            N
            <span className="absolute -bottom-1 -right-1 grid h-4 w-4 place-items-center rounded-full border border-line bg-panel text-ink-faint">
              {ui.isExpanded ? (
                <ChevronDown className="h-3 w-3" aria-hidden />
              ) : (
                <ChevronUp className="h-3 w-3" aria-hidden />
              )}
            </span>
          </span>
          <span className="grid min-w-0 gap-1">
            <span className="flex min-w-0 flex-wrap items-center gap-2">
              <span className="text-sm font-bold text-ink">Training</span>
              {historicalTrainingLockMessage && (
                <span
                  className="rounded-full border border-danger-line bg-danger-soft px-2 py-0.5 text-[11px] font-bold text-danger-text"
                  role="status"
                >
                  {historicalTrainingLockMessage}
                </span>
              )}
              <Badge
                className={
                  jobStatus === "failed"
                    ? "border-danger-line bg-danger-soft text-danger-text"
                    : jobStatus === "completed"
                      ? "border-ok/30 bg-ok/10 text-ok"
                      : "border-line bg-white/[0.05] text-ink-faint"
                }
              >
                {jobStatus}
              </Badge>
              <span className="truncate font-mono text-xs text-ink-dim">
                {currentPreset
                  ? `${currentPreset} / ${currentDataset}`
                  : currentDataset}
              </span>
            </span>
            <span className="truncate text-xs text-ink-faint">
              {epochStep} · {jobMetricLabel}
              {` · ${presetCountLabel} · ${datasetCountLabel} · ${plannedRunLabel}`}
              {configSnapshotCount > 0
                ? ` · ${configSnapshotCount} snapshots`
                : ""}
              {selectedMonitors.length > 0
                ? ` · ${selectedMonitors.length} monitors`
                : ""}
              {activeSearchLabel ? ` · ${activeSearchLabel}` : ""}
              {" · "}
              {logFolderLabel}
            </span>
          </span>
        </button>
        <div className="flex shrink-0 items-center gap-2">
          {isRunning && (
            <Button
              variant="danger"
              onClick={cancelTraining}
              disabled={isCancelling || !trainingEnabled}
            >
              <CircleStop className="h-4 w-4" aria-hidden />
              Cancel
            </Button>
          )}
          <Button
            variant="secondary"
            onClick={ui.openProgress}
            disabled={
              !trainingEnabled ||
              (!progressRunPlan && !isProgressPlanning && !progressPlanError)
            }
            className="h-10 px-3 text-sm"
          >
            {isProgressPlanning ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <ListChecks className="h-4 w-4" aria-hidden />
            )}
            {progressButtonLabel}
          </Button>
          <Button
            variant="primary"
            onClick={startTraining}
            disabled={!canRequestTraining || !canStart}
            className="h-10 px-[22px] text-sm"
          >
            {isStarting ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <Play className="h-4 w-4" aria-hidden />
            )}
            Start Training
          </Button>
        </div>
      </div>

      {ui.isExpanded && (
        <div
          id="training-panel-details"
          className="grid max-h-[46vh] gap-3 overflow-y-auto border-t border-line bg-bg-2/90 px-4 py-3 sm:px-5 lg:grid-cols-[minmax(0,1fr)_minmax(280px,360px)]"
        >
          {!trainingEnabled && (
            <InlineStatus tone="warning" compact className="lg:col-span-2">
              Training is disabled by this backend.
            </InlineStatus>
          )}
          {historicalTrainingLockMessage && (
            <InlineStatus tone="danger" compact className="lg:col-span-2">
              {historicalTrainingLockMessage}
            </InlineStatus>
          )}
          <div className="grid gap-3">
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              <div className="grid gap-2">
                {logFolderMode === "existing" ? (
                  <TrainingFooterField
                    label={
                      <span className={footerFieldLabelClass}>
                        <FolderOpen className={footerIconClass} aria-hidden />
                        Existing folder
                      </span>
                    }
                    actions={logFolderModeControl}
                  >
                    <Select
                      value={selectedExistingLogFolder}
                      onChange={(event) =>
                        setSelectedExistingLogFolder(event.target.value)
                      }
                      disabled={
                        logFoldersLoading || logFolderOptions.length === 0
                      }
                      aria-label="Log experiment folder"
                    >
                      <option value="">Select folder</option>
                      {logFolderOptions.map((option) => (
                        <option
                          key={option.experiment}
                          value={option.experiment}
                        >
                          {option.experiment} ({option.runCount} runs)
                        </option>
                      ))}
                    </Select>
                    <span className="text-xs text-ink-faint">
                      {existingHelp}
                    </span>
                  </TrainingFooterField>
                ) : (
                  <TrainingFooterField
                    label={
                      <span className={footerFieldLabelClass}>
                        <FolderPlus className={footerIconClass} aria-hidden />
                        New folder
                      </span>
                    }
                    actions={logFolderModeControl}
                  >
                    <Input
                      value={newLogFolder}
                      onChange={(event) => setNewLogFolder(event.target.value)}
                      placeholder="my_experiment"
                      aria-label="New log folder"
                      aria-invalid={
                        newLogFolder.length > 0 && !newLogFolderValid
                      }
                      autoComplete="off"
                    />
                    <span
                      className={
                        newLogFolderError && newLogFolder.length > 0
                          ? "text-xs text-danger-text"
                          : "text-xs text-ink-faint"
                      }
                      role={
                        newLogFolderError && newLogFolder.length > 0
                          ? "alert"
                          : undefined
                      }
                    >
                      {newLogFolderError || "Folder name is valid"}
                    </span>
                  </TrainingFooterField>
                )}
              </div>

              <TrainingTargetDatasetPanel
                modelOptions={modelOptions}
                presetOptions={presetOptions}
                selectedModel={selectedModel}
                selectedPreset={selectedPreset}
                selectedTrainingPresets={selectedTrainingPresets}
                datasetOptions={datasetOptions}
                selectedDatasets={selectedDatasets}
                onSelectModel={onSelectModel}
                onSelectPreset={onSelectPreset}
                onSetTrainingPresets={onSetTrainingPresets}
                onToggleTrainingPreset={onToggleTrainingPreset}
                onMakeTrainingPresetPrimary={onMakeTrainingPresetPrimary}
                onSelectAllTrainingPresets={onSelectAllTrainingPresets}
                onSelectPrimaryTrainingPreset={onSelectPrimaryTrainingPreset}
                onSetDatasets={onSetDatasets}
                onToggleDataset={onToggleDataset}
                onSelectAllDatasets={onSelectAllDatasets}
                onSelectFirstDataset={onSelectFirstDataset}
                presentation="footer"
              />

              <TrainingFooterField
                icon={<Activity className={footerIconClass} aria-hidden />}
                label="Monitors"
                detail={<StatChip>{monitorCount}</StatChip>}
              >
                <MultiSelectDropdown
                  label="Training monitors"
                  values={selectedMonitors}
                  options={trainingMonitorOptions}
                  onChange={changeMonitors}
                  placeholder="Select monitors"
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
              </TrainingFooterField>

              <TrainingFooterField
                icon={
                  <SlidersHorizontal className={footerIconClass} aria-hidden />
                }
                label="Overrides"
                detail={
                  <>
                    <Badge>{fieldCount} fields</Badge>
                    <Badge
                      className={
                        overrideCount > 0
                          ? "border-violet/30 bg-violet/15 text-violet"
                          : undefined
                      }
                    >
                      {overrideCount} overrides
                    </Badge>
                    {configSnapshotCount > 0 && (
                      <Badge className="border-ok/30 bg-ok/10 text-ok">
                        {configSnapshotCount} snapshots
                      </Badge>
                    )}
                  </>
                }
                actions={
                  <Button
                    variant="ghost"
                    onClick={onResetOverrides}
                    disabled={overrideCount === 0}
                    className="h-8 border border-line bg-white/[0.025] px-2.5 text-xs"
                  >
                    <RotateCcw className="h-3.5 w-3.5" aria-hidden />
                    Reset
                  </Button>
                }
              >
                <Button
                  variant="primary"
                  aria-label="Open Full Config"
                  onClick={onOpenFullConfig}
                  disabled={!canOpenFullConfig}
                  className="h-10 w-full text-[13.5px]"
                >
                  <Maximize2 className="h-4 w-4" aria-hidden />
                  Config
                </Button>
                {configSections.length === 0 && (
                  <InlineStatus compact>
                    Select a model and preset to load config fields
                  </InlineStatus>
                )}
              </TrainingFooterField>
            </div>

            <TrainingFooterField label={null}>
              <TrainingSearchSetup
                axes={searchAxes}
                search={effectiveTrainingSearch}
                overrides={overrides}
                selectedDatasetCount={selectedDatasets.length}
                selectedPresetCount={selectedTrainingPresetCount}
                isLoading={searchLoading}
                disabledReason={
                  hasConfigSnapshots
                    ? "Config snapshots train fixed variants; grid and random search are unavailable."
                    : undefined
                }
                onChange={onTrainingSearchChange}
              />
            </TrainingFooterField>
          </div>

          <aside className="grid gap-3">
            <div className="edge grid gap-2 rounded-card p-3">
              <SectionHeading title="Request" />
              <div className="grid gap-1.5 text-xs text-ink-dim">
                <div className="truncate font-mono">
                  {selectedModel || "No model"}
                </div>
                <div className="truncate font-mono">
                  {selectedPreset || "No preset"}
                </div>
                <div>{presetCountLabel}</div>
                <div>{datasetCountLabel}</div>
                <div>{selectedMonitors.length} monitors</div>
                {hasConfigSnapshots && (
                  <div>{configSnapshotCount} config snapshots</div>
                )}
                <div>{selectedFieldSummary.length} effective overrides</div>
                <div>{plannedRunLabel}</div>
                {effectiveTrainingSearch.mode !== "off" && (
                  <>
                    <div>
                      {searchModeLabel} search
                    </div>
                    <div>
                      {activeSearchAxisCount} axes · {displayedRunCount} planned
                      runs
                    </div>
                    {effectiveTrainingSearch.mode === "random" && (
                      <div>{effectiveTrainingSearch.randomSamples} random samples</div>
                    )}
                  </>
                )}
                <div className="truncate font-mono">{logFolderLabel}</div>
              </div>
              {searchConflictKeys.length > 0 && (
                <div className="rounded-[8px] border border-amber/30 bg-amber/[0.055] px-2 py-1 text-xs text-amber">
                  {searchConflictKeys.length} override
                  {searchConflictKeys.length === 1 ? "" : "s"} replaced by
                  search.
                </div>
              )}
              {effectiveTrainingSearch.mode !== "off" &&
                !trainingSearchValidation.ready && (
                  <div className="rounded-[8px] border border-danger-line bg-danger-soft px-2 py-1 text-xs text-danger-text">
                    {trainingSearchValidation.message}
                  </div>
                )}
              {selectedFieldSummary.length > 0 && (
                <div className="grid max-h-24 gap-1 overflow-y-auto pr-1">
                  {selectedFieldSummary.map((entry) => (
                    <div
                      key={entry.key}
                      className="flex min-w-0 items-center justify-between gap-2 rounded-[8px] border border-violet/30 bg-violet/10 px-2 py-1 text-xs"
                    >
                      <span className="truncate text-ink">{entry.label}</span>
                      <span className="max-w-[9rem] truncate font-mono text-violet">
                        {entry.value}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {clusterGrowth.length > 0 && (
              <div className="edge grid gap-2 rounded-card p-3">
                <SectionHeading title="Cluster growth" />
                <div className="grid gap-1.5">
                  {clusterGrowth.map((summary) => (
                    <div
                      key={summary.node}
                      className="grid gap-1.5 rounded-[10px] border border-line bg-white/[0.018] px-2.5 py-2 text-xs"
                    >
                      <div className="flex min-w-0 items-center justify-between gap-2">
                        <span className="truncate font-mono text-ink">
                          {summary.node}
                        </span>
                        <span className="font-mono text-violet-muted">
                          {summary.count}
                          {summary.capacityTotal
                            ? ` / ${summary.capacityTotal}`
                            : ""}{" "}
                          neurons
                          {summary.additions.length > 0
                            ? ` · +${summary.additions.length}`
                            : ""}
                        </span>
                      </div>
                      {summary.additions.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {summary.additions
                            .slice(-12)
                            .map((addition, index) => (
                              <span
                                key={`${addition.coord.join("-")}-${index}`}
                                title={`added at ${addition.step !== null ? `step ${addition.step}` : "unknown step"}`}
                                className="rounded-[6px] border border-violet/30 bg-violet/10 px-1.5 py-0.5 font-mono text-[11px] text-violet"
                              >
                                ({addition.coord.join(", ")})
                                {addition.step !== null
                                  ? ` @${addition.step}`
                                  : ""}
                              </span>
                            ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="edge grid gap-2 rounded-card p-3">
              <SectionHeading title="Log Tail" />
              <pre className="max-h-36 overflow-y-auto whitespace-pre-wrap rounded-[10px] border border-line bg-black/25 p-2 font-mono text-xs leading-5 text-ink-dim">
                {(job?.logTail.length
                  ? job.logTail
                  : ["No log output yet"]
                ).join("\n")}
              </pre>
            </div>

            {trainingError && (
              <div className="rounded-[10px] border border-danger-line bg-danger-soft p-3 text-sm text-danger-text">
                {trainingError}
              </div>
            )}
          </aside>
        </div>
      )}
      {ui.isProgressOpen && (
        <TrainingProgressDialog
          plan={progressRunPlan}
          isLoading={isProgressPlanning}
          error={progressPlanError}
          canResample={canResampleRunPlan}
          isResampling={isResampling}
          onResample={resampleRunPlan}
          canRemoveSnapshots={hasConfigSnapshots && !job}
          onRemoveSnapshot={onRemoveConfigSnapshot}
          onClose={ui.closeProgress}
        />
      )}
      {showLargeGridConfirmation && (
        <DialogShell
          titleId="large-grid-search-title"
          size="sm"
          onClose={cancelLargeGridSearch}
          closeOnEscape={!isStarting}
          className="grid place-items-center bg-black/65 p-4 sm:p-4"
          panelClassName="grid max-h-none max-w-md gap-4 overflow-visible p-4 shadow-[0_24px_80px_-30px_rgba(0,0,0,0.9)] sm:max-h-none"
          header={
            <div className="grid gap-1">
              <h2
                id="large-grid-search-title"
                className="text-base font-semibold text-ink"
              >
                Confirm Grid Search
              </h2>
              <p className="text-sm leading-6 text-ink-dim">
                This grid search plans {displayedRunCount} training runs across{" "}
                {selectedTrainingPresetCount} presets and{" "}
                {selectedDatasets.length} datasets.
              </p>
            </div>
          }
        >
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={cancelLargeGridSearch}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={confirmLargeGridSearch}
              disabled={isStarting}
            >
              {isStarting && (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              )}
              Start Training
            </Button>
          </div>
        </DialogShell>
      )}
    </section>
  );
}
