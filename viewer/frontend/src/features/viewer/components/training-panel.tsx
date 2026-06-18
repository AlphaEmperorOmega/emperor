import {
  ChevronDown,
  ChevronUp,
  CircleStop,
  FolderOpen,
  FolderPlus,
  Loader2,
  Play,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { TrainingSearchSetup } from "@/features/viewer/components/training-search-setup";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { TrainingTargetDatasetPanel } from "@/features/viewer/components/training/training-target-dataset-panel";
import { TrainingCompactRunList } from "@/features/viewer/components/training/training-compact-run-list";
import { TrainingFooterRunSummary } from "@/features/viewer/components/training/training-footer-run-summary";
import { TrainingLogTailCard } from "@/features/viewer/components/training/training-log-tail-card";
import { TrainingRunPlanCard } from "@/features/viewer/components/training/training-run-plan-card";
import {
  type TrainingPanelViewModel,
} from "@/features/viewer/state/training/use-training-panel-view-model";

type TrainingPanelProps = {
  viewModel: TrainingPanelViewModel;
};

const footerIconClass = "h-[15px] w-[15px] text-violet";

export function TrainingPanel({ viewModel }: TrainingPanelProps) {
  const { input, logFolder, options, request, status, training, ui } =
    viewModel;
  const {
    datasetOptions,
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedTrainingPresets,
    selectedTrainingSnapshotIds,
    selectedDatasets,
    overrides,
    allConfigSnapshots,
    configSnapshotCount,
    monitorOptions,
    selectedMonitors,
    monitorsLoading,
    searchAxes,
    searchLoading,
    trainingEnabled,
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
    onRemoveConfigSnapshot,
    onExcludeConfigSnapshot,
    onExcludeDraftTrainingPreset,
    onCreatePresetSnapshot,
    onEditConfigSnapshot,
    onDuplicateConfigSnapshot,
    onTrainingSearchChange,
  } = input;
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
  const { modelTypeOptions, modelOptions, presetOptions } = options;
  const {
    activeConfigSnapshotCount,
    activeSearchAxisCount,
    canRequestTraining,
    effectiveTrainingSearch,
    searchConflictKeys,
    searchModeLabel,
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
    requiresLargeGridConfirmation,
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
    plannedRunLabel,
    presetCountLabel,
  } = status;
  const planChangingControlsDisabled = isRunning;
  const setupLockMessage = planChangingControlsDisabled
    ? "Training setup is locked while the active job is running or queued."
    : "";

  const logFolderModeControl = (
    <SegmentedControl aria-label="Log folder mode">
      <ViewModeButton
        active={logFolder.mode === "existing"}
        disabled={planChangingControlsDisabled}
        onClick={() => logFolder.setMode("existing")}
      >
        <FolderOpen className="h-3.5 w-3.5" aria-hidden />
        Existing folder
      </ViewModeButton>
      <ViewModeButton
        active={logFolder.mode === "new"}
        disabled={planChangingControlsDisabled}
        onClick={() => logFolder.setMode("new")}
      >
        <FolderPlus className="h-3.5 w-3.5" aria-hidden />
        New folder
      </ViewModeButton>
    </SegmentedControl>
  );
  const logFolderDropdownOptions = logFolderOptions.map((option) => ({
    value: option.experiment,
    label: `${option.experiment} (${option.runCount} runs)`,
  }));

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
          <TrainingFooterRunSummary
            plan={progressRunPlan}
            job={job}
            isLoading={isProgressPlanning}
            error={progressPlanError}
          />
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
          className="grid max-h-[52vh] min-h-[22rem] grid-rows-[auto_minmax(0,1fr)] gap-3 overflow-x-auto overflow-y-hidden border-t border-line bg-bg-2/90 px-4 py-3 sm:px-5"
        >
          {!trainingEnabled && (
            <InlineStatus tone="warning" compact>
              Training is disabled by this backend.
            </InlineStatus>
          )}
          {historicalTrainingLockMessage && (
            <InlineStatus tone="danger" compact>
              {historicalTrainingLockMessage}
            </InlineStatus>
          )}
          <div className="grid min-h-0 min-w-[920px] gap-3 grid-cols-[minmax(300px,340px)_minmax(22rem,1fr)_minmax(280px,360px)]">
            <aside
              aria-label="Training Setup Sidebar"
              className="grid min-h-0 content-start gap-4 overflow-y-auto pr-1"
            >
              {setupLockMessage && (
                <InlineStatus tone="warning" compact>
                  {setupLockMessage}
                </InlineStatus>
              )}

              <section className="grid gap-2 border-b border-line-soft pb-3">
                <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
                  <SectionHeading
                    icon={
                      logFolderMode === "existing" ? (
                        <FolderOpen className={footerIconClass} aria-hidden />
                      ) : (
                        <FolderPlus className={footerIconClass} aria-hidden />
                      )
                    }
                    title="Log Folder"
                  />
                  {logFolderModeControl}
                </div>
                <div className="grid min-h-[4rem] grid-rows-[2.5rem_1rem] gap-2">
                  {logFolderMode === "existing" ? (
                    <>
                      <SelectOnlyDropdown
                        label="Log experiment folder"
                        value={selectedExistingLogFolder}
                        options={logFolderDropdownOptions}
                        onChange={setSelectedExistingLogFolder}
                        disabled={
                          planChangingControlsDisabled ||
                          logFoldersLoading ||
                          logFolderDropdownOptions.length === 0
                        }
                        placeholder="Select folder"
                      />
                      <span className="min-h-4 text-xs leading-4 text-ink-faint">
                        {existingHelp}
                      </span>
                    </>
                  ) : (
                    <>
                      <Input
                        value={newLogFolder}
                        onChange={(event) => setNewLogFolder(event.target.value)}
                        placeholder="my_experiment"
                        aria-label="New log folder"
                        aria-invalid={
                          newLogFolder.length > 0 && !newLogFolderValid
                        }
                        autoComplete="off"
                        disabled={planChangingControlsDisabled}
                        className="h-10"
                      />
                      <span
                        className={
                          newLogFolderError && newLogFolder.length > 0
                            ? "min-h-4 text-xs leading-4 text-danger-text"
                            : "min-h-4 text-xs leading-4 text-ink-faint"
                        }
                        role={
                          newLogFolderError && newLogFolder.length > 0
                            ? "alert"
                            : undefined
                        }
                      >
                        {newLogFolderError || "Folder name is valid"}
                      </span>
                    </>
                  )}
                </div>
              </section>

              <TrainingTargetDatasetPanel
                modelTypeOptions={modelTypeOptions}
                modelOptions={modelOptions}
                selectedModelType={selectedModelType}
                presetOptions={presetOptions}
                selectedModel={selectedModel}
                selectedPreset={selectedPreset}
                selectedTrainingPresets={selectedTrainingPresets}
                configSnapshots={allConfigSnapshots}
                selectedTrainingSnapshotIds={selectedTrainingSnapshotIds}
                datasetOptions={datasetOptions}
                selectedDatasets={selectedDatasets}
                onSelectModelType={onSelectModelType}
                onSelectModel={onSelectModel}
                onSelectPreset={onSelectPreset}
                onSetTrainingPresets={onSetTrainingPresets}
                onSetTrainingSnapshotSelection={onSetTrainingSnapshotSelection}
                onToggleTrainingPreset={onToggleTrainingPreset}
                onMakeTrainingPresetPrimary={onMakeTrainingPresetPrimary}
                onSelectAllTrainingPresets={onSelectAllTrainingPresets}
                onSelectPrimaryTrainingPreset={onSelectPrimaryTrainingPreset}
                onSetDatasets={onSetDatasets}
                onToggleDataset={onToggleDataset}
                onSelectAllDatasets={onSelectAllDatasets}
                onSelectFirstDataset={onSelectFirstDataset}
                monitorOptions={monitorOptions}
                selectedMonitors={selectedMonitors}
                monitorsLoading={monitorsLoading}
                onSetMonitors={onSetMonitors}
                onSelectAllMonitors={onSelectAllMonitors}
                onClearMonitors={onClearMonitors}
                onCreatePresetSnapshot={onCreatePresetSnapshot}
                onEditConfigSnapshot={onEditConfigSnapshot}
                onDuplicateConfigSnapshot={onDuplicateConfigSnapshot}
                onDeleteConfigSnapshot={onRemoveConfigSnapshot}
                disabled={planChangingControlsDisabled}
                presentation="setup"
              />

              <section className="grid gap-2 border-t border-line-soft pt-3">
                <TrainingSearchSetup
                  axes={searchAxes}
                  search={effectiveTrainingSearch}
                  overrides={overrides}
                  selectedDatasetCount={selectedDatasets.length}
                  selectedPresetCount={selectedTrainingPresetCount}
                  isLoading={searchLoading}
                  disabledReason={
                    setupLockMessage ||
                    (activeConfigSnapshotCount > 0
                      ? "Config snapshots train fixed variants; grid and random search are unavailable."
                      : undefined)
                  }
                  onChange={onTrainingSearchChange}
                />
              </section>
            </aside>

            <main
              aria-label="Training Run List"
              className="grid min-h-0"
            >
              <TrainingCompactRunList
                plan={progressRunPlan}
                isLoading={isProgressPlanning}
                error={progressPlanError}
                canResample={canResampleRunPlan}
                isResampling={isResampling}
                onResample={resampleRunPlan}
                canManageDraftRuns={!job}
                onExcludePreset={onExcludeDraftTrainingPreset}
                onExcludeSnapshot={onExcludeConfigSnapshot}
              />
            </main>

            <aside
              aria-label="Training Status Sidebar"
              aria-live="polite"
              className="grid min-h-0 content-start gap-3 overflow-y-auto pr-1"
            >
              <TrainingRunPlanCard
                plan={progressRunPlan}
                job={job}
                isPlanning={isProgressPlanning}
                planError={progressPlanError}
                trainingError={trainingError}
                effectiveTrainingSearch={effectiveTrainingSearch}
                searchModeLabel={searchModeLabel}
                activeSearchAxisCount={activeSearchAxisCount}
                searchConflictCount={searchConflictKeys.length}
                trainingSearchValidation={trainingSearchValidation}
                displayedRunCount={displayedRunCount}
                requiresLargeGridConfirmation={requiresLargeGridConfirmation}
                selectedMonitorCount={selectedMonitors.length}
                presetCountLabel={presetCountLabel}
                datasetCountLabel={datasetCountLabel}
              />

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
                            {summary.additionCount > 0
                              ? ` · +${summary.additionCount}`
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
                                  title={`added at ${
                                    addition.step !== null
                                      ? `step ${addition.step}`
                                      : "unknown step"
                                  }`}
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

              <TrainingLogTailCard logTail={job?.logTail} />
            </aside>
          </div>
        </div>
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
