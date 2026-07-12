import {
  Activity,
  CircleStop,
  FolderOpen,
  FolderPlus,
  Loader2,
  Maximize2,
  Play,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { TrainingSearchSetup } from "@/features/workbench/components/training-search-setup";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";
import { SelectOnlyDropdown } from "@/features/workbench/components/screen/select-only-dropdown";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { SectionHeading } from "@/components/ui/section-heading";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { TrainingTargetDatasetPanel } from "@/features/workbench/components/training/training-target-dataset-panel";
import {
  TrainingAllCommandsButton,
  TrainingCompactRunList,
} from "@/features/workbench/components/training/training-compact-run-list";
import { TrainingRunSummaryBadge } from "@/features/workbench/components/training/training-footer-run-summary";
import { TrainingLogTailCard } from "@/features/workbench/components/training/training-log-tail-card";
import { TrainingRunPlanCard } from "@/features/workbench/components/training/training-run-plan-card";
import { WorkbenchWideThreeRegionLayout } from "@/features/workbench/components/_workbench-wide-three-region-layout";
import { useTrainingWorkspace } from "@/features/workbench/providers/training-provider";

const trainingIconClass = "h-[15px] w-[15px] text-violet";

export function TrainingPanel() {
  const { draft, plan, job: activeJob, dialogs, actions } =
    useTrainingWorkspace();
  const { setup, logFolder } = draft;
  const { variants } = setup;
  const selectedDatasets = setup.datasets.selected;
  const selectedMonitors = setup.monitors.selected;
  const selectedTrainingPresetCount = draft.status.selectedPresetCount;
  const trainingEnabled = draft.status.trainingEnabled;
  const canOpenFullConfig = draft.status.canOpenFullConfig;
  const {
    display: progressRunPlan,
    displayedRunCount,
    isPlanning: isProgressPlanning,
    error: progressPlanError,
    canStart,
    canResample: canResampleRunPlan,
    canRetry: canRetryRunPlan,
    isResampling,
    datasetCountLabel,
    presetCountLabel,
    search: {
      effective: effectiveTrainingSearch,
      conflictKeys: searchConflictKeys,
      validation: trainingSearchValidation,
      lockSummary: searchLockSummary,
      modeLabel: searchModeLabel,
      activeAxisCount: activeSearchAxisCount,
    },
  } = plan;
  const {
    value: job,
    status: jobStatus,
    isRunning,
    canReset: canResetTraining,
    isStarting,
    isCancelling,
    error: trainingError,
    clusterGrowth,
  } = activeJob;
  const {
    isOpen: showLargeGridConfirmation,
    isRequired: requiresLargeGridConfirmation,
  } = dialogs.largeGridConfirmation;
  const {
    openFullConfig: onOpenFullConfig,
    startJob: startTraining,
    confirmLargeGridSearch,
    cancelLargeGridSearch,
    cancelJob: cancelTraining,
    resetJob: resetTraining,
    resamplePlan: resampleRunPlan,
    retryPlan: retryRunPlan,
  } = actions;
  const onExcludeDraftTrainingPreset = variants.excludePreset;
  const onExcludeConfigSnapshot = variants.excludeSnapshot;
  const {
    state: {
    existingHelp,
    existingValue: selectedExistingLogFolder,
    isLoading: logFoldersLoading,
    mode: logFolderMode,
    newError: newLogFolderError,
    newValid: newLogFolderValid,
    newValue: newLogFolder,
    options: logFolderOptions,
    },
    actions: {
      selectMode: selectLogFolderMode,
      selectExisting: setSelectedExistingLogFolder,
      nameNew: setNewLogFolder,
    },
  } = logFolder;
  const planChangingControlsDisabled = isRunning;
  const setupLockMessage = planChangingControlsDisabled
    ? "Training setup is locked while the active job is running or queued."
    : "";

  const logFolderModeControl = (
    <SegmentedControl aria-label="Log folder mode">
      <ViewModeButton
        active={logFolderMode === "existing"}
        disabled={planChangingControlsDisabled}
        onClick={() => selectLogFolderMode("existing")}
      >
        <FolderOpen className="h-3.5 w-3.5" aria-hidden />
        Existing folder
      </ViewModeButton>
      <ViewModeButton
        active={logFolderMode === "new"}
        disabled={planChangingControlsDisabled}
        onClick={() => selectLogFolderMode("new")}
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
    <section
      id="training-workspace"
      aria-label="Training workspace"
      className="h-full min-w-0 overflow-hidden bg-[linear-gradient(180deg,rgba(13,12,22,0.72),rgba(8,8,14,0.88))]"
    >
      <WorkbenchWideThreeRegionLayout
        leadingLabel="Training Setup Sidebar"
        primaryLabel="Training Run List"
        trailingLabel="Training Status Sidebar"
        notices={
          <>
          {trainingError && (
            <InlineStatus tone="danger" compact role="alert">
              {trainingError}
            </InlineStatus>
          )}
          {!trainingEnabled && (
            <InlineStatus tone="warning" compact>
              Training is disabled by this backend.
            </InlineStatus>
          )}
          </>
        }
        leading={
          <>
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
                        <FolderOpen className={trainingIconClass} aria-hidden />
                      ) : (
                        <FolderPlus className={trainingIconClass} aria-hidden />
                      )
                    }
                    title="Log Folder"
                  />
                  {logFolderModeControl}
                </div>
                <div className="grid min-h-[4rem] grid-rows-[2.5rem_1rem] gap-2">
                  {logFolderMode === "existing" ? (
                    <>
                      <div className="h-10 min-h-10">
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
                          className="h-10 min-h-10"
                          triggerClassName="h-10"
                        />
                      </div>
                      <span className="min-h-4 text-xs leading-4 text-ink-faint">
                        {existingHelp || "\u00a0"}
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
                            ? "block min-h-4 text-xs leading-4 text-danger-text"
                            : "block min-h-4 text-xs leading-4 text-ink-faint"
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
                setup={setup}
                disabled={planChangingControlsDisabled}
              />

              <section className="grid gap-2 border-t border-line-soft pt-3">
                <TrainingSearchSetup search={plan.search} />
              </section>
          </>
        }
        primary={
          <>
              <header className="grid gap-3 border-b border-line px-4 py-3 backdrop-blur-xl sm:px-[22px] xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center">
                <div className="grid min-w-0 gap-1.5">
                  <div className="flex min-w-0 flex-wrap items-center gap-2">
                    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-[9px] border border-violet/35 bg-[linear-gradient(135deg,#2a2740,#16142a)] text-violet">
                      <Activity className="h-4 w-4" aria-hidden />
                    </span>
                    <h1 className="text-base font-bold text-ink">Training</h1>
                    <Badge
                      className={
                        jobStatus === "failed" || jobStatus === "cancelled"
                          ? "border-danger-line bg-danger-soft text-danger-text"
                          : jobStatus === "completed"
                            ? "border-ok/30 bg-ok/10 text-ok"
                            : jobStatus === "running" || jobStatus === "queued"
                              ? "border-amber/40 bg-amber/[0.12] text-amber"
                              : "border-line bg-white/[0.05] text-ink-faint"
                      }
                    >
                      {jobStatus}
                    </Badge>
                  </div>
                </div>
                <div className="flex min-w-0 flex-wrap items-center gap-2 xl:justify-end">
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
                  <TrainingRunSummaryBadge
                    plan={progressRunPlan}
                    job={job}
                    isLoading={isProgressPlanning}
                    error={progressPlanError}
                    className="min-w-0"
                  />
                  <TrainingAllCommandsButton plan={progressRunPlan} />
                  {canRetryRunPlan && (
                    <Button
                      variant="secondary"
                      onClick={retryRunPlan}
                      className="h-10 px-3 text-[13px]"
                    >
                      <RefreshCw className="h-4 w-4" aria-hidden />
                      Retry Plan
                    </Button>
                  )}
                  <Button
                    variant="secondary"
                    onClick={onOpenFullConfig}
                    disabled={planChangingControlsDisabled || !canOpenFullConfig}
                    className="h-10 px-3 text-[13px]"
                  >
                    <Maximize2 className="h-4 w-4" aria-hidden />
                    Open Full Config
                  </Button>
                  {canResampleRunPlan && (
                    <Button
                      variant="secondary"
                      onClick={resampleRunPlan}
                      disabled={isResampling}
                      className="h-10 px-3 text-[13px]"
                    >
                      <RefreshCw
                        className={
                          isResampling ? "h-4 w-4 animate-spin" : "h-4 w-4"
                        }
                        aria-hidden
                      />
                      Resample
                    </Button>
                  )}
                  {canResetTraining && (
                    <Button
                      variant="secondary"
                      onClick={resetTraining}
                      disabled={!trainingEnabled}
                      className="h-10 px-3 text-[13px]"
                    >
                      <RotateCcw className="h-4 w-4" aria-hidden />
                      Reset Training
                    </Button>
                  )}
                  <Button
                    variant="primary"
                    onClick={startTraining}
                    disabled={!trainingEnabled || !canStart}
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
              </header>
              <TrainingCompactRunList
                plan={progressRunPlan}
                isLoading={isProgressPlanning}
                error={progressPlanError}
                canManageDraftRuns={!job}
                onExcludePreset={onExcludeDraftTrainingPreset}
                onExcludeSnapshot={onExcludeConfigSnapshot}
              />
          </>
        }
        trailing={
          <>
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
                searchLockSummary={searchLockSummary}
                trainingSearchValidation={trainingSearchValidation}
                displayedRunCount={displayedRunCount}
                requiresLargeGridConfirmation={requiresLargeGridConfirmation}
                selectedMonitorCount={selectedMonitors.length}
                presetCountLabel={presetCountLabel}
                datasetCountLabel={datasetCountLabel}
              />

              {clusterGrowth.length > 0 && (
                <SurfacePanel padding="roomy">
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
                </SurfacePanel>
              )}

              <TrainingLogTailCard logTail={job?.logTail} />
          </>
        }
      />
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
