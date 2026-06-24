import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LogBestRunPanel } from "@/features/viewer/components/logs/log-best-run-panel";
import { type LogBestRunViewModel } from "@/features/viewer/state/logs/logs-chart-view-model";
import {
  type LogMetricDatasetRankingRow,
  type LogMetricRankingRow,
} from "@/features/viewer/state/logs/log-metric-ranking";
import { type LogRun } from "@/lib/api";

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  const experiment = overrides.experiment ?? "experiment";
  const dataset = overrides.dataset ?? "Mnist";
  const modelType = overrides.modelType ?? "linears";
  const model = overrides.model ?? "linear";
  const preset = overrides.preset ?? "BASELINE";

  return {
    id: overrides.id,
    group: overrides.group ?? null,
    experiment,
    modelType,
    model,
    preset,
    dataset,
    runName: overrides.runName ?? overrides.id,
    timestamp: overrides.timestamp ?? "2026-06-01 01:02:03",
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ??
      `${experiment}/${modelType}/${model}/${preset}/${dataset}/${overrides.id}/version_0`,
    hasResult: overrides.hasResult ?? true,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function rankingRow(overrides: Partial<LogMetricRankingRow>): LogMetricRankingRow {
  const run = overrides.run ?? logRun({ id: overrides.runId ?? "run-a" });
  return {
    runId: run.id,
    run,
    tag: overrides.tag ?? "validation/accuracy",
    score: overrides.score ?? 0.84,
    step: overrides.step ?? 2,
    wallTime: overrides.wallTime ?? 1780000001,
    visibleIndex: overrides.visibleIndex ?? 0,
    sourceIndex: overrides.sourceIndex ?? 0,
    pointIndex: overrides.pointIndex ?? 1,
    ...overrides,
  };
}

function datasetRankingRow({
  best = rankingRow({}),
  dataset,
  runCount,
  visibleIndex,
}: Partial<LogMetricDatasetRankingRow> &
  Pick<LogMetricDatasetRankingRow, "dataset">): LogMetricDatasetRankingRow {
  return {
    dataset,
    runCount: runCount ?? 1,
    visibleIndex: visibleIndex ?? 0,
    best,
  };
}

function bestRunViewModel(
  overrides: Partial<LogBestRunViewModel> = {},
): LogBestRunViewModel {
  return {
    metricTagOptions: [
      { value: "validation/accuracy", label: "validation/accuracy", count: 2 },
      { value: "test/accuracy", label: "test/accuracy", count: 2 },
    ],
    selectedMetricTag: "validation/accuracy",
    selectedDirection: "higher",
    selectedPointPolicy: "best",
    rows: [datasetRankingRow({ dataset: "Mnist" })],
    visibleRunCount: 2,
    hasMoreRuns: false,
    isLoading: false,
    isFetching: false,
    isError: false,
    error: null,
    onMetricTagChange: vi.fn(),
    onDirectionChange: vi.fn(),
    onPointPolicyChange: vi.fn(),
    ...overrides,
  };
}

describe("LogBestRunPanel", () => {
  it("renders per-dataset best rows, controls, scoped copy, and Details action", async () => {
    const user = userEvent.setup();
    const onSelectRun = vi.fn();
    const onMetricTagChange = vi.fn();
    const onDirectionChange = vi.fn();
    const onPointPolicyChange = vi.fn();
    render(
      <LogBestRunPanel
        bestRun={bestRunViewModel({
          hasMoreRuns: true,
          onMetricTagChange,
          onDirectionChange,
          onPointPolicyChange,
          rows: [
            datasetRankingRow({ dataset: "Mnist" }),
            datasetRankingRow({
              dataset: "Cifar10",
              visibleIndex: 1,
              best: null,
            }),
          ],
        })}
        onSelectRun={onSelectRun}
      />,
    );

    const panel = screen.getByRole("heading", { name: "Best Run" }).closest("section");
    expect(panel).toBeInstanceOf(HTMLElement);
    const section = panel as HTMLElement;
    expect(within(section).getByText(/best run per visible dataset/i))
      .toBeInTheDocument();
    expect(within(section).getByText(/exclude unloaded runs/i)).toBeInTheDocument();
    expect(within(section).queryByRole("combobox", { name: /best run dataset/i }))
      .not.toBeInTheDocument();
    expect(within(section).queryByRole("article")).not.toBeInTheDocument();

    const table = within(section).getByRole("table", {
      name: /validation\/accuracy best run leaderboard/i,
    });
    expect(within(table).getByRole("columnheader", { name: "Dataset" }))
      .toBeInTheDocument();
    expect(within(table).queryByRole("columnheader", { name: "Rank" }))
      .not.toBeInTheDocument();
    expect(within(table).getByText("Mnist")).toBeInTheDocument();
    expect(within(table).getByText("Cifar10")).toBeInTheDocument();
    expect(within(table).getByText("No points")).toBeInTheDocument();
    expect(within(table).getByText("0.84")).toBeInTheDocument();
    expect(within(table).getByText("run-a")).toBeInTheDocument();
    expect(within(table).getByRole("button", { name: /no details for cifar10/i }))
      .toBeDisabled();

    await user.selectOptions(
      within(section).getByRole("combobox", { name: /best run metric/i }),
      "test/accuracy",
    );
    await user.click(within(section).getByRole("radio", { name: "Latest" }));
    await user.click(within(section).getByRole("radio", { name: "Lower" }));
    await user.click(within(section).getAllByRole("button", { name: /open details/i })[0]);

    expect(onMetricTagChange).toHaveBeenCalledWith("test/accuracy");
    expect(onPointPolicyChange).toHaveBeenCalledWith("latest");
    expect(onDirectionChange).toHaveBeenCalledWith("lower");
    expect(onSelectRun).toHaveBeenCalledWith("run-a");
  });

  it("renders empty, loading, and error states", () => {
    const onSelectRun = vi.fn();
    const { rerender } = render(
      <LogBestRunPanel
        bestRun={bestRunViewModel({ visibleRunCount: 0, rows: [] })}
        onSelectRun={onSelectRun}
      />,
    );

    expect(screen.getByText("No visible runs to rank.")).toBeInTheDocument();

    rerender(
      <LogBestRunPanel
        bestRun={bestRunViewModel({
          metricTagOptions: [],
          selectedMetricTag: null,
          rows: [],
        })}
        onSelectRun={onSelectRun}
      />,
    );
    expect(screen.getByText("No scalar metric tags found for the visible runs."))
      .toBeInTheDocument();

    rerender(
      <LogBestRunPanel
        bestRun={bestRunViewModel({ isLoading: true, rows: [] })}
        onSelectRun={onSelectRun}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      "Loading best run scalar points",
    );

    rerender(
      <LogBestRunPanel
        bestRun={bestRunViewModel({
          rows: [datasetRankingRow({ dataset: "Mnist", best: null })],
        })}
        onSelectRun={onSelectRun}
      />,
    );
    expect(screen.getByText("No selected metric points for any visible dataset."))
      .toBeInTheDocument();

    rerender(
      <LogBestRunPanel
        bestRun={bestRunViewModel({
          isError: true,
          error: new Error("scalar read failed"),
          rows: [],
        })}
        onSelectRun={onSelectRun}
      />,
    );
    expect(screen.getByText("Best Run scalar read failed")).toBeInTheDocument();
    expect(screen.getByText("scalar read failed")).toBeInTheDocument();
  });
});
