import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelExperimentsPanel } from "@/features/viewer/components/model-experiments-panel";
import { type LogRun } from "@/lib/api";
import { type HistoricalParameterSummaryState } from "@/lib/parameter-summary";

const mocks = vi.hoisted(() => ({
  useHistoricalRuns: vi.fn(),
}));

vi.mock("@/features/viewer/providers/viewer-providers", () => ({
  useHistoricalRuns: mocks.useHistoricalRuns,
}));

function run(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? overrides.experiment ?? "exp_a",
    experiment: overrides.experiment ?? "exp_a",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    dataset: overrides.dataset ?? "Mnist",
    runName: overrides.runName ?? `${overrides.id}_20260601_010203`,
    timestamp: overrides.timestamp ?? "2026-06-01 01:02:03",
    version: overrides.version ?? "version_0",
    relativePath: overrides.relativePath ?? "exp_a/linear/baseline/Mnist/run/version_0",
    hasResult: overrides.hasResult ?? true,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function summaryState(
  overrides: Partial<HistoricalParameterSummaryState> = {},
): HistoricalParameterSummaryState {
  return {
    summary:
      overrides.summary ?? {
        counts: {
          updated: 6,
          unchanged: 1,
          mixed: 1,
          notTracked: 0,
        },
        breakdown: {
          missing: 0,
          unknown: 0,
        },
        total: 8,
        severity: "danger",
      },
    isLoading: overrides.isLoading ?? false,
    isError: overrides.isError ?? false,
    error: overrides.error,
  };
}

function renderPanel({
  selectedRunId = "run-1",
  summaries = new Map<string, HistoricalParameterSummaryState>([
    ["run-1", summaryState()],
  ]),
}: {
  selectedRunId?: string | null;
  summaries?: Map<string, HistoricalParameterSummaryState>;
} = {}) {
  mocks.useHistoricalRuns.mockReturnValue({
    visibleHistoricalRuns: [run({ id: "run-1" })],
    historicalPresetOptions: [{ value: "baseline", label: "baseline", count: 1 }],
    selectedHistoricalPreset: "",
    setSelectedHistoricalPreset: vi.fn(),
    selectedLogRunId: selectedRunId,
    selectLogRun: vi.fn(),
    historicalParameterSummariesByRunId: summaries,
    experimentsLoading: false,
    experimentsError: null,
  });

  render(<ModelExperimentsPanel />);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("ModelExperimentsPanel", () => {
  it("renders parameter counters with accessible labels, focus tooltips, and severity classes", () => {
    renderPanel();

    const card = screen.getByTestId("experiment-run-card-run-1");
    expect(card).toHaveClass("border-danger-line", "bg-danger-soft", "ring-2");

    expect(
      screen.getByRole("button", { name: /updated parameters: 6 of 8/i }),
    ).toBeInTheDocument();
    const unchanged = screen.getByRole("button", {
      name: /unchanged parameters: 1 of 8/i,
    });
    expect(
      screen.getByRole("button", { name: /mixed parameters: 1 of 8/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /not tracked parameters: 0 of 8/i }),
    ).toBeInTheDocument();

    fireEvent.focus(unchanged);

    expect(screen.getByRole("tooltip")).toHaveTextContent("Unchanged: 1 of 8");
    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "6 updated, 1 unchanged, 1 mixed, 0 not tracked (0 missing, 0 unknown)",
    );
  });

  it("keeps normal card styling and unavailable counters when summary loading fails", () => {
    renderPanel({
      selectedRunId: null,
      summaries: new Map<string, HistoricalParameterSummaryState>([
        [
          "run-1",
          {
            isLoading: false,
            isError: true,
            error: new Error("summary failed"),
          },
        ],
      ]),
    });

    const card = screen.getByTestId("experiment-run-card-run-1");
    expect(card).toHaveClass("border-line-soft", "bg-black/18");
    expect(card).not.toHaveClass("border-danger-line");

    expect(
      screen.getByRole("button", { name: /updated parameters: summary unavailable/i }),
    ).toHaveTextContent("-");
  });
});
