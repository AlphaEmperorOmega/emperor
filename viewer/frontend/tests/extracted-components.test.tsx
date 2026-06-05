import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FullPageError } from "@/components/layout/page-error-status";
import { FullPageLoading, FullPageStatus } from "@/components/layout/page-status";
import { FeatureListDialog } from "@/components/features/viewer/feature-list-dialog";
import { createViewerContext } from "@/components/features/viewer/providers/create-context";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import { KeyValueRow } from "@/components/features/viewer/shared/key-value-row";
import { MetricCard } from "@/components/features/viewer/shared/metric-card";
import { SectionHeading } from "@/components/features/viewer/shared/section-heading";
import { StatChip } from "@/components/features/viewer/shared/stat-chip";
import { TrainingProgressDialog } from "@/components/features/viewer/training/training-progress-dialog";
import { ViewerWorkspaceNav } from "@/components/features/viewer/viewer-workspace-nav";
import { type TrainingRunPlan } from "@/lib/api";
import { IMPLEMENTED_FEATURES } from "@/lib/feature-catalog";

describe("FullPageStatus", () => {
  it("renders status content, detail text, icon, and action", () => {
    render(
      <FullPageStatus
        title="Backend offline"
        detail="Start the API server."
        icon={<span aria-hidden>!</span>}
        action={<button type="button">Retry</button>}
      />,
    );

    expect(screen.getByRole("heading", { name: "Backend offline" })).toBeInTheDocument();
    expect(screen.getByText("Start the API server.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders the loading title without optional detail text", () => {
    render(<FullPageLoading />);

    expect(screen.getByRole("heading", { name: "Loading viewer" })).toBeInTheDocument();
    expect(screen.queryByText(/Start the API/i)).not.toBeInTheDocument();
  });
});

describe("FullPageError", () => {
  it("uses the fallback error copy and calls retry", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<FullPageError onRetry={onRetry} />);

    expect(screen.getByText(/unexpected error/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("renders a provided error message", () => {
    render(<FullPageError message="Exploded during render" onRetry={() => {}} />);

    expect(screen.getByText("Exploded during render")).toBeInTheDocument();
  });
});

describe("ViewerWorkspaceNav", () => {
  it("marks the active workspace and emits workspace changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ViewerWorkspaceNav activeWorkspace="model" onChange={onChange} />);

    expect(screen.getByRole("button", { name: "Model" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Logs" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    await user.click(screen.getByRole("button", { name: "Logs" }));

    expect(onChange).toHaveBeenCalledWith("logs");
  });
});

describe("FeatureListDialog", () => {
  it("renders the feature catalog grouped in a modal dialog", () => {
    render(<FeatureListDialog onClose={() => {}} />);

    const dialog = screen.getByRole("dialog", { name: "Implemented Features" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(within(dialog).getByText(`${IMPLEMENTED_FEATURES.length} features`))
      .toBeInTheDocument();
    expect(within(dialog).getByText(IMPLEMENTED_FEATURES[0].title)).toBeInTheDocument();
  });

  it("calls onClose from the close button", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<FeatureListDialog onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: "Close implemented features" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("InlineStatus", () => {
  it("renders children with default spacing and tone", () => {
    render(<InlineStatus>Waiting for config</InlineStatus>);

    const status = screen.getByText("Waiting for config");
    expect(status).toHaveClass(
      "rounded-[10px]",
      "border",
      "border-dashed",
      "p-4",
      "border-faint",
      "bg-white/[0.018]",
      "text-ink-faint",
    );
  });

  it("applies compact spacing", () => {
    render(<InlineStatus compact>Compact status</InlineStatus>);

    const status = screen.getByText("Compact status");
    expect(status).toHaveClass("p-3");
    expect(status).not.toHaveClass("p-4");
  });

  it("applies tone classes", () => {
    const { rerender } = render(<InlineStatus tone="danger">Danger</InlineStatus>);
    expect(screen.getByText("Danger")).toHaveClass(
      "border-danger-line",
      "bg-danger-soft",
      "text-[#fda4af]",
    );

    rerender(<InlineStatus tone="warning">Warning</InlineStatus>);
    expect(screen.getByText("Warning")).toHaveClass(
      "border-amber/40",
      "bg-amber/[0.12]",
      "text-amber",
    );

    rerender(<InlineStatus tone="success">Success</InlineStatus>);
    expect(screen.getByText("Success")).toHaveClass(
      "border-ok/30",
      "bg-ok/10",
      "text-ok",
    );
  });

  it("preserves explicit roles", () => {
    render(<InlineStatus role="alert">Failed to load</InlineStatus>);

    expect(screen.getByRole("alert")).toHaveTextContent("Failed to load");
  });

  it("renders a busy spinner", () => {
    render(<InlineStatus busy>Loading status</InlineStatus>);

    const spinner = screen.getByText("Loading status").querySelector("svg");
    expect(spinner).toHaveClass("h-4", "w-4", "animate-spin", "text-violet");
    expect(spinner).toHaveAttribute("aria-hidden", "true");
  });

  it("applies caller classes after defaults", () => {
    render(<InlineStatus className="p-2 text-xs custom-status">Custom status</InlineStatus>);

    const status = screen.getByText("Custom status");
    expect(status).toHaveClass("p-2", "text-xs", "custom-status");
    expect(status).not.toHaveClass("p-4", "text-sm");
  });
});

describe("SectionHeading", () => {
  it("renders an h2 title with icon and caller classes", () => {
    render(
      <SectionHeading
        as="h2"
        className="custom-heading text-ink"
        icon={<span data-testid="heading-icon" aria-hidden />}
        title="Metrics"
      />,
    );

    const heading = screen.getByRole("heading", { name: "Metrics" });
    expect(heading).toHaveClass(
      "flex",
      "items-center",
      "gap-2",
      "text-xs",
      "font-bold",
      "uppercase",
      "tracking-[0.09em]",
      "text-ink",
      "custom-heading",
    );
    expect(screen.getByTestId("heading-icon")).toBeInTheDocument();
  });

  it("renders count and actions", () => {
    render(
      <SectionHeading
        title="Logs"
        count={<span>12</span>}
        actions={<button type="button">Refresh</button>}
      />,
    );

    expect(screen.getByText("Logs")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
  });
});

describe("StatChip", () => {
  it("renders the default chip classes", () => {
    render(<StatChip>8 / 10</StatChip>);

    expect(screen.getByText("8 / 10")).toHaveClass(
      "rounded-[7px]",
      "border",
      "border-line",
      "bg-white/[0.04]",
      "px-2",
      "py-1",
      "font-mono",
      "text-xs",
      "text-ink-dim",
    );
  });

  it("applies soft tone, other tone classes, size classes, and caller classes", () => {
    const { rerender } = render(
      <StatChip tone="soft" className="custom-chip">
        Soft
      </StatChip>,
    );

    expect(screen.getByText("Soft")).toHaveClass(
      "border-line-soft",
      "bg-black/20",
      "text-[11px]",
      "font-semibold",
      "leading-none",
      "custom-chip",
    );

    rerender(
      <StatChip tone="violet" size="xs">
        Violet
      </StatChip>,
    );
    expect(screen.getByText("Violet")).toHaveClass(
      "border-violet/30",
      "bg-violet/10",
      "text-violet",
      "px-1.5",
      "py-0.5",
      "text-[11px]",
    );

    rerender(<StatChip tone="success">Success</StatChip>);
    expect(screen.getByText("Success")).toHaveClass("border-ok/30", "bg-ok/10", "text-ok");

    rerender(<StatChip tone="warning">Warning</StatChip>);
    expect(screen.getByText("Warning")).toHaveClass(
      "border-amber/40",
      "bg-amber/[0.12]",
      "text-amber",
    );

    rerender(<StatChip tone="danger">Danger</StatChip>);
    expect(screen.getByText("Danger")).toHaveClass(
      "border-danger-line",
      "bg-danger-soft",
      "text-[#fda4af]",
    );
  });
});

describe("MetricCard", () => {
  it("renders label, value, detail, and caller classes", () => {
    render(
      <MetricCard
        label="Runs"
        value="42"
        valueTitle="42 runs"
        detail="complete"
        className="custom-card py-2.5"
        valueClassName="text-sm text-violet"
        detailClassName="text-ok"
      />,
    );

    const card = screen.getByText("Runs").closest(".edge");
    expect(card).toHaveClass("rounded-[12px]", "px-3", "py-2.5", "custom-card");
    expect(screen.getByText("Runs")).toHaveClass(
      "text-xs",
      "font-bold",
      "uppercase",
      "tracking-[0.08em]",
      "text-ink-dim",
    );
    expect(screen.getByText("42")).toHaveClass(
      "mt-1",
      "font-mono",
      "font-extrabold",
      "text-sm",
      "text-violet",
    );
    expect(screen.getByTitle("42 runs")).toHaveTextContent("42");
    expect(screen.getByText("complete")).toHaveClass("mt-1", "text-ok");
  });
});

describe("KeyValueRow", () => {
  it("renders the line variant with default value classes and caller classes", () => {
    render(
      <KeyValueRow
        label="Params"
        value="1,024"
        className="custom-row last:border-b-0"
        valueClassName="text-[#d7c9ff]"
      />,
    );

    const row = screen.getByText("Params").parentElement;
    expect(row).toHaveClass(
      "grid",
      "grid-cols-[104px_minmax(0,1fr)]",
      "gap-3",
      "border-b",
      "border-line-soft",
      "py-3",
      "text-xs",
      "custom-row",
      "last:border-b-0",
    );
    expect(screen.getByText("1,024")).toHaveClass(
      "break-words",
      "text-right",
      "font-mono",
      "font-semibold",
      "text-[#d7c9ff]",
    );
  });

  it("renders the card variant", () => {
    render(<KeyValueRow variant="card" label="result.json" value="present" />);

    const row = screen.getByText("result.json").parentElement;
    expect(row).toHaveClass(
      "grid-cols-[minmax(0,1fr)_auto]",
      "items-center",
      "rounded-[9px]",
      "border-line-soft",
      "bg-black/20",
      "px-3",
      "py-2",
    );
    expect(screen.getByText("present")).toHaveClass(
      "font-mono",
      "font-semibold",
      "text-ink",
    );
  });
});

describe("TrainingProgressDialog", () => {
  it("opens the full traceback for a failed row", async () => {
    const user = userEvent.setup();
    const plan: TrainingRunPlan = {
      model: "linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: {},
      search: null,
      logFolder: "errors",
      isRandomSearch: false,
      runs: [
        {
          id: "run-0001",
          index: 1,
          status: "Failed",
          preset: "baseline",
          dataset: "Mnist",
          changes: [],
          overrides: {},
          command: "source experiment.sh linear --preset baseline",
          totalEpochs: 2,
          currentEpoch: 0,
          metrics: {},
          logDir: null,
          error: "scalar conversion failed",
          errorTraceback:
            "Traceback (most recent call last):\nRuntimeError: scalar conversion failed",
        },
      ],
      summary: {
        totalRuns: 1,
        completedRuns: 0,
        runningRuns: 0,
        pendingRuns: 0,
        failedRuns: 1,
        cancelledRuns: 0,
        skippedRuns: 0,
        totalEpochs: 2,
        completedEpochs: 0,
        remainingEpochs: 0,
      },
    };

    render(
      <TrainingProgressDialog
        plan={plan}
        isLoading={false}
        error=""
        canResample={false}
        isResampling={false}
        onResample={() => {}}
        onClose={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Full error for run 1" }));

    const dialog = screen.getByRole("dialog", { name: "Training Error" });
    expect(within(dialog).getByText(/Traceback \(most recent call last\):/))
      .toBeInTheDocument();
    expect(within(dialog).getByText(/RuntimeError: scalar conversion failed/))
      .toBeInTheDocument();
  });
});

describe("createViewerContext", () => {
  it("returns values from its generated provider", () => {
    const [Provider, useValue] = createViewerContext<{ label: string }>("TestContext");

    function Consumer() {
      return <span>{useValue().label}</span>;
    }

    render(
      <Provider value={{ label: "provided" }}>
        <Consumer />
      </Provider>,
    );

    expect(screen.getByText("provided")).toBeInTheDocument();
  });

  it("throws a clear error when rendered outside its provider", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const [, useValue] = createViewerContext<{ label: string }>("MissingContext");

    function Consumer() {
      useValue();
      return null;
    }

    expect(() => render(<Consumer />)).toThrow(
      "MissingContext value is missing; render within its provider.",
    );
    consoleError.mockRestore();
  });
});
