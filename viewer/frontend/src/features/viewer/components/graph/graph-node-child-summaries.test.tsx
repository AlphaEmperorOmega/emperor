import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { GraphNodeChildSummaries } from "@/features/viewer/components/graph/graph-node-child-summaries";
import { type GraphParameterActivity } from "@/lib/graph";

const activity: GraphParameterActivity = {
  targetPath: "main_model.0.model",
  weights: {
    status: "updated",
    source: "historical",
    sourceLabel: "2 historical runs",
    metric: "main_model.0.model/weights/relative_delta_norm",
    lastStep: 8,
    observedPoints: 3,
    updatedRuns: 2,
    unchangedRuns: 0,
    missingRuns: 0,
    unknownRuns: 0,
    totalRuns: 2,
  },
  bias: {
    status: "mixed",
    source: "historical",
    sourceLabel: "2 historical runs",
    metric: "main_model.0.model/bias/delta_norm",
    lastStep: 7,
    observedPoints: 2,
    updatedRuns: 1,
    unchangedRuns: 1,
    missingRuns: 0,
    unknownRuns: 0,
    totalRuns: 2,
  },
};

describe("GraphNodeChildSummaries", () => {
  it("uses the W/b indicator group as the only activity popup trigger", () => {
    const onParentClick = vi.fn();
    const onOpenMonitor = vi.fn();
    render(
      <div onClick={onParentClick}>
        <GraphNodeChildSummaries
          nodeId="main_model.0"
          summaries={[
            {
              label: "LinearLayer",
              dims: "128 -> 64",
              kind: "child",
              parameterActivity: activity,
            },
            { label: "Gate", kind: "mechanism" },
          ]}
          monitorButton={
            <button
              type="button"
              aria-label="Open monitor charts for main_model.0"
              onClick={(event) => {
                event.stopPropagation();
                onOpenMonitor();
              }}
            >
              monitor
            </button>
          }
        />
      </div>,
    );

    const row = screen.getByTestId("child-summary-main_model.0-0");
    const dims = within(row).getByText("128 -> 64");
    const indicators = within(row).getByRole("button", {
      name: /Parameter activity: weights updated, bias mixed/i,
    });
    const monitorButton = within(row).getByRole("button", {
      name: /open monitor charts for main_model\.0/i,
    });
    const weightsIndicator = within(indicators).getByTestId(
      "graph-parameter-indicator-weights",
    );
    const biasIndicator = within(indicators).getByTestId(
      "graph-parameter-indicator-bias",
    );

    expect(row).toHaveClass("h-9", "rounded-[10px]", "gap-2", "overflow-hidden");
    expect(row).toHaveAttribute("aria-label", "LinearLayer 128 -> 64");
    expect(row).not.toHaveAttribute("role", "button");
    expect(row).not.toHaveAttribute("tabindex");
    expect(dims).toHaveClass("shrink-0", "whitespace-nowrap", "font-mono");
    expect(indicators).toHaveClass(
      "inline-flex",
      "shrink-0",
      "items-center",
      "gap-1",
      "whitespace-nowrap",
    );
    expect(indicators).not.toHaveClass(
      "h-7",
      "rounded-[8px]",
      "border",
      "bg-white/[0.03]",
      "px-1.5",
    );
    expect(within(row).getAllByRole("button")).toEqual([
      indicators,
      monitorButton,
    ]);
    expect(
      indicators.compareDocumentPosition(monitorButton) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(weightsIndicator).not.toHaveAttribute("role", "button");
    expect(biasIndicator).not.toHaveAttribute("role", "button");
    expect(weightsIndicator).toHaveClass("text-ok");
    expect(weightsIndicator).not.toHaveClass(
      "rounded-[6px]",
      "border",
      "border-ok/35",
      "bg-ok/10",
      "px-1.5",
    );
    expect(biasIndicator).toHaveClass("text-amber");
    expect(biasIndicator).not.toHaveClass(
      "rounded-[6px]",
      "border",
      "border-amber/40",
      "bg-amber/[0.12]",
      "px-1.5",
    );

    fireEvent.click(indicators);

    const popup = screen.getByRole("tooltip");
    expect(screen.getAllByRole("tooltip")).toHaveLength(1);
    expect(popup).toHaveTextContent("Weights");
    expect(popup).toHaveTextContent("Bias");
    expect(popup).toHaveTextContent(
      "This parameter was logged and at least one sampled point showed update or value-change evidence.",
    );
    expect(popup).toHaveTextContent(
      "Historical runs have mixed or incomplete update evidence for this parameter.",
    );
    expect(popup).toHaveTextContent("2 historical runs");
    expect(popup).toHaveTextContent("step: 8 - samples: 3");
    expect(popup).toHaveTextContent(
      "1 updated / 1 unchanged / 0 missing / 0 unknown",
    );
    expect(popup).not.toHaveTextContent("main_model.0.model/bias/delta_norm");
    expect(onParentClick).not.toHaveBeenCalled();

    fireEvent.click(monitorButton);
    expect(onOpenMonitor).toHaveBeenCalledTimes(1);
    expect(onParentClick).not.toHaveBeenCalled();

    fireEvent.keyDown(indicators, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    fireEvent.keyDown(indicators, { key: "Enter" });
    expect(screen.getByRole("tooltip")).toHaveTextContent("Parameter activity");

    fireEvent.blur(indicators);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    fireEvent.keyDown(indicators, { key: " " });
    expect(screen.getByRole("tooltip")).toHaveTextContent("Weights");

    fireEvent.mouseLeave(indicators);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    fireEvent.focus(indicators);
    expect(screen.getByRole("tooltip")).toHaveTextContent("Bias");

    fireEvent.blur(indicators);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("keeps summaries without activity visual-only", () => {
    render(
      <GraphNodeChildSummaries
        nodeId="main_model.0"
        summaries={[
          { label: "LinearLayer", dims: "128 -> 64", kind: "child" },
          { label: "Gate", kind: "mechanism" },
        ]}
      />,
    );

    expect(screen.queryByRole("button", { name: /LinearLayer/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Gate/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByTestId("graph-parameter-indicators"))
      .not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("LinearLayer 128 -> 64"));

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});
