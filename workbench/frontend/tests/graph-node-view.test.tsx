import { fireEvent, render, screen, within } from "@testing-library/react";
import type React from "react";
import { describe, expect, it, vi } from "vitest";
import {
  GraphNodeRenderModeProvider,
  nodeTypes,
} from "@/features/workbench/components/graph/graph-node-view";
import { graphParameterActivityStatusClassNames } from "@/features/workbench/components/graph/graph-parameter-indicators";
import { SelectedNodeDetails } from "@/features/workbench/components/graph/selected-node-details";
import type { GraphNode } from "@/lib/api";
import type { WorkbenchNodeData } from "@/lib/graph";
import { graphCardGeometry } from "@/lib/graph/constants";
import { EXPERT_DIAGRAM_SAMPLER_WIDTH } from "@/features/workbench/components/graph/graph-node-diagram-layout";

vi.mock("@xyflow/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@xyflow/react")>();
  return {
    ...actual,
    Handle: () => null,
  };
});

function renderGraphNode(
  data: Partial<WorkbenchNodeData> = {},
  options: { isViewportMoving?: boolean } = {},
) {
  const GraphNode = nodeTypes.workbenchNode as unknown as React.ComponentType<{
    data: WorkbenchNodeData;
    selected: boolean;
  }>;

  return render(
    <GraphNodeRenderModeProvider
      isViewportMoving={options.isViewportMoving ?? false}
    >
      <GraphNode
        selected={false}
        data={{
          nodeId: "main_model.0",
          label: "Layer",
          typeName: "Layer",
          description: undefined,
          subtitle: "main_model.0",
          path: "main_model.0",
          graphRole: "architecture",
          parameterCount: 0,
          parameterSizeBytes: 0,
          details: {},
          config: null,
          childCount: 1,
          childSummaries: [{ label: "LinearLayer", kind: "child" }],
          graphDetailMode: "basic",
          height: 164,
          isRootNode: false,
          isExpanded: false,
          canToggleExpansion: true,
          isDetailsExpanded: false,
          onActivateNode: () => {},
          onToggleExpansion: () => {},
          onToggleDetails: () => {},
          ...data,
        }}
      />
    </GraphNodeRenderModeProvider>,
  );
}

describe("GraphNodeView", () => {
  it("keeps explicit bottom padding around the card contents", () => {
    renderGraphNode();

    const card = screen.getByTestId("graph-node-card-main_model.0");
    expect(card).toHaveClass(
      "nodrag",
      "nopan",
      "edge",
      "px-region",
    );
    expect(card).toHaveStyle({
      paddingBottom: `${graphCardGeometry.paddingBlock}px`,
      paddingTop: `${graphCardGeometry.paddingBlock}px`,
    });
  });

  it("renders basic-mode parameter and child badges in the footer", () => {
    renderGraphNode({
      parameterCount: 12500,
      childCount: 2,
      height: 164,
    });

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const title = within(titleRow).getByText("Layer");
    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");
    const footerStats = screen.getByTestId("graph-node-footer-stats-main_model.0");

    expect(title).toHaveClass("min-w-0", "flex-1", "truncate");
    expect(within(titleRow).queryByRole("button")).not.toBeInTheDocument();
    expect(within(titleRow).queryByTitle("12,500 parameters")).not.toBeInTheDocument();
    expect(screen.queryByTestId("graph-node-badges-main_model.0")).not.toBeInTheDocument();
    const params = within(footerStats).getByTitle("12,500 parameters");
    const children = within(footerStats).getByText("2 children");
    expect(params).toHaveTextContent("12.5K params");
    expect(params).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(children).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(actionBar).toHaveClass("items-center");
    expect(actionBar).toHaveStyle({
      height: `${graphCardGeometry.actionBar.height}px`,
      marginTop: `${graphCardGeometry.actionBar.marginBlockStart}px`,
    });
    expect(actionBar).not.toHaveClass("mt-auto", "h-10", "items-end");
    expect(screen.getByText("main_model.0")).toHaveStyle({
      height: `${graphCardGeometry.subtitle.height}px`,
      lineHeight: `${graphCardGeometry.subtitle.height}px`,
      marginTop: `${graphCardGeometry.subtitle.marginBlockStart}px`,
    });
  });

  it("renders class names as card titles with semantic subtitles", () => {
    renderGraphNode({
      nodeId: "main_model.block_model",
      label: "LayerStack",
      typeName: "LayerStack",
      subtitle: "Block Model · main_model.block_model",
      path: "main_model.block_model",
      childCount: 0,
      canToggleExpansion: false,
      height: 154,
      config: {
        typeName: "LayerStackConfig",
        fields: [{ key: "num_layers", value: 2 }],
      },
    });

    const card = screen.getByTestId("graph-node-card-main_model.block_model");
    const titleRow = screen.getByTestId("graph-node-title-row-main_model.block_model");

    expect(within(titleRow).getByText("LayerStack")).toBeInTheDocument();
    expect(within(titleRow).queryByText("Block Model")).not.toBeInTheDocument();
    expect(within(card).getByText("Block Model · main_model.block_model"))
      .toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: /^config options for main_model\.block_model$/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /config options for block model/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps full-mode parameter and child badges in the footer", () => {
    renderGraphNode({
      graphDetailMode: "full",
      parameterCount: 12500,
      childCount: 2,
      height: 164,
    });

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const footerStats = screen.getByTestId("graph-node-footer-stats-main_model.0");

    expect(within(titleRow).getByText("Layer")).toBeInTheDocument();
    expect(within(titleRow).queryByTitle("12,500 parameters")).not.toBeInTheDocument();
    expect(screen.queryByTestId("graph-node-badges-main_model.0")).not.toBeInTheDocument();
    expect(footerStats).toHaveClass("overflow-hidden");
    expect(within(footerStats).getByTitle("12,500 parameters")).toHaveClass(
      "h-6",
      "whitespace-nowrap",
    );
    expect(within(footerStats).getByTitle("12,500 parameters"))
      .toHaveTextContent("12.5K params");
    expect(within(footerStats).getByText("2 children")).toHaveClass(
      "h-6",
      "whitespace-nowrap",
    );
    expect(screen.getByText("main_model.0")).toHaveStyle({
      height: `${graphCardGeometry.subtitle.height}px`,
      lineHeight: `${graphCardGeometry.subtitle.height}px`,
      marginTop: `${graphCardGeometry.subtitle.marginBlockStart}px`,
    });
  });

  it("renders model size only on the root model node", () => {
    renderGraphNode({
      nodeId: "model",
      path: "model",
      subtitle: "model",
      graphDetailMode: "full",
      parameterCount: 65792,
      parameterSizeBytes: 263168,
      isRootNode: true,
      height: 154,
    });

    expect(screen.getByTitle("263,168 bytes of parameter tensors"))
      .toHaveTextContent("0.25 MB");
  });

  it("does not render model size on non-root graph nodes", () => {
    renderGraphNode({
      graphDetailMode: "full",
      parameterCount: 65792,
      parameterSizeBytes: 263168,
      height: 154,
    });

    expect(screen.queryByTitle("263,168 bytes of parameter tensors"))
      .not.toBeInTheDocument();
  });

  it("keeps footer count badges from wrapping around long titles", () => {
    const longLabel = "Layer With An Extremely Long Display Name That Should Truncate";
    renderGraphNode({
      label: longLabel,
      parameterCount: 1234567,
      childCount: 12,
      height: 164,
    });

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const title = within(titleRow).getByText(longLabel);
    const footerStats = screen.getByTestId("graph-node-footer-stats-main_model.0");
    const params = within(footerStats).getByTitle("1,234,567 parameters");
    const children = within(footerStats).getByText("12 children");

    expect(title).toHaveClass("min-w-0", "flex-1", "truncate");
    expect(params).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(children).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(within(titleRow).queryByTitle("1,234,567 parameters")).not.toBeInTheDocument();
    expect(screen.queryByTestId("graph-node-badges-main_model.0")).not.toBeInTheDocument();
  });

  it("activates the node from card body clicks", () => {
    const onActivateNode = vi.fn();
    renderGraphNode({ onActivateNode });

    fireEvent.click(screen.getByRole("button", { name: /select and expand main_model\.0/i }));

    expect(onActivateNode).toHaveBeenCalledTimes(1);
  });

  it("uses a lightweight card shell while the graph viewport is moving", () => {
    const onActivateNode = vi.fn();
    renderGraphNode(
      {
        onActivateNode,
        parameterCount: 12500,
        details: { weightShape: "128 x 128" },
        config: {
          typeName: "Layer",
          fields: [{ key: "activation", value: "GELU" }],
        },
        childSummaries: [{ label: "LinearLayer", kind: "child", dims: "128 -> 128" }],
        height: 180,
      },
      { isViewportMoving: true },
    );

    const shell = screen.getByTestId("graph-node-moving-main_model.0");
    expect(shell).toHaveTextContent("Layer");
    expect(shell).toHaveTextContent("12.5K params");
    expect(screen.queryByTestId("graph-node-title-row-main_model.0"))
      .not.toBeInTheDocument();
    expect(screen.queryByTestId("child-summaries-main_model.0"))
      .not.toBeInTheDocument();
    expect(screen.queryByTestId("parameter-shapes-main_model.0"))
      .not.toBeInTheDocument();
    expect(screen.queryByTestId("graph-node-action-bar-main_model.0"))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /config options for main_model\.0/i }))
      .not.toBeInTheDocument();

    fireEvent.click(
      within(shell).getByRole("button", {
        name: /select and expand main_model\.0/i,
      }),
    );

    expect(onActivateNode).toHaveBeenCalledTimes(1);
  });

  it("uses class names in the moving shell for semantic containers", () => {
    renderGraphNode(
      {
        nodeId: "main_model.block_model",
        label: "LayerStack",
        typeName: "LayerStack",
        subtitle: "Block Model · main_model.block_model",
        path: "main_model.block_model",
        childCount: 0,
        canToggleExpansion: false,
        height: graphCardGeometry.simpleHeight,
      },
      { isViewportMoving: true },
    );

    const shell = screen.getByTestId("graph-node-moving-main_model.block_model");
    expect(within(shell).getByText("LayerStack")).toBeInTheDocument();
    expect(within(shell).queryByText("Block Model")).not.toBeInTheDocument();
    expect(within(shell).getByText("main_model.block_model")).toBeInTheDocument();
  });

  it("uses the chevron as the explicit expansion toggle", () => {
    const onActivateNode = vi.fn();
    const onToggleExpansion = vi.fn();
    renderGraphNode({ onActivateNode, onToggleExpansion });

    fireEvent.click(screen.getByRole("button", { name: /^expand tree main_model\.0$/i }));

    expect(onToggleExpansion).toHaveBeenCalledTimes(1);
    expect(onActivateNode).not.toHaveBeenCalled();
  });

  it("keeps header title rows free of action controls", () => {
    renderGraphNode({
      details: { activation: "GELU" },
      parameterCount: 12500,
      canOpenMonitor: true,
      onOpenMonitor: vi.fn(),
      parameterActivity: {
        targetPath: "main_model.0.model",
        weights: {
          status: "updated",
          source: "active-job",
          sourceLabel: "active job job-1",
          observedPoints: 2,
        },
      },
    });

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");

    expect(within(titleRow).getByText("Layer")).toBeInTheDocument();
    expect(within(titleRow).queryByRole("button")).not.toBeInTheDocument();
    expect(within(titleRow).queryByTestId("graph-parameter-indicators"))
      .not.toBeInTheDocument();
  });

  it("renders graph controls in the footer action bar order without parameter activity", () => {
    renderGraphNode({
      details: { activation: "GELU" },
      canOpenMonitor: true,
      onOpenMonitor: vi.fn(),
      parameterActivity: {
        targetPath: "main_model.0.model",
        weights: {
          status: "updated",
          source: "active-job",
          sourceLabel: "active job job-1",
          metric: "main_model.0.model/weights/relative_delta_norm",
          lastStep: 8,
          observedPoints: 2,
        },
        bias: {
          status: "missing",
          source: "active-job",
          sourceLabel: "active job job-1",
          observedPoints: 0,
        },
      },
    });

    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");
    const detailsButton = within(actionBar).getByRole("button", {
      name: /^details for main_model\.0$/i,
    });
    const expandButton = within(actionBar).getByRole("button", {
      name: /^expand tree main_model\.0$/i,
    });
    const monitorButton = within(actionBar).getByRole("button", {
      name: /^open monitor charts for main_model\.0$/i,
    });
    const infoButton = within(actionBar).getByRole("button", {
      name: /^open component info for main_model\.0$/i,
    });
    const footerStats = within(actionBar).getByTestId("graph-node-footer-stats-main_model.0");

    expect(detailsButton.compareDocumentPosition(expandButton) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(expandButton.compareDocumentPosition(infoButton) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(infoButton.compareDocumentPosition(footerStats) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(footerStats.compareDocumentPosition(monitorButton) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(within(actionBar).queryByTestId("graph-parameter-indicators"))
      .not.toBeInTheDocument();
  });

  it("opens component info from the graph cell footer", () => {
    const onActivateNode = vi.fn();
    const onToggleExpansion = vi.fn();
    renderGraphNode({
      typeName: "LinearLayer",
      label: "LinearLayer",
      description: "Applies a learned linear projection.",
      path: "main_model.0.model",
      graphRole: "architecture",
      config: {
        typeName: "LinearLayerConfig",
        fields: [
          {
            key: "input_dim",
            value: 128,
            description: "Input feature dimension.",
          },
          { key: "bias_flag", value: false },
        ],
      },
      onActivateNode,
      onToggleExpansion,
    });

    fireEvent.click(
      screen.getByRole("button", {
        name: /^open component info for main_model\.0\.model$/i,
      }),
    );

    const dialog = screen.getByRole("dialog", { name: "Component Info" });
    expect(within(dialog).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(dialog).getAllByText("main_model.0.model").length).toBeGreaterThan(0);
    expect(
      within(dialog).getByText("Applies a learned linear projection."),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("LinearLayerConfig")).toBeInTheDocument();
    expect(
      within(dialog).getByTestId("component-info-config-field-input_dim"),
    ).toHaveTextContent("input_dim - 128");
    expect(within(dialog).getByText("Input feature dimension.")).toBeInTheDocument();
    expect(
      within(dialog).getByTestId("component-info-config-field-bias_flag"),
    ).toHaveTextContent("bias_flag - false");
    expect(
      within(dialog).getByText("No field description available"),
    ).toBeInTheDocument();
    expect(onActivateNode).not.toHaveBeenCalled();
    expect(onToggleExpansion).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole("button", { name: "Close component info" }));

    expect(screen.queryByRole("dialog", { name: "Component Info" }))
      .not.toBeInTheDocument();
  });

  it("uses fallback text for graph component info without descriptions or config", () => {
    renderGraphNode({
      typeName: "Dropout",
      label: "Dropout",
      path: "main_model.dropout",
      graphRole: "internal",
      canToggleExpansion: false,
      config: null,
    });

    fireEvent.click(
      screen.getByRole("button", {
        name: /^open component info for main_model\.dropout$/i,
      }),
    );

    const dialog = screen.getByRole("dialog", { name: "Component Info" });
    expect(within(dialog).getByText("Dropout")).toBeInTheDocument();
    expect(within(dialog).getByText("No description available")).toBeInTheDocument();
    expect(within(dialog).getByText("No config")).toBeInTheDocument();
    expect(within(dialog).getByText("No config fields available")).toBeInTheDocument();
  });

  it("does not render node-level parameter activity in the action bar", () => {
    renderGraphNode({
      parameterActivity: {
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
      },
    });

    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");
    expect(within(actionBar).queryByTestId("graph-parameter-indicators"))
      .not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Weights parameter activity:/i))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("moves monitor charts next to child-summary W/b activity indicators", () => {
    const onActivateNode = vi.fn();
    const onToggleExpansion = vi.fn();
    const onOpenMonitor = vi.fn();
    renderGraphNode({
      canOpenMonitor: true,
      onActivateNode,
      onToggleExpansion,
      onOpenMonitor,
      childSummaries: [
        {
          label: "LinearLayer",
          dims: "128 -> 64",
          kind: "child",
          parameterActivity: {
            targetPath: "main_model.0.model",
            weights: {
              status: "updated",
              source: "historical",
              sourceLabel: "2 historical runs",
              observedPoints: 3,
            },
            bias: {
              status: "updated",
              source: "historical",
              sourceLabel: "2 historical runs",
              observedPoints: 2,
            },
          },
        },
      ],
    });

    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");
    const row = screen.getByTestId("child-summary-main_model.0-0");
    const indicators = within(row).getByTestId("graph-parameter-indicators");
    const monitorButton = within(row).getByRole("button", {
      name: /^open monitor charts for main_model\.0$/i,
    });

    expect(
      within(actionBar).queryByRole("button", {
        name: /^open monitor charts for main_model\.0$/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      indicators.compareDocumentPosition(monitorButton) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(indicators).not.toHaveClass("h-7", "rounded-[8px]");
    expect(monitorButton).toHaveClass(
      "h-touch",
      "w-touch",
      "md:h-control-sm",
      "md:w-control-sm",
      "rounded-control-sm",
      "border-violet/35",
      "bg-accent-soft",
    );
    expect(monitorButton).not.toHaveClass(
      "border-transparent",
      "bg-transparent",
    );

    fireEvent.click(monitorButton);

    expect(onOpenMonitor).toHaveBeenCalledTimes(1);
    expect(onToggleExpansion).not.toHaveBeenCalled();
    expect(onActivateNode).not.toHaveBeenCalled();
  });

  it("opens monitor charts without activating or expanding the node", () => {
    const onActivateNode = vi.fn();
    const onToggleExpansion = vi.fn();
    const onOpenMonitor = vi.fn();
    renderGraphNode({
      canOpenMonitor: true,
      onActivateNode,
      onToggleExpansion,
      onOpenMonitor,
    });

    fireEvent.click(
      screen.getByRole("button", { name: /^open monitor charts for main_model\.0$/i }),
    );

    expect(onOpenMonitor).toHaveBeenCalledTimes(1);
    expect(onToggleExpansion).not.toHaveBeenCalled();
    expect(onActivateNode).not.toHaveBeenCalled();
  });

  it("hides monitor charts when no monitor data is available", () => {
    renderGraphNode({
      canOpenMonitor: false,
      onOpenMonitor: vi.fn(),
    });

    expect(
      screen.queryByRole("button", { name: /^open monitor charts for main_model\.0$/i }),
    ).not.toBeInTheDocument();
  });

  it("advertises collapse on the expanded card body", () => {
    renderGraphNode({ isExpanded: true });

    expect(screen.getByRole("button", { name: /select and collapse main_model\.0/i }))
      .toHaveAttribute("aria-expanded", "true");
    expect(
      screen.queryByRole("button", { name: /select and expand main_model\.0/i }),
    ).not.toBeInTheDocument();
  });

  it("renders child summary rows with explicit bottom edge styling", () => {
    renderGraphNode();

    const summaries = screen.getByText("LinearLayer").parentElement?.parentElement;
    expect(summaries).not.toHaveClass("overflow-hidden");
    expect(summaries).not.toHaveClass("flex-1");
    const row = screen.getByText("LinearLayer").parentElement;
    expect(row).toHaveClass(
      "rounded-control-md",
      "border",
      "border-line-soft",
      "bg-control",
      "px-3",
      "type-compact",
      "leading-none",
    );
    expect(row).toHaveStyle({
      height: `${graphCardGeometry.childSummary.rowHeight}px`,
    });
  });

  it("renders mechanism summary rows with neutral child row styling", () => {
    renderGraphNode({
      childSummaries: [{ label: "Gate", kind: "mechanism" }],
    });

    const gateSummary = screen.getByText("Gate").parentElement;
    expect(gateSummary).toHaveClass(
      "rounded-control-md",
      "border",
      "border-line-soft",
      "bg-control",
      "type-compact",
      "text-ink-dim",
      "leading-none",
    );
    expect(gateSummary).toHaveStyle({
      height: `${graphCardGeometry.childSummary.rowHeight}px`,
    });
    expect(gateSummary).not.toHaveClass("border-violet/30");
    expect(gateSummary).not.toHaveClass("text-violet-text");
    expect(gateSummary).not.toHaveClass(
      "shadow-control-selected",
    );
    expect(gateSummary?.className).not.toContain("linear-gradient");
  });

  it("renders direct weight and bias shapes inside the card", () => {
    renderGraphNode({
      details: {
        weightShape: "128 x 128",
        biasShape: "128",
      },
      height: 162,
    });

    const shapes = screen.getByTestId("parameter-shapes-main_model.0");
    expect(shapes).toHaveClass("grid-cols-2");
    const weights = within(shapes).getByLabelText("W shape 128 x 128");
    const bias = within(shapes).getByLabelText("B shape 128");
    expect(weights).toHaveClass(
      "rounded-control-md",
      "border-violet/25",
      "bg-violet/15",
      "px-2",
      "type-label",
      "leading-none",
    );
    expect(weights).toHaveStyle({
      height: `${graphCardGeometry.parameterShapes.rowHeight}px`,
    });
    expect(bias).toHaveClass("type-label");
    expect(bias).toHaveStyle({
      height: `${graphCardGeometry.parameterShapes.rowHeight}px`,
    });
    expect(within(shapes).getByText("W")).toBeInTheDocument();
    expect(within(shapes).getByText("W")).toHaveClass("truncate");
    expect(within(shapes).getByText("128 x 128")).toBeInTheDocument();
    expect(within(shapes).getByText("128 x 128")).toHaveClass("truncate");
    expect(within(shapes).getByText("B")).toBeInTheDocument();
    expect(within(shapes).getByText("128")).toBeInTheDocument();
  });

  it("colors direct weight and bias shapes from parameter activity", () => {
    renderGraphNode({
      details: {
        weightShape: "256 x 10",
        biasShape: "10",
      },
      parameterActivity: {
        targetPath: "main_model.0.model",
        weights: {
          status: "updated",
          source: "active-job",
          sourceLabel: "active job job-1",
          observedPoints: 2,
        },
        bias: {
          status: "unchanged",
          source: "active-job",
          sourceLabel: "active job job-1",
          observedPoints: 2,
        },
      },
      height: 162,
    });

    const shapes = screen.getByTestId("parameter-shapes-main_model.0");
    const weights = within(shapes).getByLabelText(
      "W shape 256 x 10, weights activity updated",
    );
    const bias = within(shapes).getByLabelText("B shape 10, bias activity unchanged");

    expect(weights).toHaveAttribute(
      "title",
      "W shape: 256 x 10 (weights activity updated)",
    );
    expect(bias).toHaveAttribute("title", "B shape: 10 (bias activity unchanged)");
    expect(weights).toHaveClass(
      ...graphParameterActivityStatusClassNames.updated.split(" "),
    );
    expect(bias).toHaveClass(
      ...graphParameterActivityStatusClassNames.unchanged.split(" "),
    );
    expect(within(weights).getByText("W")).toHaveClass("text-current");
    expect(within(bias).getByText("B")).toHaveClass("text-current");
    expect(within(bias).queryByText("b")).not.toBeInTheDocument();
  });

  it("does not color unchanged weight shapes as updated", () => {
    renderGraphNode({
      details: {
        weightShape: "32 x 32",
      },
      parameterActivity: {
        targetPath: "main_model.0.model",
        weights: {
          status: "unchanged",
          source: "active-job",
          sourceLabel: "active job job-1",
          observedPoints: 2,
        },
      },
      height: 162,
    });

    const shapes = screen.getByTestId("parameter-shapes-main_model.0");
    const weights = within(shapes).getByLabelText(
      "W shape 32 x 32, weights activity unchanged",
    );

    expect(weights).toHaveClass(
      ...graphParameterActivityStatusClassNames.unchanged.split(" "),
    );
    expect(weights).not.toHaveClass(
      ...graphParameterActivityStatusClassNames.updated.split(" "),
    );
  });

  it("renders direct LinearLayer dims in shape metadata without an empty summary spacer", () => {
    renderGraphNode({
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.0.model",
      subtitle: "main_model.0.model",
      details: {
        weightShape: "10 x 256",
        biasShape: "10",
      },
      config: {
        typeName: "LinearLayerConfig",
        fields: [
          { key: "input_dim", value: 256 },
          { key: "output_dim", value: 10 },
        ],
      },
      childCount: 0,
      childSummaries: [],
      canToggleExpansion: false,
      height: 156,
    });

    const shapes = screen.getByTestId("parameter-shapes-main_model.0");
    const dims = within(shapes.parentElement ?? shapes).getByTitle(
      "input/output: 256 -> 10",
    );
    const weights = within(shapes).getByLabelText("W shape 10 x 256");
    const bias = within(shapes).getByLabelText("B shape 10");
    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");

    expect(shapes).toHaveClass(
      "grid",
      "items-center",
      "grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)]",
    );
    expect(dims).toHaveTextContent("256 -> 10");
    expect(dims).toHaveClass("rounded-control-md", "px-2", "type-label");
    expect(dims).toHaveStyle({
      height: `${graphCardGeometry.parameterShapes.rowHeight}px`,
    });
    expect(dims).not.toHaveClass("h-5", "px-1.5", "type-caption");
    expect(dims.parentElement).toBe(shapes);
    expect(weights.parentElement).toBe(shapes);
    expect(bias.parentElement).toBe(shapes);
    expect(
      dims.compareDocumentPosition(weights) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      dims.compareDocumentPosition(bias) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(screen.queryByTestId("child-summaries-main_model.0"))
      .not.toBeInTheDocument();
    expect(
      (shapes.parentElement ?? shapes).compareDocumentPosition(actionBar) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("renders checkpoint-derived LinearLayer dims instead of stale config dims", () => {
    renderGraphNode({
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.0.gate_model.model.layers.0.model",
      subtitle: "main_model.0.gate_model.model.layers.0.model",
      details: {
        dims: "32 -> 32",
        inputDim: 32,
        outputDim: 32,
        weightShape: "32 x 32",
        biasShape: "32",
      },
      config: {
        typeName: "LinearLayerConfig",
        fields: [
          { key: "input_dim", value: 32 },
          { key: "output_dim", value: 256 },
        ],
      },
      childCount: 0,
      childSummaries: [],
      canToggleExpansion: false,
      height: 156,
    });

    const shapes = screen.getByTestId("parameter-shapes-main_model.0");

    expect(within(shapes).getByTitle("input/output: 32 -> 32"))
      .toHaveTextContent("32 -> 32");
    expect(within(shapes).getByLabelText("W shape 32 x 32"))
      .toHaveTextContent("32 x 32");
    expect(within(shapes).getByLabelText("B shape 32"))
      .toHaveTextContent("32");
    expect(screen.queryByText("32 -> 256")).not.toBeInTheDocument();
  });

  it("omits shape dims when a visible child summary already shows the same dims", () => {
    renderGraphNode({
      details: {
        dims: "128 -> 64",
        weightShape: "64 x 128",
        biasShape: "64",
      },
      childSummaries: [{ label: "LinearLayer", dims: "128 -> 64", kind: "child" }],
      height: 200,
    });

    expect(screen.getByTestId("parameter-shapes-main_model.0")).toBeInTheDocument();
    expect(screen.getByLabelText("LinearLayer 128 -> 64")).toBeInTheDocument();
    expect(screen.queryByTestId("parameter-shape-dims-main_model.0"))
      .not.toBeInTheDocument();
  });

  it("renders layer dims on the inner-model child summary row", () => {
    renderGraphNode({
      childSummaries: [{ label: "LinearLayer", dims: "128 -> 64", kind: "child" }],
    });

    const summary = screen.getByLabelText("LinearLayer 128 -> 64");
    expect(summary).toHaveAttribute("title", "LinearLayer 128 -> 64");
    expect(summary).toHaveClass("rounded-control-md", "gap-2", "overflow-hidden");
    expect(summary).toHaveStyle({
      height: `${graphCardGeometry.childSummary.rowHeight}px`,
    });
    expect(within(summary).getByText("LinearLayer")).toHaveClass("flex-1", "truncate");
    expect(within(summary).getByText("128 -> 64")).toHaveClass(
      "shrink-0",
      "text-right",
      "font-mono",
    );
  });

  it("renders config fields in expanded config options without raw preview rows", () => {
    renderGraphNode({
      details: {
        dims: "128 -> 64",
        weightShape: "128 x 64",
        biasShape: "64",
      },
      config: {
        typeName: "AdaptiveLinearLayerConfig",
        fields: [
          { key: "input_dim", value: 128 },
          { key: "output_dim", value: 64 },
          { key: "bias_flag", value: false },
          { key: "adaptive_augmentation_config", value: null },
        ],
      },
      isDetailsExpanded: true,
      height: 386,
    });

    const detailsButton = screen.getByRole("button", {
      name: /config options for main_model\.0/i,
    });
    const details = document.getElementById(
      detailsButton.getAttribute("aria-controls") ?? "",
    );

    expect(screen.getByTestId("parameter-shapes-main_model.0")).toBeInTheDocument();
    expect(detailsButton).toHaveAttribute("aria-expanded", "true");
    expect(detailsButton).toHaveAttribute("title", "Config options for main_model.0");
    expect(detailsButton).toHaveClass(
      "h-touch",
      "w-touch",
      "md:h-control-sm",
      "md:w-control-sm",
    );
    expect(detailsButton).toHaveTextContent("");
    expect(details).not.toBeNull();
    const detailRow = within(details!).getByText("bias_flag").parentElement;
    expect(detailRow).toHaveClass(
      "grid-cols-[96px_minmax(0,1fr)]",
      "type-compact",
    );
    expect(detailRow).toHaveStyle({
      height: `${graphCardGeometry.details.rowHeight}px`,
    });
    expect(within(details!).getByText("adaptive_augmentation_config")).toBeInTheDocument();
    expect(within(details!).getByText("None")).toBeInTheDocument();
    expect(within(details!).queryByText("weightShape")).not.toBeInTheDocument();
    expect(within(details!).queryByText("biasShape")).not.toBeInTheDocument();
    expect(within(details!).queryByText("dims")).not.toBeInTheDocument();
    expect(within(details!).queryByText("shapeTransition")).not.toBeInTheDocument();
  });

  it("uses an icon-only details toggle with stable accordion attributes", () => {
    const onActivateNode = vi.fn();
    const onToggleDetails = vi.fn();
    renderGraphNode({
      details: { activation: "GELU" },
      onActivateNode,
      onToggleDetails,
    });

    const detailsButton = screen.getByRole("button", {
      name: /details for main_model\.0/i,
    });
    const detailsId = detailsButton.getAttribute("aria-controls");

    expect(detailsButton).toHaveAttribute("aria-expanded", "false");
    expect(detailsId).toBe("graph-node-details-main_model-0");
    expect(document.getElementById(detailsId ?? "")).toBeNull();
    expect(detailsButton.querySelector("svg")).toBeInTheDocument();

    fireEvent.click(detailsButton);

    expect(onToggleDetails).toHaveBeenCalledTimes(1);
    expect(onActivateNode).not.toHaveBeenCalled();
  });

  it("uses an icon between nested child summary labels", () => {
    renderGraphNode({
      childSummaries: [
        { label: "Layer", nestedLabel: "LinearLayer", dims: "512 -> 10", kind: "child" },
      ],
    });

    const relationshipRow = screen.getByLabelText("Layer LinearLayer 512 -> 10");
    expect(relationshipRow).toHaveAttribute("title", "Layer LinearLayer 512 -> 10");
    expect(within(relationshipRow).getByText("Layer")).toBeInTheDocument();
    expect(within(relationshipRow).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(relationshipRow).getByText("512 -> 10")).toBeInTheDocument();
    expect(within(relationshipRow).queryByText("Layer -> LinearLayer")).not.toBeInTheDocument();
    expect(relationshipRow.querySelector("svg")).toBeInTheDocument();
  });

  it("omits the metadata section when details are empty", () => {
    renderGraphNode({ details: {} });

    expect(screen.queryByRole("button", { name: /details for/i })).not.toBeInTheDocument();
    expect(screen.queryByText("No metadata")).not.toBeInTheDocument();
  });

  it("does not stretch the child summary area above expanded details", () => {
    renderGraphNode({
      details: { activation: "GELU" },
      isDetailsExpanded: true,
      height: 208,
    });

    const summaries = screen.getByText("LinearLayer").parentElement?.parentElement;
    const detailsButton = screen.getByRole("button", { name: /details for main_model\.0/i });
    const details = document.getElementById(
      detailsButton.getAttribute("aria-controls") ?? "",
    );
    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");

    expect(summaries).not.toHaveClass("flex-1");
    expect(details).not.toBeNull();
    expect(details!.compareDocumentPosition(actionBar) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
  });

  it("renders stack cells above details and replaces child summaries", () => {
    renderGraphNode({
      details: { dims: "128 -> 128", activation: "GELU" },
      childSummaries: [{ label: "LinearLayer", kind: "child" }],
      stackDiagram: {
        totalLayers: 3,
        hasOverflow: false,
        cells: [
          {
            label: "Layer 0 · LinearLayer",
            title: "Layer 0 · LinearLayer · 128 -> 128",
            dims: "128 -> 128",
            kind: "layer",
            layerIndex: 0,
          },
          {
            label: "Layer 1 · LinearLayer",
            title: "Layer 1 · LinearLayer · 128 -> 128",
            dims: "128 -> 128",
            kind: "layer",
            layerIndex: 1,
          },
          {
            label: "Layer 2 · LinearLayer",
            title: "Layer 2 · LinearLayer · 128 -> 128",
            dims: "128 -> 128",
            kind: "layer",
            layerIndex: 2,
          },
        ],
      },
      height: 264,
    });

    const diagram = screen.getByTestId("stack-diagram-main_model.0");
    const detailsButton = screen.getByRole("button", { name: /details for main_model\.0/i });
    const layer0 = within(diagram).getByTitle("Layer 0 · LinearLayer · 128 -> 128");
    const layer1 = within(diagram).getByTitle("Layer 1 · LinearLayer · 128 -> 128");
    const layer2 = within(diagram).getByTitle("Layer 2 · LinearLayer · 128 -> 128");
    expect(layer0).toHaveTextContent("Layer 0 · LinearLayer");
    expect(layer0).toHaveAttribute("aria-label", "Layer 0 · LinearLayer · 128 -> 128");
    expect(layer0).toHaveClass("rounded-control-md", "px-3", "type-compact");
    expect(layer0).toHaveStyle({
      height: `${graphCardGeometry.childSummary.rowHeight}px`,
    });
    expect(within(layer0).getByText("Layer 0 · LinearLayer")).toHaveClass(
      "min-w-0",
      "flex-1",
      "truncate",
    );
    expect(within(layer0).getByText("128 -> 128")).toHaveClass(
      "shrink-0",
      "text-right",
      "font-mono",
    );
    expect(layer0.compareDocumentPosition(layer1) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(layer1.compareDocumentPosition(layer2) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(within(diagram).queryByText("LinearLayer")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child-summaries-main_model.0")).not.toBeInTheDocument();
    expect(
      diagram.compareDocumentPosition(detailsButton) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("uses graph child-row styling for stack previews", () => {
    renderGraphNode({
      stackDiagram: {
        totalLayers: 5,
        hasOverflow: false,
        cells: Array.from({ length: 5 }, (_, index) => ({
          label: `Layer ${index} · ${
            index === 0 ? "MixtureOfExpertsLayerWithVeryLongName" : "MixtureOfExpertsLayer"
          }`,
          title: `Layer ${index} · ${
            index === 0 ? "MixtureOfExpertsLayerWithVeryLongName" : "MixtureOfExpertsLayer"
          } · 128 -> 128`,
          dims: "128 -> 128",
          kind: "layer" as const,
          layerIndex: index,
        })),
      },
      height: 250,
    });

    const diagram = screen.getByTestId("stack-diagram-main_model.0");
    const layer0 = within(diagram).getByTitle(
      "Layer 0 · MixtureOfExpertsLayerWithVeryLongName · 128 -> 128",
    );
    expect(diagram).toHaveClass("grid");
    expect(diagram).toHaveStyle({
      marginTop: `${graphCardGeometry.contentMarginBlockStart}px`,
      rowGap: `${graphCardGeometry.childSummary.rowGap}px`,
    });
    expect(diagram).not.toHaveClass("h-[160px]");
    expect(layer0).toHaveClass("rounded-control-md", "px-3", "type-compact");
    expect(layer0.style.left).toBe("");
    expect(layer0.style.width).toBe("");
    expect(layer0.style.height).toBe(
      `${graphCardGeometry.childSummary.rowHeight}px`,
    );
    expect(within(layer0).getByText("Layer 0 · MixtureOfExpertsLayerWithVeryLongName"))
      .toHaveClass("min-w-0", "flex-1", "truncate");
    expect(within(layer0).getByText("128 -> 128")).toHaveClass("shrink-0", "text-right");
  });

  it("does not render SVG connector paths for stack cells", () => {
    const { container } = renderGraphNode({
      stackDiagram: {
        totalLayers: 8,
        hasOverflow: true,
        cells: [
          {
            label: "Layer 0 · LinearLayer",
            title: "Layer 0 · LinearLayer",
            kind: "layer",
            layerIndex: 0,
          },
          {
            label: "Layer 1 · LinearLayer",
            title: "Layer 1 · LinearLayer",
            kind: "layer",
            layerIndex: 1,
          },
          { label: "…", title: "5 more layers", kind: "overflow" },
          {
            label: "Layer 7 · LinearLayer",
            title: "Layer 7 · LinearLayer · 256 -> 10",
            dims: "256 -> 10",
            kind: "layer",
            layerIndex: 7,
          },
        ],
      },
      height: 250,
    });

    const connector = container
      .querySelector('[data-testid="stack-diagram-main_model.0"]')
      ?.querySelector("svg");
    expect(connector).toBeNull();
    expect(screen.getByTitle("5 more layers")).toHaveTextContent("…");
    expect(screen.getByTitle("Layer 7 · LinearLayer · 256 -> 10"))
      .toHaveTextContent("Layer 7 · LinearLayer");
    expect(screen.getByTitle("Layer 7 · LinearLayer · 256 -> 10"))
      .toHaveTextContent("256 -> 10");
  });

  it("keeps stack diagram cells visual-only while card activation still works", () => {
    const onActivateNode = vi.fn();
    renderGraphNode({
      onActivateNode,
      stackDiagram: {
        totalLayers: 1,
        hasOverflow: false,
        cells: [
          {
            label: "Layer 0 · LinearLayer",
            title: "Layer 0 · LinearLayer · 128 -> 128",
            dims: "128 -> 128",
            kind: "layer",
            layerIndex: 0,
          },
        ],
      },
      height: 202,
    });

    expect(screen.getByRole("button", { name: /select and expand main_model\.0/i }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^expand tree main_model\.0$/i }))
      .toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", {
        name: /select and expand main_model\.0/i,
      }),
    );
    expect(onActivateNode).toHaveBeenCalledTimes(1);
  });

  it("renders cluster maps inside graph cards and replaces child summaries", () => {
    renderGraphNode({
      nodeId: "neuron_cluster",
      path: "neuron_cluster",
      label: "Neuron Cluster",
      childSummaries: [{ label: "Terminal", kind: "child" }],
      clusterDiagram: {
        columns: 2,
        rows: 2,
        instantiated: 2,
        capacityTotal: 8,
        maxSteps: 1,
        growthThreshold: null,
        hasColumnOverflow: false,
        hasRowOverflow: false,
        hasPlaneOverflow: true,
        planes: [
          {
            z: 1,
            cells: [
              { x: 1, y: 1, filled: true, title: "Neuron (1, 1, 1) active" },
              { x: 2, y: 1, filled: false, title: "Neuron (2, 1, 1) empty" },
              { x: 1, y: 2, filled: true, title: "Neuron (1, 2, 1) active" },
              { x: 2, y: 2, filled: false, title: "Neuron (2, 2, 1) empty" },
            ],
          },
        ],
      },
      height: 190,
    });

    const diagram = screen.getByTestId("cluster-diagram-neuron_cluster");
    const headerHeight =
      graphCardGeometry.clusterDiagram.headerHeight +
      2 * graphCardGeometry.clusterDiagram.cellSize +
      graphCardGeometry.clusterDiagram.cellGap;
    const planeWidth =
      2 * graphCardGeometry.clusterDiagram.cellSize +
      graphCardGeometry.clusterDiagram.cellGap;
    const plane = within(diagram).getByTitle("Z plane 1");
    const activeCell = within(diagram).getByLabelText(/Neuron \(1, 1, 1\).*active/i);
    expect(within(diagram).getByText("Cluster map")).toBeInTheDocument();
    expect(within(diagram).getByText("2 / 8")).toBeInTheDocument();
    expect(within(diagram).getByText("1z")).toHaveClass(
      "rounded-chip",
      "border-violet/25",
      "px-1.5",
      "py-1",
      "type-caption",
      "leading-none",
    );
    expect(within(diagram).getByText("clipped")).toHaveClass(
      "rounded-chip",
      "border-line-soft",
      "px-1.5",
      "py-1",
    );
    expect(Number.parseFloat(diagram.style.height)).toBe(headerHeight);
    expect(Number.parseFloat(plane.style.width)).toBe(planeWidth);
    expect(Number.parseFloat(activeCell.style.height)).toBe(
      graphCardGeometry.clusterDiagram.cellSize,
    );
    expect(Number.parseFloat(activeCell.style.width)).toBe(
      graphCardGeometry.clusterDiagram.cellSize,
    );
    expect(activeCell).toHaveClass("border-violet/45");
    expect(within(diagram).getByLabelText(/Neuron \(2, 1, 1\).*empty/i)).toHaveClass(
      "border-line-soft",
    );
    expect(screen.queryByTestId("child-summaries-neuron_cluster")).not.toBeInTheDocument();
  });

  it("highlights terminal reach locations while hovering an active cluster cell", () => {
    renderGraphNode({
      nodeId: "neuron_cluster",
      path: "neuron_cluster",
      label: "Neuron Cluster",
      clusterDiagram: {
        columns: 3,
        rows: 2,
        instantiated: 3,
        capacityTotal: 6,
        maxSteps: 1,
        growthThreshold: null,
        hasColumnOverflow: false,
        hasRowOverflow: false,
        hasPlaneOverflow: false,
        planes: [
          {
            z: 1,
            cells: [
              {
                x: 1,
                y: 1,
                filled: true,
                title: "Neuron (1, 1, 1) — active",
                reach: {
                  position: [1, 1, 1],
                  connections: [
                    [1, 1, 1],
                    [2, 1, 1],
                    [3, 1, 1],
                    [4, 1, 1],
                    [1, 2, 1],
                  ],
                  inBoundsConnections: [
                    [1, 1, 1],
                    [2, 1, 1],
                    [3, 1, 1],
                    [1, 2, 1],
                  ],
                  activeConnectionTotal: 2,
                  emptyConnectionTotal: 1,
                  outOfBoundsTotal: 1,
                },
              },
              { x: 2, y: 1, filled: true, title: "Neuron (2, 1, 1) — active" },
              { x: 3, y: 1, filled: false, title: "Neuron (3, 1, 1) — empty" },
              { x: 1, y: 2, filled: true, title: "Neuron (1, 2, 1) — active" },
              { x: 2, y: 2, filled: false, title: "Neuron (2, 2, 1) — empty" },
              { x: 3, y: 2, filled: false, title: "Neuron (3, 2, 1) — empty" },
            ],
          },
        ],
      },
      height: 214,
    });

    const diagram = screen.getByTestId("cluster-diagram-neuron_cluster");
    const source = within(diagram).getByLabelText(/Neuron \(1, 1, 1\).*active/i);
    const activeReach = within(diagram).getByLabelText(/Neuron \(2, 1, 1\).*active/i);
    const emptyReach = within(diagram).getByLabelText(/Neuron \(3, 1, 1\).*empty/i);
    const unrelated = within(diagram).getByLabelText(/Neuron \(2, 2, 1\).*empty/i);

    fireEvent.mouseEnter(source);

    expect(within(diagram).getByTestId("cluster-diagram-summary-neuron_cluster"))
      .toHaveTextContent("(1, 1, 1) · 5 reach · 2 active · 1 outside");
    expect(source).toHaveClass("ring-2", "ring-violet-text/80");
    expect(activeReach).toHaveClass("border-cyan/90", "ring-cyan/60");
    expect(emptyReach).toHaveClass("border-cyan/55", "bg-cyan/15");
    expect(unrelated).toHaveClass("opacity-35");
  });

  it("resets the cluster reach overlay on mouse leave", () => {
    renderGraphNode({
      nodeId: "neuron_cluster",
      path: "neuron_cluster",
      label: "Neuron Cluster",
      clusterDiagram: {
        columns: 2,
        rows: 1,
        instantiated: 1,
        capacityTotal: 2,
        maxSteps: null,
        growthThreshold: null,
        hasColumnOverflow: false,
        hasRowOverflow: false,
        hasPlaneOverflow: false,
        planes: [
          {
            z: 1,
            cells: [
              {
                x: 1,
                y: 1,
                filled: true,
                title: "Neuron (1, 1, 1) — active",
                reach: {
                  position: [1, 1, 1],
                  connections: [
                    [1, 1, 1],
                    [2, 1, 1],
                  ],
                  inBoundsConnections: [
                    [1, 1, 1],
                    [2, 1, 1],
                  ],
                  activeConnectionTotal: 0,
                  emptyConnectionTotal: 1,
                  outOfBoundsTotal: 0,
                },
              },
              { x: 2, y: 1, filled: false, title: "Neuron (2, 1, 1) — empty" },
            ],
          },
        ],
      },
      height: 190,
    });

    const diagram = screen.getByTestId("cluster-diagram-neuron_cluster");
    const source = within(diagram).getByLabelText(/Neuron \(1, 1, 1\).*active/i);
    const emptyReach = within(diagram).getByLabelText(/Neuron \(2, 1, 1\).*empty/i);

    fireEvent.mouseEnter(source);
    expect(emptyReach).toHaveClass("bg-cyan/15");

    fireEvent.mouseLeave(source);

    expect(within(diagram).getByTestId("cluster-diagram-summary-neuron_cluster"))
      .toHaveTextContent("1 / 2");
    expect(source).not.toHaveClass("ring-2");
    expect(emptyReach).not.toHaveClass("bg-cyan/15");
  });

  it("does not activate the cluster reach overlay for empty cells", () => {
    renderGraphNode({
      nodeId: "neuron_cluster",
      path: "neuron_cluster",
      label: "Neuron Cluster",
      clusterDiagram: {
        columns: 2,
        rows: 1,
        instantiated: 1,
        capacityTotal: 2,
        maxSteps: null,
        growthThreshold: null,
        hasColumnOverflow: false,
        hasRowOverflow: false,
        hasPlaneOverflow: false,
        planes: [
          {
            z: 1,
            cells: [
              {
                x: 1,
                y: 1,
                filled: true,
                title: "Neuron (1, 1, 1) — active",
                reach: {
                  position: [1, 1, 1],
                  connections: [
                    [1, 1, 1],
                    [2, 1, 1],
                  ],
                  inBoundsConnections: [
                    [1, 1, 1],
                    [2, 1, 1],
                  ],
                  activeConnectionTotal: 0,
                  emptyConnectionTotal: 1,
                  outOfBoundsTotal: 0,
                },
              },
              { x: 2, y: 1, filled: false, title: "Neuron (2, 1, 1) — empty" },
            ],
          },
        ],
      },
      height: 190,
    });

    const diagram = screen.getByTestId("cluster-diagram-neuron_cluster");
    const emptyCell = within(diagram).getByLabelText(/Neuron \(2, 1, 1\).*empty/i);

    fireEvent.mouseEnter(emptyCell);

    expect(within(diagram).getByTestId("cluster-diagram-summary-neuron_cluster"))
      .toHaveTextContent("1 / 2");
    expect(emptyCell).not.toHaveClass("bg-cyan/15");
  });

  it("renders expert cells above the sampler and replaces child summaries", () => {
    renderGraphNode({
      childSummaries: [{ label: "LinearLayer", kind: "child" }],
      expertDiagram: {
        samplerLabel: "Sampler",
        samplerTitle: "main_model.0.sampler",
        totalExperts: 3,
        hasOverflow: false,
        cells: [
          { label: "E0", title: "Expert 0", kind: "expert", expertIndex: 0 },
          { label: "E1", title: "Expert 1", kind: "expert", expertIndex: 1 },
          { label: "E2", title: "Expert 2", kind: "expert", expertIndex: 2 },
        ],
      },
      height: 194,
    });

    const diagram = screen.getByTestId("expert-diagram-main_model.0");
    const expert0 = within(diagram).getByTitle("Expert 0");
    expect(expert0).toHaveClass(
      "h-8",
      "rounded-control-md",
      "type-label",
      "leading-none",
    );
    expect(within(expert0).getByText("E0")).toHaveClass("truncate");
    expect(expert0).toHaveTextContent("E0");
    expect(within(diagram).getByTitle("main_model.0.sampler")).toHaveClass(
      "h-8",
      "rounded-control-md",
      "px-2",
      "type-label",
    );
    expect(Number.parseFloat(within(diagram).getByTitle("main_model.0.sampler").style.width))
      .toBe(EXPERT_DIAGRAM_SAMPLER_WIDTH);
    expect(within(diagram).getByTitle("main_model.0.sampler")).toHaveTextContent("Sampler");
    expect(within(diagram).queryByText("LinearLayer")).not.toBeInTheDocument();
  });

  it("renders a non-interactive SVG connector for visible expert cells", () => {
    const { container } = renderGraphNode({
      expertDiagram: {
        samplerLabel: "Shared sampler",
        samplerTitle: "main_model.shared_sampler",
        totalExperts: 8,
        hasOverflow: true,
        cells: [
          { label: "E0", title: "Expert 0", kind: "expert", expertIndex: 0 },
          { label: "E1", title: "Expert 1", kind: "expert", expertIndex: 1 },
          { label: "E2", title: "Expert 2", kind: "expert", expertIndex: 2 },
          { label: "E3", title: "Expert 3", kind: "expert", expertIndex: 3 },
          { label: "E4", title: "Expert 4", kind: "expert", expertIndex: 4 },
          { label: "…", title: "3 more experts", kind: "overflow" },
          { label: "8 experts", title: "8 experts total", kind: "total" },
        ],
      },
      height: 194,
    });

    const connector = container
      .querySelector('[data-testid="expert-diagram-main_model.0"]')
      ?.querySelector("svg");
    expect(connector).toHaveClass("pointer-events-none");
    expect(connector?.querySelectorAll("path")).toHaveLength(7);
  });

  it("keeps expert diagram cells visual-only while card activation still works", () => {
    const onActivateNode = vi.fn();
    renderGraphNode({
      onActivateNode,
      expertDiagram: {
        samplerLabel: "Sampler",
        samplerTitle: "main_model.0.sampler",
        totalExperts: 1,
        hasOverflow: false,
        cells: [{ label: "E0", title: "Expert 0", kind: "expert", expertIndex: 0 }],
      },
      height: 194,
    });

    expect(screen.getByRole("button", { name: /select and expand main_model\.0/i }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^expand tree main_model\.0$/i }))
      .toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", {
        name: /select and expand main_model\.0/i,
      }),
    );
    expect(onActivateNode).toHaveBeenCalledTimes(1);
  });

  it("renders simple mode cards with compact metrics and footer controls", () => {
    renderGraphNode({
      graphDetailMode: "simple",
      parameterCount: 33024,
      childCount: 2,
      childSummaries: [
        { label: "LinearLayer", dims: "128 -> 128", kind: "child" },
        { label: "Gate", kind: "mechanism" },
      ],
      details: {
        dims: "128 -> 128",
        activation: "GELU",
        weightShape: "128 x 128",
      },
      stackDiagram: {
        totalLayers: 1,
        hasOverflow: false,
        cells: [
          {
            label: "Layer 0 · LinearLayer",
            title: "Layer 0 · LinearLayer · 128 -> 128",
            dims: "128 -> 128",
            kind: "layer",
            layerIndex: 0,
          },
        ],
      },
      config: {
        typeName: "LayerConfig",
        fields: [{ key: "dropout", value: 0.2 }],
      },
      isDetailsExpanded: true,
      height: graphCardGeometry.simpleHeight,
    });

    const card = screen.getByTestId("graph-node-card-main_model.0");
    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const metrics = screen.getByTestId("graph-node-simple-metrics-main_model.0");
    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");
    const footerStats = within(actionBar).getByTestId("graph-node-footer-stats-main_model.0");

    expect(within(card).getByText("Layer")).toBeInTheDocument();
    expect(within(titleRow).queryByRole("button")).not.toBeInTheDocument();
    expect(within(metrics).queryByTitle("33,024 parameters")).not.toBeInTheDocument();
    expect(within(metrics).getByTitle("input/output: 128 -> 128")).toHaveTextContent(
      "128 -> 128",
    );
    expect(within(footerStats).getByTitle("33,024 parameters")).toHaveTextContent("33K params");
    expect(within(footerStats).getByText("2 children")).toBeInTheDocument();
    expect(within(actionBar).getByRole("button", { name: /^expand tree main_model\.0$/i }))
      .toBeInTheDocument();
    expect(within(actionBar).getByRole("button", { name: /^open component info for main_model\.0$/i }))
      .toBeInTheDocument();
    expect(within(card).queryByText("main_model.0")).not.toBeInTheDocument();
    expect(screen.queryByTestId("parameter-shapes-main_model.0")).not.toBeInTheDocument();
    expect(screen.queryByTestId("stack-diagram-main_model.0")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child-summaries-main_model.0")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /config options for main_model\.0/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /details for main_model\.0/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("activation")).not.toBeInTheDocument();
    expect(screen.queryByText("GELU")).not.toBeInTheDocument();
    expect(screen.queryByText("weightShape")).not.toBeInTheDocument();
  });

  it("renders stack-derived dims in simple stack container metrics", () => {
    renderGraphNode({
      nodeId: "main_model",
      label: "Sequential",
      subtitle: "Main Model · main_model",
      path: "main_model",
      graphDetailMode: "simple",
      details: {},
      config: null,
      parameterCount: 65792,
      childCount: 4,
      stackDiagram: {
        dims: "256 -> 10",
        totalLayers: 4,
        hasOverflow: false,
        cells: [
          {
            label: "Layer 0 · LinearLayer",
            title: "Layer 0 · LinearLayer · 256 -> 256",
            dims: "256 -> 256",
            kind: "layer",
            layerIndex: 0,
          },
        ],
      },
      isDetailsExpanded: true,
      height: graphCardGeometry.simpleHeight,
    });

    const card = screen.getByTestId("graph-node-card-main_model");
    const metrics = screen.getByTestId("graph-node-simple-metrics-main_model");
    const footerStats = screen.getByTestId("graph-node-footer-stats-main_model");
    expect(within(metrics).queryByTitle("65,792 parameters")).not.toBeInTheDocument();
    expect(within(metrics).getByTitle("input/output: 256 -> 10")).toHaveTextContent("256 -> 10");
    expect(within(footerStats).getByTitle("65,792 parameters")).toHaveTextContent("65.8K params");
    expect(within(footerStats).getByText("4 children")).toBeInTheDocument();
    expect(screen.queryByTestId("stack-diagram-main_model")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child-summaries-main_model")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /details for main_model/i }))
      .not.toBeInTheDocument();
    expect(within(card).getByRole("button", { name: /^open component info for main_model$/i }))
      .toBeInTheDocument();
  });

  it("keeps simple mode footer controls limited to existing applicable actions", () => {
    renderGraphNode({
      graphDetailMode: "simple",
      canOpenMonitor: true,
      onOpenMonitor: vi.fn(),
      details: { activation: "GELU" },
      config: {
        typeName: "LayerConfig",
        fields: [{ key: "dropout", value: 0.2 }],
      },
      parameterActivity: {
        targetPath: "main_model.0.model",
        weights: {
          status: "updated",
          source: "active-job",
          sourceLabel: "active job job-1",
          observedPoints: 2,
        },
      },
      height: graphCardGeometry.simpleHeight,
    });

    const actionBar = screen.getByTestId("graph-node-action-bar-main_model.0");

    expect(within(actionBar).getByRole("button", { name: /^expand tree main_model\.0$/i }))
      .toBeInTheDocument();
    expect(within(actionBar).getByRole("button", { name: /^open component info for main_model\.0$/i }))
      .toBeInTheDocument();
    expect(within(actionBar).getByTestId("graph-node-footer-stats-main_model.0"))
      .toHaveTextContent("1 child");
    expect(within(actionBar).queryByTestId("graph-parameter-indicators"))
      .not.toBeInTheDocument();
    expect(within(actionBar).getByRole("button", { name: /^open monitor charts for main_model\.0$/i }))
      .toBeInTheDocument();
    expect(within(actionBar).queryByRole("button", { name: /details for main_model\.0/i }))
      .not.toBeInTheDocument();
    expect(within(actionBar).queryByRole("button", { name: /config options for main_model\.0/i }))
      .not.toBeInTheDocument();
  });

  it("keeps simple mode compact metrics from wrapping around long titles", () => {
    const longLabel = "Layer With An Extremely Long Display Name That Should Truncate";
    renderGraphNode({
      graphDetailMode: "simple",
      label: longLabel,
      parameterCount: 1234567,
      details: {
        inputDim: 256,
        outputDim: 512,
      },
      height: graphCardGeometry.simpleHeight,
    });

    const card = screen.getByTestId("graph-node-card-main_model.0");
    const title = within(card).getByText(longLabel);
    const metrics = screen.getByTestId("graph-node-simple-metrics-main_model.0");
    const footerStats = screen.getByTestId("graph-node-footer-stats-main_model.0");
    const params = within(footerStats).getByTitle("1,234,567 parameters");
    const dims = within(metrics).getByTitle("input/output: 256 -> 512");

    expect(title).toHaveClass("min-w-0", "truncate");
    expect(metrics).toHaveClass("h-5", "overflow-hidden");
    expect(params).toHaveTextContent("1.2M params");
    expect(params).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(dims).toHaveTextContent("256 -> 512");
    expect(dims).toHaveClass("shrink-0", "whitespace-nowrap");
  });
});

describe("SelectedNodeDetails", () => {
  it("uses typeName for the selected node title instead of API label", () => {
    const node: GraphNode = {
      id: "main_model.block_model",
      label: "Block Model",
      typeName: "LayerStack",
      path: "main_model.block_model",
      graphRole: "architecture",
      parameterCount: 0,
      parameterSizeBytes: 0,
      details: {},
      config: null,
    };

    render(<SelectedNodeDetails node={node} activeTrainingJob={undefined} />);

    expect(screen.getByText("LayerStack")).toBeInTheDocument();
    expect(screen.getAllByText("main_model.block_model")).toHaveLength(2);
    expect(screen.queryByText("Block Model")).not.toBeInTheDocument();
  });

  it("uses config fields instead of raw preview details", () => {
    const node: GraphNode = {
      id: "main_model.0.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      description: "Applies a learned linear projection.",
      path: "main_model.0.model",
      graphRole: "architecture",
      parameterCount: 512,
      parameterSizeBytes: 2048,
      details: {
        dims: "128 -> 64",
        weightShape: "128 x 64",
        biasShape: "64",
      },
      config: {
        typeName: "AdaptiveLinearLayerConfig",
        fields: [
          {
            key: "input_dim",
            value: 128,
            description: "Input feature dimension.",
          },
          { key: "output_dim", value: 64 },
          { key: "bias_flag", value: false },
          { key: "adaptive_augmentation_config", value: null },
        ],
      },
    };

    render(<SelectedNodeDetails node={node} activeTrainingJob={undefined} />);

    expect(screen.getByText("bias_flag")).toBeInTheDocument();
    expect(screen.getByText("adaptive_augmentation_config")).toBeInTheDocument();
    expect(screen.getByText("dims: 128 -> 64")).toBeInTheDocument();
    expect(screen.getByText("None")).toBeInTheDocument();
    expect(screen.queryByText("weightShape")).not.toBeInTheDocument();
    expect(screen.queryByText("biasShape")).not.toBeInTheDocument();
    expect(screen.queryByText("shapeTransition")).not.toBeInTheDocument();
  });

  it("displays layer and recurrent gate option details", () => {
    const node: GraphNode = {
      id: "main_model.0",
      label: "RecurrentLayer",
      typeName: "RecurrentLayer",
      path: "main_model.0",
      graphRole: "architecture",
      parameterCount: 0,
      parameterSizeBytes: 0,
      details: {
        gateOption: "MULTIPLIER",
        recurrent: {
          maxSteps: 4,
          gateOption: "MULTIPLIER",
        },
      },
      config: {
        typeName: "LayerConfig",
        fields: [
          { key: "gate_config", value: "GateConfig" },
          { key: "activation", value: "RELU" },
        ],
      },
    };

    render(<SelectedNodeDetails node={node} activeTrainingJob={undefined} />);

    expect(screen.getByText("gate_config")).toBeInTheDocument();
    expect(screen.getByText("GateConfig")).toBeInTheDocument();
    expect(screen.getByText("gateOption")).toBeInTheDocument();
    expect(screen.getByText("MULTIPLIER")).toBeInTheDocument();
    expect(screen.getByText("recurrent")).toBeInTheDocument();
    expect(
      screen.getByText('{"maxSteps":4,"gateOption":"MULTIPLIER"}'),
    ).toBeInTheDocument();
  });
});
