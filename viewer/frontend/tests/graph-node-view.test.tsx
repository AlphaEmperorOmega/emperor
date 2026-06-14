import { fireEvent, render, screen, within } from "@testing-library/react";
import type React from "react";
import { describe, expect, it, vi } from "vitest";
import {
  GraphNodeRenderModeProvider,
  nodeTypes,
} from "@/features/viewer/components/graph/graph-node-view";
import { SelectedNodeDetails } from "@/features/viewer/components/graph/selected-node-details";
import type { GraphNode } from "@/lib/api";
import type { ViewerNodeData } from "@/lib/graph";
import {
  CLUSTER_DIAGRAM_CELL_HEIGHT,
  CLUSTER_DIAGRAM_HEADER_HEIGHT,
  CLUSTER_DIAGRAM_CELL_GAP,
  SIMPLE_NODE_HEIGHT,
} from "@/lib/graph/constants";
import { EXPERT_DIAGRAM_SAMPLER_WIDTH } from "@/features/viewer/components/graph/graph-node-diagram-layout";

vi.mock("@xyflow/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@xyflow/react")>();
  return {
    ...actual,
    Handle: () => null,
  };
});

function renderGraphNode(
  data: Partial<ViewerNodeData> = {},
  options: { isViewportMoving?: boolean } = {},
) {
  const GraphNode = nodeTypes.viewerNode as unknown as React.ComponentType<{
    data: ViewerNodeData;
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
          subtitle: "main_model.0",
          path: "main_model.0",
          parameterCount: 0,
          parameterSizeBytes: 0,
          details: {},
          config: null,
          childCount: 1,
          childSummaries: [{ label: "LinearLayer", kind: "child" }],
          graphDetailMode: "basic",
          height: 148,
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

    expect(screen.getByRole("button", { name: /select and expand main_model\.0/i })).toHaveClass(
      "nodrag",
      "nopan",
      "edge",
      "px-8",
      "pb-4",
      "pt-4",
    );
  });

  it("renders basic-mode parameter and child badges inline with the title", () => {
    renderGraphNode({
      parameterCount: 12500,
      childCount: 2,
      height: 126,
    });

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const title = within(titleRow).getByText("Layer");
    const params = within(titleRow).getByTitle("12,500 parameters");
    const children = within(titleRow).getByText("2 children");

    expect(title.parentElement).toHaveClass("min-w-0", "flex-1", "flex-nowrap");
    expect(title).toHaveClass("min-w-0", "flex-1", "truncate");
    expect(params).toHaveTextContent("12.5K");
    expect(params).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(children).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(screen.queryByTestId("graph-node-badges-main_model.0")).not.toBeInTheDocument();
    expect(screen.getByText("main_model.0")).toHaveClass("mt-1.5", "leading-5");
  });

  it("keeps full-mode parameter and child badges on a dedicated header row", () => {
    renderGraphNode({
      graphDetailMode: "full",
      parameterCount: 12500,
      childCount: 2,
      height: 154,
    });

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const badgeRow = screen.getByTestId("graph-node-badges-main_model.0");

    expect(within(titleRow).getByText("Layer")).toBeInTheDocument();
    expect(within(titleRow).queryByTitle("12,500 parameters")).not.toBeInTheDocument();
    expect(badgeRow).toHaveClass("mt-1", "h-6", "overflow-hidden");
    expect(within(badgeRow).getByTitle("12,500 parameters")).toHaveClass(
      "h-6",
      "whitespace-nowrap",
    );
    expect(within(badgeRow).getByTitle("12,500 parameters")).toHaveTextContent("12.5K");
    expect(within(badgeRow).getByText("2 children")).toHaveClass(
      "h-6",
      "whitespace-nowrap",
    );
    expect(screen.getByText("main_model.0")).toHaveClass("mt-1.5", "leading-5");
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

  it("keeps basic-mode inline badges from wrapping around long titles", () => {
    const longLabel = "Layer With An Extremely Long Display Name That Should Truncate";
    renderGraphNode({
      label: longLabel,
      parameterCount: 1234567,
      childCount: 12,
      height: 126,
    });

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const title = within(titleRow).getByText(longLabel);
    const params = within(titleRow).getByTitle("1,234,567 parameters");
    const children = within(titleRow).getByText("12 children");

    expect(title.parentElement).toHaveClass("flex-nowrap", "min-w-0");
    expect(title).toHaveClass("min-w-0", "flex-1", "truncate");
    expect(params).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(children).toHaveClass("shrink-0", "whitespace-nowrap");
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

    fireEvent.click(shell);

    expect(onActivateNode).toHaveBeenCalledTimes(1);
  });

  it("uses the chevron as the explicit expansion toggle", () => {
    const onActivateNode = vi.fn();
    const onToggleExpansion = vi.fn();
    renderGraphNode({ onActivateNode, onToggleExpansion });

    fireEvent.click(screen.getByRole("button", { name: /^expand tree main_model\.0$/i }));

    expect(onToggleExpansion).toHaveBeenCalledTimes(1);
    expect(onActivateNode).not.toHaveBeenCalled();
  });

  it("renders expansion on the left and monitor charts on the right", () => {
    renderGraphNode({
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

    const titleRow = screen.getByTestId("graph-node-title-row-main_model.0");
    const expandButton = within(titleRow).getByRole("button", {
      name: /^expand tree main_model\.0$/i,
    });
    const monitorButton = within(titleRow).getByRole("button", {
      name: /^open monitor charts for main_model\.0$/i,
    });
    const indicators = within(titleRow).getByTestId("graph-parameter-indicators");

    expect(titleRow.firstElementChild).toBe(expandButton);
    expect(indicators.compareDocumentPosition(monitorButton) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(titleRow.lastElementChild).toBe(monitorButton);
  });

  it("renders weight and bias activity indicators with hover pills", () => {
    const { container } = renderGraphNode({
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

    const weights = screen.getByLabelText("Weights parameter activity: updated");
    const bias = screen.getByLabelText("Bias parameter activity: mixed");
    expect(weights).toHaveClass("border-ok/35", "bg-ok/10", "text-ok");
    expect(bias).toHaveClass("border-amber/40", "bg-amber/[0.12]", "text-amber");

    fireEvent.focus(weights);

    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "This parameter was logged and at least one sampled point showed update or value-change evidence.",
    );

    fireEvent.blur(weights);
    fireEvent.focus(bias);

    const tooltip = screen.getByRole("tooltip");
    expect(container).not.toContainElement(tooltip);
    expect(tooltip).toHaveClass("fixed");
    expect(tooltip).toHaveTextContent("Bias");
    expect(tooltip).toHaveTextContent(
      "At least one historical run showed update evidence, but at least one other run did not.",
    );
    expect(tooltip).toHaveTextContent("2 historical runs");
    expect(tooltip).not.toHaveTextContent("main_model.0.model/bias/delta_norm");
    expect(tooltip).toHaveTextContent(
      "1 updated / 1 unchanged / 0 missing / 0 unknown",
    );
  });

  it("does not activate the node when activity indicators are clicked", () => {
    const onActivateNode = vi.fn();
    renderGraphNode({
      onActivateNode,
      parameterActivity: {
        targetPath: "main_model.0.model",
        weights: {
          status: "unchanged",
          source: "active-job",
          sourceLabel: "active job job-1",
          metric: "main_model.0.model/weights/delta_norm",
          lastStep: 4,
          observedPoints: 1,
        },
        bias: {
          status: "unknown",
          source: "active-job",
          sourceLabel: "active job job-1",
          observedPoints: 0,
        },
      },
    });

    fireEvent.click(screen.getByLabelText("Weights parameter activity: unchanged"));

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
    expect(screen.getByText("LinearLayer").parentElement).toHaveClass(
      "h-9",
      "rounded-[10px]",
      "border",
      "border-line-soft",
      "bg-white/[0.02]",
      "px-3",
      "text-[13px]",
      "leading-none",
    );
  });

  it("renders mechanism summary rows with explicit bottom edge styling", () => {
    renderGraphNode({
      childSummaries: [{ label: "Gate", kind: "mechanism" }],
    });

    expect(screen.getByText("Gate").parentElement).toHaveClass(
      "h-9",
      "rounded-[10px]",
      "border",
      "border-violet/30",
      "text-[13px]",
      "shadow-[inset_0_-1px_0_rgba(146,113,255,0.24)]",
    );
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
    expect(within(shapes).getByLabelText("W shape 128 x 128")).toHaveClass(
      "h-7",
      "rounded-[7px]",
      "border-violet/25",
      "bg-violet/15",
      "px-2",
      "text-[12px]",
      "leading-none",
      "shadow-[inset_0_-1px_0_rgba(146,113,255,0.24)]",
    );
    expect(within(shapes).getByLabelText("b shape 128")).toHaveClass("h-7", "text-[12px]");
    expect(within(shapes).getByText("W")).toBeInTheDocument();
    expect(within(shapes).getByText("W")).toHaveClass("truncate");
    expect(within(shapes).getByText("128 x 128")).toBeInTheDocument();
    expect(within(shapes).getByText("128 x 128")).toHaveClass("truncate");
    expect(within(shapes).getByText("b")).toBeInTheDocument();
    expect(within(shapes).getByText("128")).toBeInTheDocument();
  });

  it("renders layer dims on the inner-model child summary row", () => {
    renderGraphNode({
      childSummaries: [{ label: "LinearLayer", dims: "128 -> 64", kind: "child" }],
    });

    const summary = screen.getByLabelText("LinearLayer 128 -> 64");
    expect(summary).toHaveAttribute("title", "LinearLayer 128 -> 64");
    expect(summary).toHaveClass("h-9", "rounded-[10px]", "gap-2", "overflow-hidden");
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
      height: 372,
    });

    const detailsButton = screen.getByRole("button", {
      name: /config options for main_model\.0/i,
    });
    const details = document.getElementById(
      detailsButton.getAttribute("aria-controls") ?? "",
    );

    expect(screen.getByTestId("parameter-shapes-main_model.0")).toBeInTheDocument();
    expect(detailsButton).toHaveTextContent("Config options");
    expect(detailsButton).toHaveClass("h-9", "text-sm");
    expect(details).not.toBeNull();
    expect(within(details!).getByText("bias_flag").parentElement).toHaveClass(
      "h-8",
      "grid-cols-[96px_minmax(0,1fr)]",
      "text-[12.5px]",
    );
    expect(within(details!).getByText("adaptive_augmentation_config")).toBeInTheDocument();
    expect(within(details!).getByText("None")).toBeInTheDocument();
    expect(within(details!).queryByText("weightShape")).not.toBeInTheDocument();
    expect(within(details!).queryByText("biasShape")).not.toBeInTheDocument();
    expect(within(details!).queryByText("dims")).not.toBeInTheDocument();
    expect(within(details!).queryByText("shapeTransition")).not.toBeInTheDocument();
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
      height: 232,
    });

    const summaries = screen.getByText("LinearLayer").parentElement?.parentElement;
    expect(summaries).not.toHaveClass("flex-1");
    expect(screen.getByRole("button", { name: /details for main_model\.0/i })).toBeInTheDocument();
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
    expect(layer0).toHaveClass("text-[12px]");
    expect(within(layer0).getByText("Layer 0 · LinearLayer")).toHaveClass(
      "min-w-0",
      "flex-1",
      "truncate",
      "text-left",
    );
    expect(within(layer0).getByText("128 -> 128")).toHaveClass(
      "shrink-0",
      "text-right",
      "font-mono",
    );
    expect(Number.parseFloat(layer0.style.top)).toBeLessThan(
      Number.parseFloat(layer1.style.top),
    );
    expect(Number.parseFloat(layer1.style.top)).toBeLessThan(
      Number.parseFloat(layer2.style.top),
    );
    expect(within(diagram).queryByText("LinearLayer")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child-summaries-main_model.0")).not.toBeInTheDocument();
    expect(
      diagram.compareDocumentPosition(detailsButton) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("uses wide readable cells for five-layer stack previews", () => {
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
    expect(diagram).toHaveClass("h-[160px]");
    expect(Number.parseFloat(layer0.style.left)).toBe(0);
    expect(Number.parseFloat(layer0.style.width)).toBe(296);
    expect(Number.parseFloat(layer0.style.height)).toBe(24);
    expect(layer0).toHaveClass("rounded-[8px]", "px-2.5", "text-[12px]");
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
          {
            label: "Layer 2 · LinearLayer",
            title: "Layer 2 · LinearLayer",
            kind: "layer",
            layerIndex: 2,
          },
          {
            label: "Layer 3 · LinearLayer",
            title: "Layer 3 · LinearLayer",
            kind: "layer",
            layerIndex: 3,
          },
          {
            label: "Layer 4 · LinearLayer",
            title: "Layer 4 · LinearLayer",
            kind: "layer",
            layerIndex: 4,
          },
          { label: "...", title: "3 more layers", kind: "overflow" },
          { label: "8 layers", title: "8 layers total", kind: "total" },
        ],
      },
      height: 250,
    });

    const connector = container
      .querySelector('[data-testid="stack-diagram-main_model.0"]')
      ?.querySelector("svg");
    expect(connector).toBeNull();
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
    fireEvent.click(screen.getByText("128 -> 128"));
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
      CLUSTER_DIAGRAM_HEADER_HEIGHT +
      2 * CLUSTER_DIAGRAM_CELL_HEIGHT +
      CLUSTER_DIAGRAM_CELL_GAP;
    const planeWidth = 2 * CLUSTER_DIAGRAM_CELL_HEIGHT + CLUSTER_DIAGRAM_CELL_GAP;
    const plane = within(diagram).getByTitle("Z plane 1");
    const activeCell = within(diagram).getByLabelText(/Neuron \(1, 1, 1\).*active/i);
    expect(within(diagram).getByText("Cluster map")).toBeInTheDocument();
    expect(within(diagram).getByText("2 / 8")).toBeInTheDocument();
    expect(within(diagram).getByText("1z")).toHaveClass(
      "rounded-[7px]",
      "border-violet/25",
      "px-1.5",
      "py-1",
      "text-[10px]",
      "leading-none",
    );
    expect(within(diagram).getByText("clipped")).toHaveClass(
      "rounded-[7px]",
      "border-line-soft",
      "px-1.5",
      "py-1",
    );
    expect(Number.parseFloat(diagram.style.height)).toBe(headerHeight);
    expect(Number.parseFloat(plane.style.width)).toBe(planeWidth);
    expect(Number.parseFloat(activeCell.style.height)).toBe(CLUSTER_DIAGRAM_CELL_HEIGHT);
    expect(Number.parseFloat(activeCell.style.width)).toBe(CLUSTER_DIAGRAM_CELL_HEIGHT);
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
      "rounded-[8px]",
      "text-[12px]",
      "leading-none",
    );
    expect(within(expert0).getByText("E0")).toHaveClass("truncate");
    expect(expert0).toHaveTextContent("E0");
    expect(within(diagram).getByTitle("main_model.0.sampler")).toHaveClass(
      "h-8",
      "rounded-[8px]",
      "px-2",
      "text-[12px]",
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
          { label: "...", title: "3 more experts", kind: "overflow" },
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
    fireEvent.click(screen.getByText("E0"));
    expect(onActivateNode).toHaveBeenCalledTimes(1);
  });

  it("renders simple mode cards with inline metrics and the expansion chevron", () => {
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
      height: SIMPLE_NODE_HEIGHT,
    });

    const card = screen.getByRole("button", { name: /select and expand main_model\.0/i });
    expect(within(card).getByText("Layer")).toBeInTheDocument();
    expect(within(card).getByTitle("33,024 parameters")).toHaveTextContent("33K params");
    expect(within(card).getByTitle("input/output: 128 -> 128")).toHaveTextContent(
      "128 -> 128",
    );
    expect(screen.getByRole("button", { name: /^expand tree main_model\.0$/i }))
      .toBeInTheDocument();
    expect(within(card).queryByText("main_model.0")).not.toBeInTheDocument();
    expect(within(card).queryByText("2 children")).not.toBeInTheDocument();
    expect(screen.queryByTestId("parameter-shapes-main_model.0")).not.toBeInTheDocument();
    expect(screen.queryByTestId("stack-diagram-main_model.0")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child-summaries-main_model.0")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /config options for main_model\.0/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("activation")).not.toBeInTheDocument();
    expect(screen.queryByText("GELU")).not.toBeInTheDocument();
    expect(screen.queryByText("weightShape")).not.toBeInTheDocument();
  });

  it("renders stack-derived dims inline on simple stack container cards", () => {
    renderGraphNode({
      nodeId: "main_model",
      label: "Main Model",
      subtitle: "Sequential · main_model",
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
      height: SIMPLE_NODE_HEIGHT,
    });

    const card = screen.getByRole("button", { name: /select and expand main_model/i });
    expect(within(card).getByTitle("65,792 parameters")).toHaveTextContent("65.8K params");
    expect(within(card).getByTitle("input/output: 256 -> 10")).toHaveTextContent("256 -> 10");
    expect(screen.queryByTestId("stack-diagram-main_model")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child-summaries-main_model")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /details for main_model/i }))
      .not.toBeInTheDocument();
  });

  it("keeps simple mode inline metrics from wrapping around long titles", () => {
    const longLabel = "Layer With An Extremely Long Display Name That Should Truncate";
    renderGraphNode({
      graphDetailMode: "simple",
      label: longLabel,
      parameterCount: 1234567,
      details: {
        inputDim: 256,
        outputDim: 512,
      },
      height: SIMPLE_NODE_HEIGHT,
    });

    const card = screen.getByRole("button", { name: /select and expand main_model\.0/i });
    const title = within(card).getByText(longLabel);
    const params = within(card).getByTitle("1,234,567 parameters");
    const dims = within(card).getByTitle("input/output: 256 -> 512");

    expect(title.parentElement).toHaveClass("flex-nowrap", "min-w-0");
    expect(title).toHaveClass("min-w-0", "flex-1", "truncate");
    expect(params).toHaveTextContent("1.2M params");
    expect(params).toHaveClass("shrink-0", "whitespace-nowrap");
    expect(dims).toHaveTextContent("256 -> 512");
    expect(dims).toHaveClass("shrink-0", "whitespace-nowrap");
  });
});

describe("SelectedNodeDetails", () => {
  it("uses config fields instead of raw preview details", () => {
    const node: GraphNode = {
      id: "main_model.0.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
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
          { key: "input_dim", value: 128 },
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
});
