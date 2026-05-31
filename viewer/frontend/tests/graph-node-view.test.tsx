import { fireEvent, render, screen, within } from "@testing-library/react";
import type React from "react";
import { describe, expect, it, vi } from "vitest";
import { nodeTypes } from "@/components/viewer/graph-node-view";
import type { ViewerNodeData } from "@/lib/graph";

vi.mock("@xyflow/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@xyflow/react")>();
  return {
    ...actual,
    Handle: () => null,
  };
});

function renderGraphNode(data: Partial<ViewerNodeData> = {}) {
  const GraphNode = nodeTypes.viewerNode as React.ComponentType<{
    data: ViewerNodeData;
    selected: boolean;
  }>;

  return render(
    <GraphNode
      selected={false}
      data={{
        nodeId: "main_model.0",
        label: "Layer",
        subtitle: "main_model.0",
        path: "main_model.0",
        parameterCount: 0,
        details: {},
        childCount: 1,
        childSummaries: [{ label: "LinearLayer", kind: "child" }],
        height: 132,
        isExpanded: false,
        canToggleExpansion: true,
        isDetailsExpanded: false,
        onActivateNode: () => {},
        onToggleDetails: () => {},
        ...data,
      }}
    />,
  );
}

describe("GraphNodeView", () => {
  it("keeps explicit bottom padding around the card contents", () => {
    renderGraphNode();

    expect(screen.getByRole("button", { name: /select and expand main_model\.0/i })).toHaveClass(
      "px-3",
      "pb-4",
      "pt-3",
    );
  });

  it("renders child summary rows with explicit bottom edge styling", () => {
    renderGraphNode();

    const summaries = screen.getByText("LinearLayer").parentElement?.parentElement;
    expect(summaries).not.toHaveClass("overflow-hidden");
    expect(summaries).not.toHaveClass("flex-1");
    expect(screen.getByText("LinearLayer").parentElement).toHaveClass(
      "h-6",
      "rounded-md",
      "border",
      "border-subtle",
      "shadow-[inset_0_-1px_0_#d8ded9]",
    );
  });

  it("renders mechanism summary rows with explicit bottom edge styling", () => {
    renderGraphNode({
      childSummaries: [{ label: "Gate", kind: "mechanism" }],
    });

    expect(screen.getByText("Gate").parentElement).toHaveClass(
      "h-6",
      "rounded-md",
      "border",
      "border-accent-edge",
      "shadow-[inset_0_-1px_0_#b9cfc7]",
    );
  });

  it("uses an icon between nested child summary labels", () => {
    renderGraphNode({
      childSummaries: [{ label: "Layer", nestedLabel: "LinearLayer", kind: "child" }],
    });

    const relationshipRow = screen.getByLabelText("Layer LinearLayer");
    expect(within(relationshipRow).getByText("Layer")).toBeInTheDocument();
    expect(within(relationshipRow).getByText("LinearLayer")).toBeInTheDocument();
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
      details: { dims: "128 -> 128" },
      isDetailsExpanded: true,
      height: 162,
    });

    const summaries = screen.getByText("LinearLayer").parentElement?.parentElement;
    expect(summaries).not.toHaveClass("flex-1");
    expect(screen.getByRole("button", { name: /details for main_model\.0/i })).toBeInTheDocument();
  });

  it("renders stack cells above details and replaces child summaries", () => {
    renderGraphNode({
      details: { dims: "128 -> 128" },
      childSummaries: [{ label: "LinearLayer", kind: "child" }],
      stackDiagram: {
        totalLayers: 3,
        hasOverflow: false,
        cells: [
          {
            label: "Layer 0 · LinearLayer",
            title: "Layer 0 · LinearLayer · 128 -> 128",
            kind: "layer",
            layerIndex: 0,
          },
          {
            label: "Layer 1 · LinearLayer",
            title: "Layer 1 · LinearLayer · 128 -> 128",
            kind: "layer",
            layerIndex: 1,
          },
          {
            label: "Layer 2 · LinearLayer",
            title: "Layer 2 · LinearLayer · 128 -> 128",
            kind: "layer",
            layerIndex: 2,
          },
        ],
      },
      height: 194,
    });

    const diagram = screen.getByTestId("stack-diagram-main_model.0");
    const detailsButton = screen.getByRole("button", { name: /details for main_model\.0/i });
    const layer0 = within(diagram).getByTitle("Layer 0 · LinearLayer · 128 -> 128");
    const layer1 = within(diagram).getByTitle("Layer 1 · LinearLayer · 128 -> 128");
    const layer2 = within(diagram).getByTitle("Layer 2 · LinearLayer · 128 -> 128");
    expect(layer0).toHaveTextContent("Layer 0 · LinearLayer");
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

  it("renders a non-interactive SVG connector for stack cells", () => {
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
      height: 178,
    });

    const connector = container
      .querySelector('[data-testid="stack-diagram-main_model.0"]')
      ?.querySelector("svg");
    expect(connector).toHaveClass("pointer-events-none");
    expect(connector?.querySelectorAll("path")).toHaveLength(8);
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
            kind: "layer",
            layerIndex: 0,
          },
        ],
      },
      height: 178,
    });

    expect(screen.getAllByRole("button")).toHaveLength(1);
    fireEvent.click(screen.getByText("Layer 0 · LinearLayer"));
    expect(onActivateNode).toHaveBeenCalledTimes(1);
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
      height: 178,
    });

    const diagram = screen.getByTestId("expert-diagram-main_model.0");
    expect(within(diagram).getByTitle("Expert 0")).toHaveTextContent("E0");
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
      height: 178,
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
      height: 178,
    });

    expect(screen.getAllByRole("button")).toHaveLength(1);
    fireEvent.click(screen.getByText("E0"));
    expect(onActivateNode).toHaveBeenCalledTimes(1);
  });
});
