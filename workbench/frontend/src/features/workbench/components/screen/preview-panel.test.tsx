import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PreviewPanel } from "@/features/workbench/components/screen/preview-panel";
import { type InspectResponse } from "@/lib/api";

const previewState = vi.hoisted(() => ({
  graph: undefined as InspectResponse | undefined,
}));

vi.mock("next/dynamic", () => ({
  default: () =>
    function DynamicGraphCanvas() {
      return <div data-testid="dynamic-graph-canvas" />;
    },
}));

vi.mock("@/features/workbench/providers/workbench-providers", () => ({
  useModelTargetConfig: () => ({
    selectedModelType: "linears",
    selectedTargetMode: "preset",
    selectedExperimentRunId: null,
  }),
  useHistoricalRuns: () => ({
    selectedLogRun: undefined,
    selectedLogRunMonitorEligibility: "unknown",
  }),
  useGraphView: () => ({
    graph: previewState.graph,
    graphForDetail: previewState.graph,
    nodes: [],
    edges: [],
    selectedNodeId: null,
    previewInspection: {
      isBuilding: false,
      isError: false,
      error: null,
    },
    isParameterStatusLoading: false,
    isParameterStatusPathMismatch: false,
    setSelectedNodeId: vi.fn(),
    revealGraphNode: vi.fn(),
    openCluster3d: vi.fn(),
  }),
}));

vi.mock(
  "@/features/workbench/state/graph-monitor/use-parameter-activity-minimap-state",
  () => ({
    useParameterActivityMinimapState: () => undefined,
  }),
);

describe("PreviewPanel module boundary", () => {
  beforeEach(() => {
    previewState.graph = undefined;
  });

  it("mounts the dynamically loaded graph canvas only when a graph exists", () => {
    const view = render(<PreviewPanel />);

    expect(screen.queryByTestId("dynamic-graph-canvas")).not.toBeInTheDocument();

    previewState.graph = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [],
      edges: [],
    };
    view.rerender(<PreviewPanel />);

    expect(screen.getByTestId("dynamic-graph-canvas")).toBeInTheDocument();
  });
});
