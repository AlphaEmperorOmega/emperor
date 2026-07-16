import { type ComponentType, type CSSProperties, type ReactNode } from "react";
import { expect } from "vitest";

type TestFlowNode = {
  id: string;
  type?: string;
  position: { x: number; y: number };
  selected?: boolean;
  style?: { height?: number };
  data: unknown;
};

type TestNodeRenderer = ComponentType<{
  data: unknown;
  selected: boolean;
}>;

export function ReactFlowProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

export function ReactFlow({
  children,
  edges,
  elementsSelectable,
  nodeClickDistance,
  nodeTypes,
  nodes,
  nodesConnectable,
  nodesDraggable,
  nodesFocusable,
  onMoveEnd,
  onMoveStart,
  onNodeClick,
  onlyRenderVisibleElements,
}: {
  children?: ReactNode;
  edges: Array<{ id: string; source: string; target: string }>;
  elementsSelectable?: boolean;
  nodeClickDistance?: number;
  nodeTypes?: Record<string, TestNodeRenderer>;
  nodes: TestFlowNode[];
  nodesConnectable?: boolean;
  nodesDraggable?: boolean;
  nodesFocusable?: boolean;
  onMoveEnd?: () => void;
  onMoveStart?: () => void;
  onNodeClick?: (event: unknown, node: TestFlowNode) => void;
  onlyRenderVisibleElements?: boolean;
}) {
  expect(nodesDraggable).toBe(false);
  expect(nodesConnectable).toBe(false);
  expect(elementsSelectable).toBe(false);
  expect(nodesFocusable).toBe(false);
  expect(nodeClickDistance).toBe(4);

  return (
    <div
      data-testid="flow"
      data-only-render-visible-elements={
        onlyRenderVisibleElements ? "true" : "false"
      }
      data-has-move-handlers={onMoveStart && onMoveEnd ? "true" : "false"}
    >
      {nodes.map((node) => {
        const NodeRenderer = node.type ? nodeTypes?.[node.type] : undefined;
        const dataHeight =
          typeof node.data === "object" &&
          node.data !== null &&
          "height" in node.data &&
          typeof node.data.height === "number"
            ? node.data.height
            : undefined;
        return (
          <div
            key={node.id}
            data-testid={`node-${node.id}`}
            data-x={node.position.x}
            data-y={node.position.y}
            data-height={node.style?.height ?? dataHeight}
            onClick={(event) => onNodeClick?.(event, node)}
          >
            {NodeRenderer ? (
              <NodeRenderer data={node.data} selected={Boolean(node.selected)} />
            ) : null}
          </div>
        );
      })}
      {edges.map((edge) => (
        <div key={edge.id} data-testid={`edge-${edge.id}`}>
          {edge.source} to {edge.target}
        </div>
      ))}
      {children}
    </div>
  );
}

export function Background() {
  return null;
}

export function Controls() {
  return null;
}

export function Panel({
  children,
  className,
  position,
  style,
}: {
  children?: ReactNode;
  className?: string;
  position?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      data-testid={`flow-panel-${position ?? "default"}`}
      data-position={position}
      className={className}
      style={style}
    >
      {children}
    </div>
  );
}

export function Handle() {
  return null;
}

export const MarkerType = { ArrowClosed: "arrowclosed" };
export const Position = { Left: "left", Right: "right" };
