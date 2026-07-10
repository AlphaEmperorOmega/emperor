import dynamic from "next/dynamic";
import {
  AlertTriangle,
  Box,
  Eye,
  EyeOff,
  LocateFixed,
  RotateCcw,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  Component,
  type ReactNode,
} from "react";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { SectionHeading } from "@/components/ui/section-heading";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import {
  useActiveTrainingJob,
  useGraphView,
} from "@/features/workbench/providers/workbench-providers";
import {
  buildCluster3DSceneModel,
  type Cluster3DCell,
  type Cluster3DSceneModel,
  formatGraphCoordinate,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

const DynamicNeuronCluster3DScene = dynamic(
  () =>
    import("@/features/workbench/components/graph/neuron-cluster-3d-scene").then(
      (module) => module.NeuronCluster3DScene,
    ),
  {
    ssr: false,
    loading: () => (
      <div className="grid h-full min-h-[360px] place-items-center text-sm text-ink-faint">
        Loading 3D view
      </div>
    ),
  },
);

type AxisKey = "x" | "y" | "z";

type AxisControls = {
  values: number[];
  visible: Set<number>;
  toggle: (value: number) => void;
};

type SceneErrorBoundaryProps = {
  fallback: ReactNode;
  children: ReactNode;
};

type SceneErrorBoundaryState = {
  hasError: boolean;
};

class SceneErrorBoundary extends Component<
  SceneErrorBoundaryProps,
  SceneErrorBoundaryState
> {
  state: SceneErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

function axisValues(count: number) {
  return Array.from({ length: count }, (_, index) => index + 1);
}

function allVisibleSet(count: number) {
  return new Set(axisValues(count));
}

function toggleVisibleSet(current: Set<number>, value: number) {
  const next = new Set(current);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

function cellVisible(
  cell: Cluster3DCell,
  visibleX: Set<number>,
  visibleY: Set<number>,
  visibleZ: Set<number>,
) {
  const [x, y, z] = cell.coordinate;
  return visibleX.has(x) && visibleY.has(y) && visibleZ.has(z);
}

function AxisSliceControls({
  axis,
  controls,
}: {
  axis: AxisKey;
  controls: AxisControls;
}) {
  const label = axis.toUpperCase();

  return (
    <section aria-label={`${label} slices`} className="grid gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[11px] font-bold uppercase text-ink-faint">
          {label} slices
        </span>
        <span className="font-mono text-[11px] text-ink-faint">
          {controls.visible.size}/{controls.values.length}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {controls.values.map((value) => {
          const isVisible = controls.visible.has(value);
          return (
            <button
              key={`${axis}-${value}`}
              type="button"
              aria-label={`${isVisible ? "Hide" : "Show"} ${label} slice ${value}`}
              aria-pressed={isVisible}
              title={`${label} ${value}`}
              onClick={() => controls.toggle(value)}
              className={cn(
                "grid h-7 min-w-7 place-items-center rounded-[7px] border px-2 font-mono text-[11px] font-bold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                isVisible
                  ? "border-cyan-200/45 bg-cyan-300/[0.12] text-cyan-50"
                  : "border-line bg-black/25 text-ink-faint hover:border-line-soft hover:text-ink",
              )}
            >
              {value}
            </button>
          );
        })}
      </div>
    </section>
  );
}

function CellDetail({
  scene,
  selectedCell,
}: {
  scene: Cluster3DSceneModel;
  selectedCell: Cluster3DCell | null;
}) {
  if (!selectedCell) {
    return (
      <div className="rounded-[8px] border border-line bg-black/20 p-3 text-sm text-ink-faint">
        Select a cell to inspect its coordinate and graph mapping.
      </div>
    );
  }

  const reach = selectedCell.reach;
  return (
    <div className="rounded-[8px] border border-cyan-200/25 bg-cyan-300/[0.06] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-bold text-ink">
          {formatGraphCoordinate(selectedCell.coordinate)}
        </span>
        <StatChip tone="soft">{selectedCell.category}</StatChip>
        {selectedCell.isOverlayOnly && <StatChip tone="soft">growth overlay</StatChip>}
      </div>
      <div className="mt-2 grid gap-1.5 text-xs text-ink-faint">
        <div>
          Node:{" "}
          <span className="font-mono text-ink-dim">
            {selectedCell.nodeMatch?.nodePath ?? scene.clusterNodePath}
          </span>
        </div>
        {reach && (
          <div>
            Reach: {reach.activeConnectionTotal} active, {reach.emptyConnectionTotal} empty,{" "}
            {reach.outOfBoundsTotal} out of bounds
          </div>
        )}
        {!selectedCell.nodeMatch && (
          <div>This coordinate is not present as a descendant graph node yet.</div>
        )}
      </div>
    </div>
  );
}

function Fallback2DPanel({
  scene,
  selectedKey,
  visibleX,
  visibleY,
  visibleZ,
  onSelectCell,
}: {
  scene: Cluster3DSceneModel;
  selectedKey: string | null;
  visibleX: Set<number>;
  visibleY: Set<number>;
  visibleZ: Set<number>;
  onSelectCell: (cell: Cluster3DCell) => void;
}) {
  const visibleCells = scene.activeCells.filter((cell) =>
    cellVisible(cell, visibleX, visibleY, visibleZ),
  );

  return (
    <div className="grid h-full min-h-[360px] content-start gap-3 overflow-auto p-4">
      <div className="flex items-center gap-2 text-sm text-amber">
        <AlertTriangle className="h-4 w-4" aria-hidden />
        WebGL is unavailable, showing coordinate list
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(104px,1fr))] gap-2">
        {visibleCells.map((cell) => (
          <button
            key={cell.key}
            type="button"
            aria-label={`Select coordinate ${formatGraphCoordinate(cell.coordinate)}`}
            onClick={() => onSelectCell(cell)}
            className={cn(
              "rounded-[8px] border p-2 text-left font-mono text-[11px] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
              cell.key === selectedKey
                ? "border-amber-200/70 bg-amber-300/[0.16] text-amber-50"
                : "border-line bg-white/[0.035] text-ink-dim hover:border-cyan-200/35 hover:text-ink",
            )}
          >
            <span className="block font-bold">{formatGraphCoordinate(cell.coordinate)}</span>
            <span className="mt-1 block text-[10px] uppercase">{cell.category}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function CoordinateList({
  scene,
  selectedKey,
  visibleX,
  visibleY,
  visibleZ,
  onSelectCell,
}: {
  scene: Cluster3DSceneModel;
  selectedKey: string | null;
  visibleX: Set<number>;
  visibleY: Set<number>;
  visibleZ: Set<number>;
  onSelectCell: (cell: Cluster3DCell) => void;
}) {
  const visibleCells = scene.activeCells.filter((cell) =>
    cellVisible(cell, visibleX, visibleY, visibleZ),
  );

  return (
    <section aria-label="3D cluster coordinates" className="grid gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[11px] font-bold uppercase text-ink-faint">
          Coordinates
        </span>
        <span className="font-mono text-[11px] text-ink-faint">
          {visibleCells.length}/{scene.activeCells.length}
        </span>
      </div>
      <div className="grid max-h-44 gap-1.5 overflow-auto pr-1">
        {visibleCells.length === 0 ? (
          <div className="rounded-[7px] border border-line bg-black/20 px-2.5 py-2 text-xs text-ink-faint">
            No active coordinates are visible with the current slice filters.
          </div>
        ) : (
          visibleCells.map((cell) => {
            const isSelected = cell.key === selectedKey;
            return (
              <button
                key={cell.key}
                type="button"
                aria-pressed={isSelected}
                aria-label={`Select coordinate ${formatGraphCoordinate(cell.coordinate)} ${cell.category}`}
                onClick={() => onSelectCell(cell)}
                className={cn(
                  "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-[7px] border px-2.5 py-2 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                  isSelected
                    ? "border-amber-200/70 bg-amber-300/[0.16] text-amber-50"
                    : "border-line bg-white/[0.035] text-ink-dim hover:border-cyan-200/35 hover:text-ink",
                )}
              >
                <span className="font-mono text-[11px] font-bold">
                  {formatGraphCoordinate(cell.coordinate)}
                </span>
                <span className="rounded-[6px] border border-line-soft px-1.5 py-0.5 text-[10px] uppercase">
                  {cell.category}
                </span>
              </button>
            );
          })
        )}
      </div>
    </section>
  );
}

function NeuronCluster3DPopup({
  scene,
  onClose,
  onRevealCluster,
  onRevealNodeInFull,
}: {
  scene: Cluster3DSceneModel;
  onClose: () => void;
  onRevealCluster: (nodeId: string) => void;
  onRevealNodeInFull: (nodeId: string) => void;
}) {
  const [selectedKey, setSelectedKey] = useState<string | null>(
    scene.activeCells[0]?.key ?? null,
  );
  const [visibleX, setVisibleX] = useState(() => allVisibleSet(scene.capacity[0]));
  const [visibleY, setVisibleY] = useState(() => allVisibleSet(scene.capacity[1]));
  const [visibleZ, setVisibleZ] = useState(() => allVisibleSet(scene.capacity[2]));
  const selectedCell =
    scene.activeCells.find((cell) => cell.key === selectedKey) ?? null;

  useEffect(() => {
    setSelectedKey(scene.activeCells[0]?.key ?? null);
    setVisibleX(allVisibleSet(scene.capacity[0]));
    setVisibleY(allVisibleSet(scene.capacity[1]));
    setVisibleZ(allVisibleSet(scene.capacity[2]));
  }, [scene]);

  const resetSlices = useCallback(() => {
    setVisibleX(allVisibleSet(scene.capacity[0]));
    setVisibleY(allVisibleSet(scene.capacity[1]));
    setVisibleZ(allVisibleSet(scene.capacity[2]));
  }, [scene.capacity]);

  const isolateSelected = useCallback(() => {
    if (!selectedCell) {
      return;
    }
    const [x, y, z] = selectedCell.coordinate;
    setVisibleX(new Set([x]));
    setVisibleY(new Set([y]));
    setVisibleZ(new Set([z]));
  }, [selectedCell]);

  const handleSelectCell = useCallback(
    (cell: Cluster3DCell) => {
      setSelectedKey(cell.key);
      if (cell.nodeMatch) {
        onRevealNodeInFull(cell.nodeMatch.nodeId);
      } else {
        onRevealCluster(scene.clusterNodeId);
      }
    },
    [onRevealCluster, onRevealNodeInFull, scene.clusterNodeId],
  );

  const axisControls = useMemo(
    () => ({
      x: {
        values: axisValues(scene.capacity[0]),
        visible: visibleX,
        toggle: (value: number) => setVisibleX((current) => toggleVisibleSet(current, value)),
      },
      y: {
        values: axisValues(scene.capacity[1]),
        visible: visibleY,
        toggle: (value: number) => setVisibleY((current) => toggleVisibleSet(current, value)),
      },
      z: {
        values: axisValues(scene.capacity[2]),
        visible: visibleZ,
        toggle: (value: number) => setVisibleZ((current) => toggleVisibleSet(current, value)),
      },
    }),
    [scene.capacity, visibleX, visibleY, visibleZ],
  );

  const fallback = (
    <Fallback2DPanel
      scene={scene}
      selectedKey={selectedKey}
      visibleX={visibleX}
      visibleY={visibleY}
      visibleZ={visibleZ}
      onSelectCell={handleSelectCell}
    />
  );

  return (
    <DialogShell
      ariaLabel={`3D neuron cluster ${scene.clusterNodePath}`}
      size="fullscreen"
      onClose={onClose}
      panelClassName="h-[80vh] min-h-[80vh] max-h-[80vh] bg-[linear-gradient(180deg,rgba(14,16,24,0.98),rgba(8,10,16,0.98))]"
      header={
        <div className="flex min-w-0 items-center justify-between gap-3 border-b border-line px-4 py-3">
          <SectionHeading
            as="h2"
            icon={<Box className="h-[15px] w-[15px] text-cyan-100" aria-hidden />}
            title="3D Cluster"
          />
          <span className="min-w-0 flex-1 truncate font-mono text-xs text-ink-faint">
            {scene.clusterNodePath}
          </span>
          <button
            type="button"
            aria-label="Close 3D cluster view"
            title="Close"
            onClick={onClose}
            className="grid h-9 w-9 shrink-0 place-items-center rounded-[8px] border border-line bg-white/[0.035] text-ink-dim transition hover:border-line-soft hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>
      }
    >
      <div
        className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(0,1fr)_320px]"
        onPointerDown={(event) => event.stopPropagation()}
        onWheel={(event) => event.stopPropagation()}
      >
        <div className="relative min-h-[420px] overflow-hidden bg-black/20">
          <SceneErrorBoundary fallback={fallback}>
            <DynamicNeuronCluster3DScene
              scene={scene}
              visibleX={visibleX}
              visibleY={visibleY}
              visibleZ={visibleZ}
              selectedKey={selectedKey}
              onSelectCell={handleSelectCell}
            />
          </SceneErrorBoundary>
        </div>

        <aside className="grid min-h-0 content-start gap-4 overflow-auto border-t border-line bg-black/20 p-4 lg:border-l lg:border-t-0">
          <div className="grid grid-cols-2 gap-2">
            <StatChip tone="soft">{scene.activeCells.length} active</StatChip>
            <StatChip tone="soft">{scene.capacityTotal} capacity</StatChip>
            <StatChip tone="soft">{scene.initialCount} initial</StatChip>
            <StatChip tone="soft">{scene.grownCount} grown</StatChip>
            {scene.recentAddedCount > 0 && (
              <StatChip tone="soft">{scene.recentAddedCount} recent</StatChip>
            )}
            {!scene.renderGhostCells && <StatChip tone="soft">ghosts hidden</StatChip>}
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={resetSlices}
              className="inline-flex h-9 items-center gap-2 rounded-[8px] border border-line bg-white/[0.035] px-3 text-xs font-bold text-ink-dim transition hover:border-cyan-200/35 hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden />
              Reset
            </button>
            <button
              type="button"
              onClick={isolateSelected}
              disabled={!selectedCell}
              className="inline-flex h-9 items-center gap-2 rounded-[8px] border border-line bg-white/[0.035] px-3 text-xs font-bold text-ink-dim transition hover:border-amber-200/35 hover:text-ink disabled:cursor-not-allowed disabled:opacity-45 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              <LocateFixed className="h-3.5 w-3.5" aria-hidden />
              Isolate
            </button>
          </div>

          <AxisSliceControls axis="x" controls={axisControls.x} />
          <AxisSliceControls axis="y" controls={axisControls.y} />
          <AxisSliceControls axis="z" controls={axisControls.z} />

          <CoordinateList
            scene={scene}
            selectedKey={selectedKey}
            visibleX={visibleX}
            visibleY={visibleY}
            visibleZ={visibleZ}
            onSelectCell={handleSelectCell}
          />

          <div className="flex items-center gap-2 rounded-[8px] border border-line bg-white/[0.025] p-2 text-xs text-ink-faint">
            {scene.renderGhostCells ? (
              <Eye className="h-4 w-4 text-ink-dim" aria-hidden />
            ) : (
              <EyeOff className="h-4 w-4 text-ink-dim" aria-hidden />
            )}
            Empty ghost cells render only at capacity 512 or below.
          </div>

          <CellDetail scene={scene} selectedCell={selectedCell} />
        </aside>
      </div>
    </DialogShell>
  );
}

export function ConnectedNeuronCluster3DPopup() {
  const {
    graph,
    cluster3dNodeId,
    closeCluster3d,
    revealGraphNode,
    revealGraphNodeInFull,
  } = useGraphView();
  const { activeTrainingJob } = useActiveTrainingJob();
  const scene = useMemo(
    () =>
      buildCluster3DSceneModel({
        graph,
        selectedNodeId: cluster3dNodeId,
        activeTrainingJob,
      }),
    [activeTrainingJob, cluster3dNodeId, graph],
  );

  useEffect(() => {
    if (cluster3dNodeId && !scene) {
      closeCluster3d();
    }
  }, [closeCluster3d, cluster3dNodeId, scene]);

  if (!cluster3dNodeId || !scene) {
    return null;
  }

  return (
    <NeuronCluster3DPopup
      scene={scene}
      onClose={closeCluster3d}
      onRevealCluster={revealGraphNode}
      onRevealNodeInFull={revealGraphNodeInFull}
    />
  );
}
