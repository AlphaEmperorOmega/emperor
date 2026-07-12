import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import {
  type Cluster3DCell,
  type Cluster3DSceneModel,
  type GraphCoordinate,
} from "@/lib/graph";
import {
  workbenchVisualTokens,
  workbenchVisualizationTokens,
} from "@/lib/visual-tokens";

const CELL_SPACING = 1.12;

type SceneProps = {
  scene: Cluster3DSceneModel;
  visibleX: Set<number>;
  visibleY: Set<number>;
  visibleZ: Set<number>;
  selectedKey: string | null;
  onSelectCell: (cell: Cluster3DCell) => void;
};

type RenderCell = {
  key: string;
  coordinate: GraphCoordinate;
};

const categoryColors: Record<Cluster3DCell["category"], string> =
  workbenchVisualizationTokens.clusterCategories;

function coordinatePosition(
  coordinate: GraphCoordinate,
  capacity: GraphCoordinate,
) {
  const [x, y, z] = coordinate;
  return [
    (x - (capacity[0] + 1) / 2) * CELL_SPACING,
    (z - (capacity[2] + 1) / 2) * CELL_SPACING,
    (y - (capacity[1] + 1) / 2) * CELL_SPACING,
  ] as const;
}

function isCoordinateVisible(
  coordinate: GraphCoordinate,
  visibleX: Set<number>,
  visibleY: Set<number>,
  visibleZ: Set<number>,
) {
  const [x, y, z] = coordinate;
  return visibleX.has(x) && visibleY.has(y) && visibleZ.has(z);
}

function SceneControls() {
  const invalidate = useThree((state) => state.invalidate);
  return (
    <OrbitControls
      makeDefault
      enableDamping={false}
      minDistance={3}
      maxDistance={80}
      onChange={() => invalidate()}
    />
  );
}

function InstancedCells({
  cells,
  capacity,
  color,
  opacity = 1,
  radius = 0.34,
  depthWrite = true,
  renderOrder = 0,
  wireframe = false,
  onSelectCell,
}: {
  cells: RenderCell[];
  capacity: GraphCoordinate;
  color: string;
  opacity?: number;
  radius?: number;
  depthWrite?: boolean;
  renderOrder?: number;
  wireframe?: boolean;
  onSelectCell?: (cell: RenderCell) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) {
      return;
    }
    cells.forEach((cell, index) => {
      const [x, y, z] = coordinatePosition(cell.coordinate, capacity);
      dummy.position.set(x, y, z);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
  }, [capacity, cells, dummy]);

  if (cells.length === 0) {
    return null;
  }

  return (
    <instancedMesh
      ref={meshRef}
      renderOrder={renderOrder}
      args={[undefined, undefined, cells.length]}
      onClick={
        onSelectCell
          ? (event: ThreeEvent<MouseEvent>) => {
              event.stopPropagation();
              const cell =
                typeof event.instanceId === "number"
                  ? cells[event.instanceId]
                  : undefined;
              if (cell) {
                onSelectCell(cell);
              }
            }
          : undefined
      }
    >
      <sphereGeometry args={[radius, 16, 12]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        depthWrite={depthWrite}
        wireframe={wireframe}
      />
    </instancedMesh>
  );
}

function CapacityBounds({ capacity }: { capacity: GraphCoordinate }) {
  return (
    <mesh>
      <boxGeometry
        args={[
          capacity[0] * CELL_SPACING,
          capacity[2] * CELL_SPACING,
          capacity[1] * CELL_SPACING,
        ]}
      />
      <meshBasicMaterial
        color={workbenchVisualTokens.clusterBounds}
        wireframe
        transparent
        opacity={0.35}
      />
    </mesh>
  );
}

function SelectedOutline({
  selectedCell,
  capacity,
}: {
  selectedCell: Cluster3DCell | null;
  capacity: GraphCoordinate;
}) {
  if (!selectedCell) {
    return null;
  }
  const [x, y, z] = coordinatePosition(selectedCell.coordinate, capacity);
  return (
    <mesh position={[x, y, z]}>
      <sphereGeometry args={[0.46, 20, 14]} />
      <meshBasicMaterial
        color={workbenchVisualTokens.clusterSelected}
        wireframe
        transparent
        opacity={0.95}
      />
    </mesh>
  );
}

function ReachHighlights({
  selectedCell,
  capacity,
  visibleX,
  visibleY,
  visibleZ,
}: {
  selectedCell: Cluster3DCell | null;
  capacity: GraphCoordinate;
  visibleX: Set<number>;
  visibleY: Set<number>;
  visibleZ: Set<number>;
}) {
  const reachCells = useMemo(() => {
    if (!selectedCell?.reach) {
      return [];
    }
    return selectedCell.reach.inBoundsConnections
      .filter((coordinate) =>
        isCoordinateVisible(coordinate, visibleX, visibleY, visibleZ),
      )
      .map((coordinate) => ({
        key: coordinate.join(","),
        coordinate,
      }));
  }, [selectedCell, visibleX, visibleY, visibleZ]);

  return (
    <>
      <InstancedCells
        cells={reachCells}
        capacity={capacity}
        color={workbenchVisualTokens.clusterReach}
        opacity={0.035}
        radius={0.58}
        depthWrite={false}
        renderOrder={1}
      />
      <InstancedCells
        cells={reachCells}
        capacity={capacity}
        color={workbenchVisualTokens.clusterReachWireframe}
        opacity={0.09}
        radius={0.6}
        depthWrite={false}
        renderOrder={2}
        wireframe
      />
    </>
  );
}

function ClusterCells({
  scene,
  visibleX,
  visibleY,
  visibleZ,
  selectedKey,
  onSelectCell,
}: SceneProps) {
  const visibleCells = useMemo(
    () =>
      scene.activeCells.filter((cell) =>
        isCoordinateVisible(cell.coordinate, visibleX, visibleY, visibleZ),
      ),
    [scene.activeCells, visibleX, visibleY, visibleZ],
  );
  const selectedCell =
    scene.activeCells.find((cell) => cell.key === selectedKey) ?? null;
  const cellsByCategory = useMemo(
    () => ({
      initial: visibleCells.filter((cell) => cell.category === "initial"),
      grown: visibleCells.filter((cell) => cell.category === "grown"),
      recentAdded: visibleCells.filter((cell) => cell.category === "recentAdded"),
    }),
    [visibleCells],
  );
  const ghostCells = useMemo(() => {
    if (!scene.renderGhostCells) {
      return [];
    }
    const activeKeys = new Set(scene.activeCells.map((cell) => cell.key));
    const cells: RenderCell[] = [];
    for (let z = 1; z <= scene.capacity[2]; z += 1) {
      if (!visibleZ.has(z)) {
        continue;
      }
      for (let y = 1; y <= scene.capacity[1]; y += 1) {
        if (!visibleY.has(y)) {
          continue;
        }
        for (let x = 1; x <= scene.capacity[0]; x += 1) {
          if (!visibleX.has(x)) {
            continue;
          }
          const key = `${x},${y},${z}`;
          if (!activeKeys.has(key)) {
            cells.push({ key, coordinate: [x, y, z] });
          }
        }
      }
    }
    return cells;
  }, [scene.activeCells, scene.capacity, scene.renderGhostCells, visibleX, visibleY, visibleZ]);

  return (
    <>
      <InstancedCells
        cells={ghostCells}
        capacity={scene.capacity}
        color={workbenchVisualTokens.clusterGhost}
        opacity={0.14}
        radius={0.22}
      />
      <ReachHighlights
        selectedCell={selectedCell}
        capacity={scene.capacity}
        visibleX={visibleX}
        visibleY={visibleY}
        visibleZ={visibleZ}
      />
      {(["initial", "grown", "recentAdded"] as const).map((category) => (
        <InstancedCells
          key={category}
          cells={cellsByCategory[category]}
          capacity={scene.capacity}
          color={categoryColors[category]}
          renderOrder={3}
          onSelectCell={(cell) => {
            const activeCell = scene.activeCells.find(
              (candidate) => candidate.key === cell.key,
            );
            if (activeCell) {
              onSelectCell(activeCell);
            }
          }}
        />
      ))}
      <SelectedOutline selectedCell={selectedCell} capacity={scene.capacity} />
    </>
  );
}

export function NeuronCluster3DScene(props: SceneProps) {
  const maxAxis = Math.max(...props.scene.capacity);
  const cameraDistance = Math.max(6, maxAxis * 2.1);

  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 1.5]}
      camera={{
        position: [cameraDistance, cameraDistance * 0.72, cameraDistance],
        fov: 42,
        near: 0.1,
        far: 200,
      }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      onPointerDown={(event) => event.stopPropagation()}
      onWheel={(event) => event.stopPropagation()}
      className="h-full min-h-[420px] w-full"
    >
      <color attach="background" args={[workbenchVisualTokens.scene]} />
      <ambientLight intensity={0.85} />
      <CapacityBounds capacity={props.scene.capacity} />
      <ClusterCells {...props} />
      <SceneControls />
    </Canvas>
  );
}
