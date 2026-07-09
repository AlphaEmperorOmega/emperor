import {
  memo,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { formatNumber } from "@/features/workbench/state/logs/logs-selectors";
import { type ConfusionMatrixHeatmap } from "@/features/workbench/state/logs/log-diagnostics";
import { cn } from "@/lib/utils";

const LABEL_SLOT_SIZE = 30;
const CELL_SLOT_SIZE = 30;
const CELL_DRAW_SIZE = 28;
const CELL_GAP = 4;
const CELL_RADIUS = 4;
const INTERSECTION_ROOT_MARGIN = "640px 0px";

const MONO_FONT =
  "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace";

type MatrixDimensions = {
  width: number;
  height: number;
};

type MatrixRenderData = {
  classCount: number;
  classes: number[];
  dimensions: MatrixDimensions;
  values: Float64Array;
};

type HoveredCell = {
  clientX: number;
  clientY: number;
  predictedClass: number;
  trueClass: number;
  value: number;
};

function heatmapColor(value: number, trueClass: number, predictedClass: number) {
  const alpha = Math.max(0.08, Math.min(0.92, value));
  if (trueClass === predictedClass) {
    return `rgba(64, 201, 127, ${alpha})`;
  }
  return `rgba(239, 86, 86, ${alpha})`;
}

function matrixIndex(trueClass: number, predictedClass: number, classCount: number) {
  return trueClass * classCount + predictedClass;
}

function matrixDimensions(classCount: number): MatrixDimensions {
  const size = LABEL_SLOT_SIZE + classCount * (CELL_SLOT_SIZE + CELL_GAP);
  return { width: size, height: size };
}

function buildMatrixRenderData(heatmap: ConfusionMatrixHeatmap): MatrixRenderData {
  const classCount = Math.max(0, heatmap.classCount);
  const values = new Float64Array(classCount * classCount);

  for (const cell of heatmap.cells) {
    if (
      cell.trueClass < 0 ||
      cell.predictedClass < 0 ||
      cell.trueClass >= classCount ||
      cell.predictedClass >= classCount
    ) {
      continue;
    }
    values[matrixIndex(cell.trueClass, cell.predictedClass, classCount)] = cell.value;
  }

  return {
    classCount,
    classes: Array.from({ length: classCount }, (_, index) => index),
    dimensions: matrixDimensions(classCount),
    values,
  };
}

function roundedRectPath(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const right = x + width;
  const bottom = y + height;
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(right - radius, y);
  context.quadraticCurveTo(right, y, right, y + radius);
  context.lineTo(right, bottom - radius);
  context.quadraticCurveTo(right, bottom, right - radius, bottom);
  context.lineTo(x + radius, bottom);
  context.quadraticCurveTo(x, bottom, x, bottom - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function drawConfusionMatrix(
  canvas: HTMLCanvasElement,
  renderData: MatrixRenderData,
) {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const pixelWidth = Math.ceil(renderData.dimensions.width * pixelRatio);
  const pixelHeight = Math.ceil(renderData.dimensions.height * pixelRatio);
  if (canvas.width !== pixelWidth) {
    canvas.width = pixelWidth;
  }
  if (canvas.height !== pixelHeight) {
    canvas.height = pixelHeight;
  }

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, renderData.dimensions.width, renderData.dimensions.height);
  context.textAlign = "center";
  context.textBaseline = "middle";

  context.font = `600 10px ${MONO_FONT}`;
  context.fillStyle = "rgba(210, 219, 235, 0.58)";
  for (const classIndex of renderData.classes) {
    const offset =
      LABEL_SLOT_SIZE +
      CELL_GAP +
      classIndex * (CELL_SLOT_SIZE + CELL_GAP) +
      CELL_SLOT_SIZE / 2;
    context.fillText(String(classIndex), offset, LABEL_SLOT_SIZE / 2);
    context.fillText(String(classIndex), LABEL_SLOT_SIZE / 2, offset);
  }

  context.font = `600 9px ${MONO_FONT}`;
  for (const trueClass of renderData.classes) {
    for (const predictedClass of renderData.classes) {
      const value =
        renderData.values[
          matrixIndex(trueClass, predictedClass, renderData.classCount)
        ] ?? 0;
      const x =
        LABEL_SLOT_SIZE +
        CELL_GAP +
        predictedClass * (CELL_SLOT_SIZE + CELL_GAP) +
        1;
      const y =
        LABEL_SLOT_SIZE +
        CELL_GAP +
        trueClass * (CELL_SLOT_SIZE + CELL_GAP) +
        1;

      context.fillStyle = heatmapColor(value, trueClass, predictedClass);
      roundedRectPath(context, x, y, CELL_DRAW_SIZE, CELL_DRAW_SIZE, CELL_RADIUS);
      context.fill();

      if (value >= 0.01) {
        context.fillStyle = "rgba(255, 255, 255, 0.94)";
        context.fillText(
          String(Math.round(value * 100)),
          x + CELL_DRAW_SIZE / 2,
          y + CELL_DRAW_SIZE / 2,
        );
      }
    }
  }
}

function getCellFromPointer({
  canvas,
  clientX,
  clientY,
  renderData,
}: {
  canvas: HTMLCanvasElement;
  clientX: number;
  clientY: number;
  renderData: MatrixRenderData;
}): HoveredCell | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }

  const localX = (clientX - rect.left) * (renderData.dimensions.width / rect.width);
  const localY = (clientY - rect.top) * (renderData.dimensions.height / rect.height);
  const matrixX = localX - LABEL_SLOT_SIZE - CELL_GAP;
  const matrixY = localY - LABEL_SLOT_SIZE - CELL_GAP;
  if (matrixX < 0 || matrixY < 0) {
    return null;
  }

  const stride = CELL_SLOT_SIZE + CELL_GAP;
  const predictedClass = Math.floor(matrixX / stride);
  const trueClass = Math.floor(matrixY / stride);
  const xWithinSlot = matrixX - predictedClass * stride;
  const yWithinSlot = matrixY - trueClass * stride;

  if (
    trueClass < 0 ||
    predictedClass < 0 ||
    trueClass >= renderData.classCount ||
    predictedClass >= renderData.classCount ||
    xWithinSlot >= CELL_SLOT_SIZE ||
    yWithinSlot >= CELL_SLOT_SIZE
  ) {
    return null;
  }

  return {
    clientX,
    clientY,
    predictedClass,
    trueClass,
    value:
      renderData.values[
        matrixIndex(trueClass, predictedClass, renderData.classCount)
      ] ?? 0,
  };
}

function scheduleAnimationFrame(callback: FrameRequestCallback) {
  if (typeof window.requestAnimationFrame === "function") {
    return window.requestAnimationFrame(callback);
  }
  return window.setTimeout(() => callback(performance.now()), 16);
}

function cancelScheduledAnimationFrame(handle: number) {
  if (typeof window.cancelAnimationFrame === "function") {
    window.cancelAnimationFrame(handle);
    return;
  }
  window.clearTimeout(handle);
}

const LogConfusionMatrixCard = memo(function LogConfusionMatrixCard({
  heatmap,
}: {
  heatmap: ConfusionMatrixHeatmap;
}) {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointerRef = useRef<{ clientX: number; clientY: number } | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const tooltipId = useId();
  const canObserveVisibility = typeof IntersectionObserver !== "undefined";
  const [hasEnteredView, setHasEnteredView] = useState(!canObserveVisibility);
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null);
  const renderData = useMemo(() => buildMatrixRenderData(heatmap), [heatmap]);
  const ariaLabel = `${heatmap.split} confusion matrix for ${heatmap.runLabel}, ${renderData.classCount} classes`;

  useEffect(() => {
    if (hasEnteredView) {
      return;
    }
    const node = cardRef.current;
    if (!node || !canObserveVisibility) {
      setHasEnteredView(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setHasEnteredView(true);
          observer.disconnect();
        }
      },
      { rootMargin: INTERSECTION_ROOT_MARGIN },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [canObserveVisibility, hasEnteredView]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !hasEnteredView) {
      return;
    }
    drawConfusionMatrix(canvas, renderData);
  }, [hasEnteredView, renderData]);

  useEffect(
    () => () => {
      if (animationFrameRef.current !== null) {
        cancelScheduledAnimationFrame(animationFrameRef.current);
      }
    },
    [],
  );

  const updateHoveredCell = useCallback(() => {
    animationFrameRef.current = null;
    const pointer = pointerRef.current;
    const canvas = canvasRef.current;
    if (!pointer || !canvas) {
      setHoveredCell(null);
      return;
    }
    setHoveredCell(
      getCellFromPointer({
        canvas,
        clientX: pointer.clientX,
        clientY: pointer.clientY,
        renderData,
      }),
    );
  }, [renderData]);

  const clearHoveredCell = useCallback(() => {
    pointerRef.current = null;
    if (animationFrameRef.current !== null) {
      cancelScheduledAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setHoveredCell(null);
  }, []);

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) {
        clearHoveredCell();
        return;
      }
      pointerRef.current = {
        clientX: event.clientX,
        clientY: event.clientY,
      };
      if (animationFrameRef.current !== null) {
        return;
      }
      animationFrameRef.current = scheduleAnimationFrame(updateHoveredCell);
    },
    [clearHoveredCell, updateHoveredCell],
  );

  return (
    <div
      ref={cardRef}
      className={cn(
        "grid min-w-0 gap-3 rounded-[8px] border border-line bg-white/[0.018] p-3",
        "[contain:layout_paint_style] [content-visibility:auto] [contain-intrinsic-size:480px_520px]",
      )}
    >
      <div className="min-w-0">
        <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
          {heatmap.split}
        </div>
        <div className="truncate text-sm font-semibold text-ink">
          {heatmap.runLabel}
        </div>
      </div>
      <div className="overflow-x-auto">
        {hasEnteredView ? (
          <canvas
            ref={canvasRef}
            role="img"
            aria-label={ariaLabel}
            aria-describedby={hoveredCell ? tooltipId : undefined}
            className="block max-w-none rounded-[4px]"
            style={{
              height: renderData.dimensions.height,
              width: renderData.dimensions.width,
            }}
            onPointerMove={handlePointerMove}
            onPointerLeave={clearHoveredCell}
          />
        ) : (
          <div
            aria-hidden
            className="rounded-[4px] border border-line-soft bg-white/[0.012]"
            style={{
              height: renderData.dimensions.height,
              width: renderData.dimensions.width,
            }}
          />
        )}
      </div>
      {hoveredCell && (
        <div
          id={tooltipId}
          role="tooltip"
          className="pointer-events-none fixed z-50 rounded-[8px] border border-line bg-panel/95 px-2 py-1 font-mono text-[11px] leading-5 text-ink shadow-panel"
          style={{
            left: hoveredCell.clientX + 12,
            top: hoveredCell.clientY + 12,
          }}
        >
          true {hoveredCell.trueClass}, predicted {hoveredCell.predictedClass},{" "}
          {formatNumber(hoveredCell.value)}
        </div>
      )}
    </div>
  );
});

export function LogConfusionMatrixHeatmaps({
  heatmaps,
}: {
  heatmaps: ConfusionMatrixHeatmap[];
}) {
  if (heatmaps.length === 0) {
    return null;
  }

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {heatmaps.map((heatmap) => (
        <LogConfusionMatrixCard key={heatmap.key} heatmap={heatmap} />
      ))}
    </div>
  );
}
