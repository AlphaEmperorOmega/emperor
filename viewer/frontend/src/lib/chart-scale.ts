import { formatNumber } from "@/lib/format";

export type ChartPoint = {
  step: number;
  value: number;
};

export type ChartDomain = {
  minStep: number;
  maxStep: number;
  minValue: number;
  maxValue: number;
};

export type ChartPadding = number | { x: number; y: number };

export type ChartDimensions = {
  width: number;
  height: number;
  padding: ChartPadding;
  domain?: ChartDomain;
  pathKind?: "polyline" | "path";
  pathPrecision?: number;
  stepDomainMode?: "extent" | "series";
};

export type ChartCoordinate = {
  x: number;
  y: number;
};

export type ScaleResult = {
  domain: ChartDomain;
  width: number;
  height: number;
  paddingX: number;
  paddingY: number;
  pathKind: "polyline" | "path";
  pathPrecision: number;
  x: (point: ChartPoint) => number;
  y: (point: ChartPoint) => number;
  coordinate: (point: ChartPoint) => ChartCoordinate;
};

function resolvePadding(padding: ChartPadding) {
  if (typeof padding === "number") {
    return { paddingX: padding, paddingY: padding };
  }
  return { paddingX: padding.x, paddingY: padding.y };
}

function domainForPoints(
  points: readonly ChartPoint[],
  stepDomainMode: ChartDimensions["stepDomainMode"],
): ChartDomain {
  const values = points.map((point) => point.value);
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;

  if (stepDomainMode === "series") {
    const minStep = points[0]?.step ?? 0;
    const maxStep = points.at(-1)?.step ?? minStep;
    return { minStep, maxStep, minValue, maxValue };
  }

  const steps = points.map((point) => point.step);
  return {
    minStep: steps.length ? Math.min(...steps) : 0,
    maxStep: steps.length ? Math.max(...steps) : 1,
    minValue,
    maxValue,
  };
}

export function buildLinearScale(
  points: readonly ChartPoint[],
  dimensions: ChartDimensions,
): ScaleResult {
  const { width, height } = dimensions;
  const { paddingX, paddingY } = resolvePadding(dimensions.padding);
  const domain =
    dimensions.domain ?? domainForPoints(points, dimensions.stepDomainMode ?? "extent");
  const stepSpan = domain.maxStep - domain.minStep || 1;
  const valueSpan = domain.maxValue - domain.minValue || 1;

  const x = (point: ChartPoint) =>
    paddingX + ((point.step - domain.minStep) / stepSpan) * (width - paddingX * 2);
  const y = (point: ChartPoint) =>
    height -
    paddingY -
    ((point.value - domain.minValue) / valueSpan) * (height - paddingY * 2);

  return {
    domain,
    width,
    height,
    paddingX,
    paddingY,
    pathKind: dimensions.pathKind ?? "polyline",
    pathPrecision: dimensions.pathPrecision ?? 2,
    x,
    y,
    coordinate: (point) => ({ x: x(point), y: y(point) }),
  };
}

export function buildChartPath(points: readonly ChartPoint[], scale: ScaleResult) {
  if (scale.pathKind === "path") {
    return points
      .map((point, index) => {
        const command = index === 0 ? "M" : "L";
        return `${command} ${scale.x(point).toFixed(scale.pathPrecision)} ${scale
          .y(point)
          .toFixed(scale.pathPrecision)}`;
      })
      .join(" ");
  }

  return points
    .map((point) => {
      const coordinate = scale.coordinate(point);
      return `${coordinate.x},${coordinate.y}`;
    })
    .join(" ");
}

export function formatChartDomain(domain: ChartDomain) {
  return {
    minLabel: formatNumber(domain.minValue),
    maxLabel: formatNumber(domain.maxValue),
  };
}
