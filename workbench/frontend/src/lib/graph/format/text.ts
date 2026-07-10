import { type GraphNode } from "@/lib/api";
import { SEMANTIC_LABEL_TYPE_NAMES } from "@/lib/graph/constants";
import { lastPathSegment } from "@/lib/graph/helpers";
import { type GraphCoordinate } from "@/lib/graph/types";

const exactCountFormatter = new Intl.NumberFormat("en-US");
const bytesPerMegabyte = 1024 * 1024;

export type NodeDetailEntry = {
  key: string;
  value: unknown;
  source: "config" | "details";
};

export function detailText(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export function configDetailText(value: unknown) {
  if (value === null || value === undefined) {
    return "None";
  }
  return detailText(value);
}

export function nodeDetailEntryText(entry: NodeDetailEntry) {
  return entry.source === "config" ? configDetailText(entry.value) : detailText(entry.value);
}

export function formatExactCount(count: number) {
  return exactCountFormatter.format(count);
}

export function formatGraphCoordinate(coordinate: GraphCoordinate) {
  return `(${coordinate[0]}, ${coordinate[1]}, ${coordinate[2]})`;
}

export function formatCompactCount(count: number) {
  const absoluteCount = Math.abs(count);
  if (absoluteCount < 1000) {
    return formatExactCount(count);
  }

  const units = [
    { suffix: "B", value: 1_000_000_000 },
    { suffix: "M", value: 1_000_000 },
    { suffix: "K", value: 1_000 },
  ];
  const unit = units.find((candidate) => absoluteCount >= candidate.value);
  if (!unit) {
    return formatExactCount(count);
  }

  const value = count / unit.value;
  const formatted = value >= 100 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted.replace(/\.0$/, "")}${unit.suffix}`;
}

export function formatModelSize(bytes: number | null | undefined) {
  if (
    typeof bytes !== "number" ||
    !Number.isFinite(bytes) ||
    bytes <= 0
  ) {
    return undefined;
  }

  const megabytes = bytes / bytesPerMegabyte;
  if (megabytes < 0.01) {
    return "<0.01 MB";
  }
  if (megabytes < 10) {
    return `${megabytes.toFixed(2).replace(/\.?0+$/, "")} MB`;
  }
  if (megabytes < 100) {
    return `${megabytes.toFixed(1).replace(/\.0$/, "")} MB`;
  }
  return `${formatExactCount(Math.round(megabytes))} MB`;
}

export function simpleGraphParamText(parameterCount: number | null | undefined) {
  if (
    typeof parameterCount !== "number" ||
    !Number.isFinite(parameterCount) ||
    parameterCount <= 0
  ) {
    return undefined;
  }

  return `${formatCompactCount(parameterCount)} params`;
}

export type NodeDimRange = {
  inputDim: string;
  outputDim: string;
  text: string;
};

function dimensionScalarText(value: unknown) {
  if (typeof value === "string") {
    const text = value.trim();
    if (text.length === 0) {
      return undefined;
    }
    const numericValue = Number(text);
    return Number.isFinite(numericValue) && numericValue > 0 ? text : undefined;
  }
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return String(value);
  }
  return undefined;
}

function dimensionRange(inputDim: string | undefined, outputDim: string | undefined) {
  if (!inputDim || !outputDim) {
    return undefined;
  }
  return {
    inputDim,
    outputDim,
    text: `${inputDim} -> ${outputDim}`,
  };
}

function dimensionRangeFromText(value: unknown) {
  if (typeof value !== "string") {
    return undefined;
  }

  const parts = value.trim().split(/\s*->\s*/);
  if (parts.length !== 2) {
    return undefined;
  }

  return dimensionRange(
    dimensionScalarText(parts[0]),
    dimensionScalarText(parts[1]),
  );
}

function configFieldValue(config: GraphNode["config"] | null | undefined, key: string) {
  return config?.fields.find((field) => field.key === key)?.value;
}

export function nodeDimRange(
  details: GraphNode["details"] | null | undefined,
  config?: GraphNode["config"] | null,
): NodeDimRange | undefined {
  const explicitDims = dimensionRangeFromText(details?.dims);
  if (explicitDims) {
    return explicitDims;
  }

  const detailDims = dimensionRange(
    dimensionScalarText(details?.inputDim),
    dimensionScalarText(details?.outputDim),
  );
  if (detailDims) {
    return detailDims;
  }

  return dimensionRange(
    dimensionScalarText(configFieldValue(config, "input_dim")),
    dimensionScalarText(configFieldValue(config, "output_dim")),
  );
}

export function nodeDimsText(
  details: GraphNode["details"] | null | undefined,
  config?: GraphNode["config"] | null,
) {
  return nodeDimRange(details, config)?.text;
}

function humanizePathSegment(segment: string) {
  return segment
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function semanticPathLabel(node: GraphNode) {
  const pathSegment = lastPathSegment(node.path);
  const hasSemanticPathSegment = pathSegment.length > 0 && !/^\d+$/.test(pathSegment);
  if (SEMANTIC_LABEL_TYPE_NAMES.has(node.typeName) && hasSemanticPathSegment) {
    return humanizePathSegment(pathSegment);
  }
  return undefined;
}

export function nodeTitle(node: GraphNode) {
  return node.typeName;
}

export function nodeSubtitle(node: GraphNode) {
  const semanticLabel = semanticPathLabel(node);
  return semanticLabel ? `${semanticLabel} · ${node.path}` : node.path;
}

export function structureNodeLabel(node: GraphNode) {
  const segment = lastPathSegment(node.path) || node.id;
  return `${segment}: ${node.typeName}`;
}
