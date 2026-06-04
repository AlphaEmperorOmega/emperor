import { type GraphNode } from "@/lib/api";
import { isRecord } from "@/lib/graph/helpers";
import { type NodeDetailEntry } from "@/lib/graph/format/text";

const PREVIEW_ONLY_DETAIL_KEYS = new Set([
  "weightShape",
  "biasShape",
  "dims",
  "inputDim",
  "inputShape",
  "hiddenDim",
  "outputDim",
  "outputShape",
  "shapeTransition",
  "cluster",
  "terminalReach",
]);

export function nodeBadges(details: GraphNode["details"]) {
  const badges: Array<[string, unknown]> = [];
  if (typeof details.dims === "string") {
    badges.push(["dims", details.dims]);
  }
  if (typeof details.activation === "string" && details.activation !== "DISABLED") {
    badges.push(["act", details.activation]);
  }
  if (typeof details.layerNorm === "string" && details.layerNorm !== "DISABLED") {
    badges.push(["norm", details.layerNorm]);
  }
  if (typeof details.dropout === "number" && details.dropout > 0) {
    badges.push(["drop", details.dropout]);
  }
  if (details.gate === true) {
    badges.push(["gate", "on"]);
  }
  if (details.halting === true) {
    badges.push(["halt", "on"]);
  }
  if (isRecord(details.recurrent)) {
    badges.push(["steps", details.recurrent.maxSteps]);
  }
  return badges;
}

function shapeText(value: unknown) {
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "scalar";
    }
    if (value.every((item) => typeof item === "number" || typeof item === "string")) {
      return value.join(" x ");
    }
  }
  return null;
}

export function parameterShapeEntries(details: GraphNode["details"]) {
  return [
    ["weightShape", "W"],
    ["biasShape", "b"],
  ].flatMap(([key, label]) => {
    const shape = shapeText(details[key]);
    return shape ? [{ key, label, shape }] : [];
  });
}

export function nodeDetailEntries(
  details: GraphNode["details"],
  config?: GraphNode["config"],
): NodeDetailEntry[] {
  if (config) {
    return config.fields.map((field) => ({
      key: field.key,
      value: field.value,
      source: "config" as const,
    }));
  }

  return Object.entries(details)
    .filter(([key]) => !PREVIEW_ONLY_DETAIL_KEYS.has(key))
    .map(([key, value]) => ({ key, value, source: "details" as const }));
}
