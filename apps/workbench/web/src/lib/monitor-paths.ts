import type { LogRunTags } from "@/lib/api/logs";

const PARAMETER_MONITOR_CHANNELS = new Set(["weights", "bias"]);
const PARAMETER_MONITOR_METRICS = new Set([
  "relative_delta_norm",
  "delta_norm",
  "l2_norm",
  "mean",
  "var",
  "grad_mean",
  "grad_var",
  "grad_norm",
  "update_ratio",
  "spectral_norm",
  "condition_number",
  "effective_rank",
  "dead_input_fraction",
  "dead_output_fraction",
]);

export type ParameterMonitorTag = {
  nodePath: string;
  channel: "weights" | "bias";
  metric: string;
};

function allTags(
  tags: Pick<LogRunTags, "scalarTags" | "histogramTags" | "imageTags">,
) {
  return [...tags.scalarTags, ...tags.histogramTags, ...tags.imageTags];
}

export function parseParameterMonitorTag(
  tag: string,
): ParameterMonitorTag | undefined {
  const parts = tag.split("/");
  if (parts.length < 3) {
    return undefined;
  }
  const metric = parts.at(-1) ?? "";
  const channel = parts.at(-2) ?? "";
  const nodePath = parts.slice(0, -2).join("/");
  if (!nodePath || !PARAMETER_MONITOR_CHANNELS.has(channel)) {
    return undefined;
  }
  if (!PARAMETER_MONITOR_METRICS.has(metric)) {
    return undefined;
  }
  return {
    nodePath,
    channel: channel as "weights" | "bias",
    metric,
  };
}

export function isParameterMonitorTag(tag: string) {
  return parseParameterMonitorTag(tag) !== undefined;
}

export function monitorPathsMatch(
  left: string | undefined,
  right: string | undefined,
) {
  return Boolean(left && right && left === right);
}

export function tagMatchesNodePath(tag: string, nodePath: string | undefined) {
  if (!nodePath) {
    return false;
  }
  const tagNodePath = tag.split("/").at(0);
  return monitorPathsMatch(tagNodePath, nodePath);
}

export function parameterTagMatchesNodePath(
  tag: string,
  nodePath: string | undefined,
) {
  const parsed = parseParameterMonitorTag(tag);
  return parsed ? monitorPathsMatch(parsed.nodePath, nodePath) : false;
}

export function tagsIncludeParameterMonitorData(
  tags: Pick<LogRunTags, "scalarTags" | "histogramTags" | "imageTags"> | undefined,
) {
  if (!tags) {
    return false;
  }
  return allTags(tags).some(isParameterMonitorTag);
}

export function tagsMatchParameterNodePath(
  tags: Pick<LogRunTags, "scalarTags" | "histogramTags" | "imageTags"> | undefined,
  nodePath: string | undefined,
) {
  if (!tags || !nodePath) {
    return false;
  }
  return allTags(tags).some((tag) => parameterTagMatchesNodePath(tag, nodePath));
}
