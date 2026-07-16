import { useMemo, useState } from "react";
import type { GraphNode } from "@/lib/api/inspection";
import {
  type LinearMonitorComparisonCandidateGroups,
  type LinearMonitorComparisonScope,
} from "@/lib/graph/monitor-targets";
import { type MonitorChartsSource } from "@/types/monitor";
import { useMonitorChartQueries } from "@/features/workbench/state/graph-monitor/use-monitor-chart-queries";

export const emptyMonitorComparisonCandidateGroups: LinearMonitorComparisonCandidateGroups = {
  "same-stack": [],
  "all-layers": [],
};

type UseMonitorChartsModalStateInput = {
  node: GraphNode;
  source: MonitorChartsSource;
  comparisonCandidateGroups: LinearMonitorComparisonCandidateGroups;
  protectedReadsEnabled?: boolean;
};

export function useMonitorChartsModalState({
  node,
  source,
  comparisonCandidateGroups,
  protectedReadsEnabled = true,
}: UseMonitorChartsModalStateInput) {
  const activeJob = source.kind === "active-job" ? source.job : undefined;
  const historicalRun = source.kind === "historical-run" ? source.run : undefined;
  const historicalRunGroup =
    source.kind === "historical-run-group" ? source : undefined;
  const historicalRuns = useMemo(
    () => historicalRunGroup?.runs ?? (historicalRun ? [historicalRun] : []),
    [historicalRun, historicalRunGroup?.runs],
  );
  const historicalExperiment = historicalRunGroup?.experiment ?? historicalRun?.experiment ?? "";
  const historicalDataset = historicalRunGroup?.dataset ?? historicalRun?.dataset ?? "";
  const historicalPreset = historicalRunGroup?.preset ?? historicalRun?.preset ?? "";
  const defaultDataset = activeJob?.currentDataset ?? activeJob?.datasets[0] ?? historicalDataset;
  const defaultPreset =
    activeJob?.currentPreset ?? activeJob?.preset ?? historicalPreset;
  const defaultComparisonScope: LinearMonitorComparisonScope =
    comparisonCandidateGroups["same-stack"].length > 0 ? "same-stack" : "all-layers";
  const [selectedDataset, setDataset] = useState(defaultDataset);
  const [selectedPreset, setPreset] = useState(defaultPreset);
  const [selectedComparisonScope, setComparisonScope] =
    useState<LinearMonitorComparisonScope>(defaultComparisonScope);
  const [selectedComparisonPath, setComparisonPath] = useState("");
  const sourceDatasets = useMemo(
    () => activeJob?.datasets ?? (historicalDataset ? [historicalDataset] : []),
    [activeJob?.datasets, historicalDataset],
  );
  const sourcePresets = useMemo(
    () =>
      activeJob?.presets ??
      (activeJob?.preset ? [activeJob.preset] : historicalPreset ? [historicalPreset] : []),
    [activeJob, historicalPreset],
  );
  const dataset =
    selectedDataset && sourceDatasets.includes(selectedDataset)
      ? selectedDataset
      : defaultDataset;
  const preset =
    selectedPreset && sourcePresets.includes(selectedPreset)
      ? selectedPreset
      : defaultPreset;
  const comparisonScope =
    comparisonCandidateGroups[selectedComparisonScope].length > 0
      ? selectedComparisonScope
      : defaultComparisonScope;
  const monitorCount = activeJob?.monitors.length;
  const comparisonCandidates = comparisonCandidateGroups[comparisonScope];
  const hasComparisonCandidates =
    comparisonCandidateGroups["same-stack"].length > 0 ||
    comparisonCandidateGroups["all-layers"].length > 0;
  const comparisonNode = useMemo(
    () =>
      comparisonCandidates.find(
        (candidate) => candidate.path === selectedComparisonPath,
      ),
    [comparisonCandidates, selectedComparisonPath],
  );
  const comparisonPath = comparisonNode?.path ?? "";
  const query = useMonitorChartQueries({
    source,
    nodePath: node.path,
    dataset,
    preset,
    comparisonNodePath: comparisonNode?.path,
    enabled: protectedReadsEnabled,
  });

  return {
    activeJob,
    historicalRun,
    historicalRunGroup,
    historicalRuns,
    historicalExperiment,
    historicalDataset,
    historicalPreset,
    dataset,
    setDataset,
    preset,
    setPreset,
    sourceDatasets,
    sourcePresets,
    monitorCount,
    comparisonCandidates,
    hasComparisonCandidates,
    comparisonScope,
    setComparisonScope,
    comparisonPath,
    setComparisonPath,
    comparisonNode,
    query,
  };
}
