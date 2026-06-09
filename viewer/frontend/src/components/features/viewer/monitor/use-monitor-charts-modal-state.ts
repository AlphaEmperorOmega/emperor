import { useEffect, useMemo, useState } from "react";
import { type GraphNode } from "@/lib/api";
import {
  type LinearMonitorComparisonCandidateGroups,
  type LinearMonitorComparisonScope,
} from "@/lib/graph/monitor-targets";
import { type MonitorChartsSource } from "@/types/monitor";
import { useMonitorChartQueries } from "@/components/features/viewer/monitor/use-monitor-chart-queries";

export const emptyMonitorComparisonCandidateGroups: LinearMonitorComparisonCandidateGroups = {
  "same-stack": [],
  "all-layers": [],
};

type UseMonitorChartsModalStateInput = {
  node: GraphNode;
  source: MonitorChartsSource;
  comparisonCandidateGroups: LinearMonitorComparisonCandidateGroups;
};

export function useMonitorChartsModalState({
  node,
  source,
  comparisonCandidateGroups,
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
  const [dataset, setDataset] = useState(defaultDataset);
  const [preset, setPreset] = useState(defaultPreset);
  const [comparisonScope, setComparisonScope] =
    useState<LinearMonitorComparisonScope>(defaultComparisonScope);
  const [comparisonPath, setComparisonPath] = useState("");
  const sourceDatasets = useMemo(
    () => activeJob?.datasets ?? (historicalDataset ? [historicalDataset] : []),
    [activeJob?.datasets, historicalDataset],
  );
  const sourcePresets = useMemo(
    () =>
      activeJob?.presets ??
      (activeJob?.preset ? [activeJob.preset] : historicalPreset ? [historicalPreset] : []),
    [activeJob?.preset, activeJob?.presets, historicalPreset],
  );
  const monitorCount = activeJob?.monitors.length;
  const comparisonCandidates = comparisonCandidateGroups[comparisonScope];
  const hasComparisonCandidates =
    comparisonCandidateGroups["same-stack"].length > 0 ||
    comparisonCandidateGroups["all-layers"].length > 0;
  const comparisonNode = useMemo(
    () => comparisonCandidates.find((candidate) => candidate.path === comparisonPath),
    [comparisonCandidates, comparisonPath],
  );
  const comparisonCandidatePaths = useMemo(
    () => new Set(comparisonCandidates.map((candidate) => candidate.path)),
    [comparisonCandidates],
  );
  const query = useMonitorChartQueries({
    source,
    nodePath: node.path,
    dataset,
    preset,
    comparisonNodePath: comparisonNode?.path,
  });

  useEffect(() => {
    if (dataset && sourceDatasets.includes(dataset)) {
      return;
    }
    setDataset(defaultDataset);
  }, [dataset, defaultDataset, sourceDatasets]);

  useEffect(() => {
    if (!activeJob) {
      return;
    }
    if (preset && sourcePresets.includes(preset)) {
      return;
    }
    setPreset(defaultPreset);
  }, [activeJob, defaultPreset, preset, sourcePresets]);

  useEffect(() => {
    setComparisonScope((currentScope) =>
      comparisonCandidateGroups[currentScope].length > 0
        ? currentScope
        : defaultComparisonScope,
    );
  }, [comparisonCandidateGroups, defaultComparisonScope]);

  useEffect(() => {
    if (!comparisonPath) {
      return;
    }
    if (comparisonCandidatePaths.has(comparisonPath)) {
      return;
    }
    setComparisonPath("");
  }, [comparisonCandidatePaths, comparisonPath]);

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
