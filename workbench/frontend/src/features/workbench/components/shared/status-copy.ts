export const workbenchStatusCopy = {
  loading: {
    workbench: "Loading workbench",
    modelCatalog: "Loading model catalog",
    targetData: "Loading target data",
    configSchema: "Loading config schema",
    monitorOptions: "Loading monitor options",
    searchAxes: "Loading search axes",
    runs: "Loading runs",
    comparisonData: "Loading comparison data",
    logFolders: "Loading folders",
  },
  empty: {
    configFields: "No config fields",
    optionalMonitors: "No optional monitors for this model",
    searchAxes: "No search axes for this model",
    graph: "No graph loaded",
    graphDetail: "Preview data has not returned yet.",
    comparisonData: "No comparison data",
    layerMonitorExperiments: "No experiments with layer monitor data",
    runsForPreset: "No runs for this preset",
    safeLogFolders: "No safe experiment folders found",
  },
} as const;

export function configFieldsNoMatchCopy(query: string) {
  return `No config fields match "${query}".`;
}
