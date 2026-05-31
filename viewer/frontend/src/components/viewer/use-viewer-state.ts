import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  type ConfigField,
  fetchConfigSchema,
  fetchHealth,
  fetchModels,
  fetchPresets,
  inspectModel,
} from "@/lib/api";
import { type OverrideValues } from "@/components/viewer/config-panel";
import {
  type GraphDetailMode,
  type GraphScope,
  buildChildSummaries,
  buildExpertDiagrams,
  buildGraphNavigation,
  buildStackDiagrams,
  filterGraphByDetail,
  filterGraphByExpansion,
  layoutGraph,
} from "@/lib/graph";

export type ViewMode = "graph" | "hierarchy";

export function useViewerState() {
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedPreset, setSelectedPreset] = useState("");
  const [overrides, setOverrides] = useState<OverrideValues>({});
  const [viewMode, setViewMode] = useState<ViewMode>("graph");
  const [graphDetailMode, setGraphDetailMode] = useState<GraphDetailMode>("basic");
  const [graphScope, setGraphScope] = useState<GraphScope>("opened");
  const [expandedGraphNodeIds, setExpandedGraphNodeIds] = useState<Set<string>>(new Set());
  const [expandedDetailNodeIds, setExpandedDetailNodeIds] = useState<Set<string>>(new Set());
  const [openConfigSections, setOpenConfigSections] = useState<Set<string>>(new Set());
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [previewRequest, setPreviewRequest] = useState<{
    model: string;
    preset: string;
    overrides: OverrideValues;
  } | null>(null);

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: false,
    refetchInterval: 10000,
  });

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
    retry: false,
  });

  const presetsQuery = useQuery({
    queryKey: ["presets", selectedModel],
    queryFn: () => fetchPresets(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
  });

  const schemaQuery = useQuery({
    queryKey: ["config-schema", selectedModel],
    queryFn: () => fetchConfigSchema(selectedModel),
    enabled: selectedModel.length > 0,
    retry: false,
  });

  const inspectQuery = useQuery({
    queryKey: ["inspect", previewRequest],
    queryFn: () => inspectModel(previewRequest!),
    enabled: previewRequest !== null,
    retry: false,
  });

  useEffect(() => {
    if (!selectedModel && modelsQuery.data?.models.length) {
      setSelectedModel(modelsQuery.data.models[0]);
    }
  }, [modelsQuery.data, selectedModel]);

  useEffect(() => {
    const firstPreset = presetsQuery.data?.presets[0]?.name;
    if (firstPreset && !selectedPreset) {
      setSelectedPreset(firstPreset);
      setOverrides({});
      setPreviewRequest({ model: selectedModel, preset: firstPreset, overrides: {} });
    }
  }, [presetsQuery.data, selectedModel, selectedPreset]);

  const graph = inspectQuery.data;
  const graphForDetail = useMemo(
    () => filterGraphByDetail(graph, graphDetailMode),
    [graph, graphDetailMode],
  );
  const graphNavigation = useMemo(() => buildGraphNavigation(graphForDetail), [graphForDetail]);
  const childSummariesById = useMemo(
    () => buildChildSummaries(graphForDetail, graphNavigation),
    [graphForDetail, graphNavigation],
  );
  const expertDiagramsById = useMemo(
    () => buildExpertDiagrams(graphForDetail, graphNavigation),
    [graphForDetail, graphNavigation],
  );
  const stackDiagramsById = useMemo(
    () => buildStackDiagrams(graphForDetail, graphNavigation),
    [graphForDetail, graphNavigation],
  );
  const activateGraphNode = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
    if (graphScope !== "opened") {
      return;
    }
    if (graphNavigation.rootIds.has(nodeId)) {
      return;
    }
    if ((graphNavigation.childrenById.get(nodeId)?.length ?? 0) === 0) {
      return;
    }
    setExpandedGraphNodeIds((current) => {
      const next = new Set(current);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, [graphNavigation, graphScope]);
  const toggleNodeDetails = useCallback((nodeId: string) => {
    setExpandedDetailNodeIds((current) => {
      const next = new Set(current);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);
  const graphForDisplay = useMemo(
    () => filterGraphByExpansion(graphForDetail, graphNavigation, expandedGraphNodeIds, graphScope),
    [expandedGraphNodeIds, graphForDetail, graphNavigation, graphScope],
  );
  const { nodes, edges } = useMemo(
    () =>
      layoutGraph(graphForDisplay, {
        navigation: graphNavigation,
        childSummariesById,
        expertDiagramsById,
        stackDiagramsById,
        expandedGraphNodeIds,
        expandedDetailNodeIds,
        enableExpansion: graphScope === "opened",
        selectedNodeId,
        onActivateNode: activateGraphNode,
        onToggleDetails: toggleNodeDetails,
      }),
    [
      activateGraphNode,
      expandedDetailNodeIds,
      expandedGraphNodeIds,
      graphForDisplay,
      graphNavigation,
      graphScope,
      childSummariesById,
      expertDiagramsById,
      stackDiagramsById,
      selectedNodeId,
      toggleNodeDetails,
    ],
  );
  const selectedPresetMeta = presetsQuery.data?.presets.find(
    (preset) => preset.name === selectedPreset,
  );
  const configSections = useMemo(() => {
    const groups = new Map<string, ConfigField[]>();
    for (const field of schemaQuery.data?.fields ?? []) {
      const section = field.section || "General";
      groups.set(section, [...(groups.get(section) ?? []), field]);
    }
    return Array.from(groups, ([title, fields]) => ({ title, fields }));
  }, [schemaQuery.data]);
  const selectedNode =
    (viewMode === "graph" ? graphForDisplay : graph)?.nodes.find(
      (node) => node.id === selectedNodeId,
    ) ?? (viewMode === "graph" ? graphForDisplay : graph)?.nodes[0];

  useEffect(() => {
    if (viewMode !== "graph" || !selectedNodeId || !graphForDisplay) {
      return;
    }
    if (!graphForDisplay.nodes.some((node) => node.id === selectedNodeId)) {
      setSelectedNodeId(null);
    }
  }, [graphForDisplay, selectedNodeId, viewMode]);

  const toggleConfigSection = (title: string) => {
    setOpenConfigSections((current) => {
      const next = new Set(current);
      if (next.has(title)) {
        next.delete(title);
      } else {
        next.add(title);
      }
      return next;
    });
  };

  // Changing the model resets every model-dependent piece of state. Doing this
  // in the event handler (rather than a useEffect keyed on selectedModel) keeps
  // the reset on the same path that triggers it. The initial model is picked by
  // the auto-select effect above, where no reset is needed (state is still empty).
  const selectModel = (model: string) => {
    setSelectedModel(model);
    setSelectedPreset("");
    setOverrides({});
    setSelectedNodeId(null);
    setExpandedGraphNodeIds(new Set());
    setExpandedDetailNodeIds(new Set());
    setOpenConfigSections(new Set());
  };

  const updateOverride = (key: string, value: string) => {
    setOverrides((current) => ({ ...current, [key]: value }));
  };

  const clearOverride = (key: string) => {
    setOverrides((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  };

  const updatePreview = () => {
    if (!selectedModel || !selectedPreset) {
      return;
    }
    setSelectedNodeId(null);
    setExpandedGraphNodeIds(new Set());
    setExpandedDetailNodeIds(new Set());
    setPreviewRequest({
      model: selectedModel,
      preset: selectedPreset,
      overrides: { ...overrides },
    });
  };

  const resetOverrides = () => {
    setOverrides({});
    setExpandedGraphNodeIds(new Set());
    setExpandedDetailNodeIds(new Set());
    if (selectedModel && selectedPreset) {
      setPreviewRequest({ model: selectedModel, preset: selectedPreset, overrides: {} });
    }
  };

  const collapseGraphNodes = () => {
    setExpandedGraphNodeIds(new Set());
    setSelectedNodeId(null);
  };

  const apiOnline = healthQuery.data?.status === "ok";
  const overrideCount = Object.keys(overrides).length;
  const fieldCount = configSections.reduce((total, section) => total + section.fields.length, 0);

  return {
    selectedModel,
    selectModel,
    selectedPreset,
    setSelectedPreset,
    overrides,
    setOverrides,
    viewMode,
    setViewMode,
    graphDetailMode,
    setGraphDetailMode,
    graphScope,
    setGraphScope,
    expandedGraphNodeIds,
    openConfigSections,
    selectedNodeId,
    setSelectedNodeId,
    modelsQuery,
    presetsQuery,
    schemaQuery,
    inspectQuery,
    graph,
    nodes,
    edges,
    selectedPresetMeta,
    configSections,
    selectedNode,
    toggleConfigSection,
    updateOverride,
    clearOverride,
    updatePreview,
    resetOverrides,
    collapseGraphNodes,
    apiOnline,
    overrideCount,
    fieldCount,
  };
}
