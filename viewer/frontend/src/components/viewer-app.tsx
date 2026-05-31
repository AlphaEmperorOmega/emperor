"use client";

import {
  Activity,
  CheckCircle2,
  GitBranch,
  Layers,
  ListTree,
  RefreshCw,
  RotateCcw,
  Server,
  SlidersHorizontal,
} from "lucide-react";
import { Background, Controls, ReactFlow } from "@xyflow/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { ConfigSectionAccordion } from "@/components/viewer/config-panel";
import { EmptyState } from "@/components/viewer/empty-state";
import { ErrorPanel } from "@/components/viewer/error-panel";
import { nodeTypes } from "@/components/viewer/graph-node-view";
import { HierarchyView } from "@/components/viewer/hierarchy-view";
import { SelectedNodeDetails } from "@/components/viewer/selected-node-details";
import { StatusPill } from "@/components/viewer/status-pill";
import { useViewerState } from "@/components/viewer/use-viewer-state";
import { ViewModeButton } from "@/components/viewer/view-mode-button";
import { formatCompactCount, formatExactCount } from "@/lib/graph";
import { errorMessage } from "@/lib/utils";

export function ViewerApp() {
  const {
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
  } = useViewerState();

  return (
    <main className="grid min-h-screen grid-rows-[auto_1fr] bg-surface text-ink lg:h-screen lg:min-h-[720px] lg:overflow-hidden">
      <header className="flex min-h-16 flex-wrap items-center justify-between gap-3 border-b border-border bg-panel px-4 py-3 shadow-panel lg:flex-nowrap">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-border bg-surface">
            <Layers className="h-5 w-5 text-accent" aria-hidden />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold text-ink">Emperor Model Viewer</h1>
            <div className="mt-0.5 truncate font-mono text-xs text-muted">
              {selectedModel || "No model"} {selectedPreset ? `/ ${selectedPreset}` : ""}
            </div>
          </div>
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <StatusPill
            icon={
              apiOnline ? <CheckCircle2 className="h-4 w-4" /> : <Server className="h-4 w-4" />
            }
            label="API"
            value={apiOnline ? "online" : "offline"}
            tone={apiOnline ? "good" : "danger"}
          />
          <StatusPill icon={<Activity className="h-4 w-4" />} label="nodes" value={graph?.nodes.length ?? 0} />
          <StatusPill icon={<GitBranch className="h-4 w-4" />} label="edges" value={graph?.edges.length ?? 0} />
          <StatusPill
            icon={<SlidersHorizontal className="h-4 w-4" />}
            label="overrides"
            value={overrideCount}
            tone={overrideCount > 0 ? "warn" : "neutral"}
          />
          <Button variant="secondary" onClick={resetOverrides} disabled={!selectedModel}>
            <RotateCcw className="h-4 w-4" aria-hidden />
            Reset Overrides
          </Button>
          <Button variant="primary" onClick={updatePreview} disabled={!selectedModel || !selectedPreset}>
            <RefreshCw className="h-4 w-4" aria-hidden />
            Update Preview
          </Button>
        </div>
      </header>

      <section className="grid min-h-0 grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] xl:grid-cols-[380px_minmax(0,1fr)_320px]">
        <aside className="min-h-0 overflow-y-auto border-b border-border bg-panel p-4 lg:border-b-0 lg:border-r">
          <div className="grid gap-4">
            {modelsQuery.isError && (
              <ErrorPanel title="Backend unavailable" message={errorMessage(modelsQuery.error)} />
            )}

            <section className="grid gap-3 rounded-md border border-border bg-surface p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="text-xs font-semibold uppercase tracking-[0.08em] text-muted">
                  Target
                </div>
                <Badge>{presetsQuery.data?.presets.length ?? 0} presets</Badge>
              </div>
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted">Model</span>
                <Select
                  aria-label="model"
                  name="model"
                  autoComplete="off"
                  value={selectedModel}
                  onChange={(event) => selectModel(event.target.value)}
                >
                  {(modelsQuery.data?.models ?? []).map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </Select>
              </label>
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted">Preset</span>
                <Select
                  aria-label="preset"
                  name="preset"
                  autoComplete="off"
                  value={selectedPreset}
                  onChange={(event) => {
                    setSelectedPreset(event.target.value);
                    setOverrides({});
                  }}
                >
                  {(presetsQuery.data?.presets ?? []).map((preset) => (
                    <option key={preset.name} value={preset.name}>
                      {preset.name}
                    </option>
                  ))}
                </Select>
              </label>
              {selectedPresetMeta?.description && (
                <div className="rounded-md border border-border bg-panel p-2.5 text-xs leading-5 text-muted">
                  {selectedPresetMeta.description}
                </div>
              )}
            </section>

            {presetsQuery.isError && (
              <ErrorPanel title="Model import failed" message={errorMessage(presetsQuery.error)} />
            )}
            {schemaQuery.isError && (
              <ErrorPanel title="Config schema failed" message={errorMessage(schemaQuery.error)} />
            )}

            <div className="grid gap-3">
              <div className="flex items-center justify-between gap-3 border-t border-border pt-4">
                <div className="text-xs font-semibold uppercase tracking-[0.08em] text-muted">
                  Config
                </div>
                <div className="flex items-center gap-1">
                  <Badge>{configSections.length} sections</Badge>
                  <Badge>{fieldCount} fields</Badge>
                </div>
              </div>
              {schemaQuery.isLoading && (
                <div className="rounded-md border border-dashed border-faint bg-surface p-3 text-sm text-muted">
                  Loading config schema…
                </div>
              )}
              {!schemaQuery.isLoading && configSections.length === 0 && (
                <div className="rounded-md border border-dashed border-faint bg-surface p-3 text-sm text-muted">
                  No config fields
                </div>
              )}
              {configSections.map((section) => (
                <ConfigSectionAccordion
                  key={section.title}
                  title={section.title}
                  fields={section.fields}
                  isOpen={openConfigSections.has(section.title)}
                  overrides={overrides}
                  onToggle={() => toggleConfigSection(section.title)}
                  onFieldChange={updateOverride}
                  onFieldReset={clearOverride}
                />
              ))}
            </div>
          </div>
        </aside>

        <div className="grid min-h-[560px] grid-rows-[auto_1fr] bg-surface lg:min-h-0">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-panel px-4 py-2">
            <div className="min-w-0">
              <div className="text-xs font-semibold uppercase tracking-[0.08em] text-muted">
                Preview
              </div>
              <div className="mt-0.5 truncate text-xs text-muted">
                {graph ? `${graph.model} / ${graph.preset}` : "Waiting for preview data"}
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              {viewMode === "graph" && (
                <>
                  <div
                    role="tablist"
                    aria-label="Graph detail"
                    className="inline-flex rounded-md border border-border bg-surface p-1"
                  >
                    <ViewModeButton
                      active={graphDetailMode === "basic"}
                      onClick={() => setGraphDetailMode("basic")}
                    >
                      Basic
                    </ViewModeButton>
                    <ViewModeButton
                      active={graphDetailMode === "full"}
                      onClick={() => setGraphDetailMode("full")}
                    >
                      Full
                    </ViewModeButton>
                  </div>
                  <div
                    role="tablist"
                    aria-label="Graph scope"
                    className="inline-flex rounded-md border border-border bg-surface p-1"
                  >
                    <ViewModeButton
                      active={graphScope === "opened"}
                      onClick={() => setGraphScope("opened")}
                    >
                      Opened
                    </ViewModeButton>
                    <ViewModeButton
                      active={graphScope === "entire"}
                      onClick={() => setGraphScope("entire")}
                    >
                      Entire
                    </ViewModeButton>
                  </div>
                  <Button
                    variant="secondary"
                    onClick={collapseGraphNodes}
                    disabled={!graph || expandedGraphNodeIds.size === 0}
                    className="h-8 px-2.5 text-xs"
                  >
                    <RotateCcw className="h-3.5 w-3.5" aria-hidden />
                    Collapse All
                  </Button>
                </>
              )}
              <div
                role="tablist"
                aria-label="Visualization mode"
                className="inline-flex rounded-md border border-border bg-surface p-1"
              >
                <ViewModeButton active={viewMode === "graph"} onClick={() => setViewMode("graph")}>
                  <GitBranch className="h-4 w-4" aria-hidden />
                  Graph
                </ViewModeButton>
                <ViewModeButton
                  active={viewMode === "hierarchy"}
                  onClick={() => setViewMode("hierarchy")}
                >
                  <ListTree className="h-4 w-4" aria-hidden />
                  Hierarchy
                </ViewModeButton>
              </div>
            </div>
          </div>
          <div className="relative min-h-0 overflow-hidden">
            {inspectQuery.isError && (
              <div className="absolute left-4 right-4 top-4 z-10">
                <ErrorPanel title="Preview failed" message={errorMessage(inspectQuery.error)} />
              </div>
            )}
            {inspectQuery.isFetching && (
              <div className="absolute left-4 top-4 z-10">
                <StatusPill
                  icon={<RefreshCw className="h-4 w-4 motion-safe:animate-spin" />}
                  label="preview"
                  value="building"
                  tone="warn"
                />
              </div>
            )}
            {viewMode === "graph" ? (
              <>
                <ReactFlow
                  nodes={nodes}
                  edges={edges}
                  nodeTypes={nodeTypes}
                  fitView
                  minZoom={0.15}
                  maxZoom={1.5}
                  onNodeClick={(_, node) => setSelectedNodeId(node.id)}
                >
                  <Background gap={24} color="#d5ddd8" />
                  <Controls showInteractive={false} />
                </ReactFlow>
                {!graph && !inspectQuery.isFetching && !inspectQuery.isError && (
                  <EmptyState
                    title="No graph loaded"
                    detail="Preview data has not returned yet."
                    icon={<GitBranch className="h-4 w-4" aria-hidden />}
                  />
                )}
              </>
            ) : (
              <HierarchyView
                graph={graph}
                selectedNodeId={selectedNodeId}
                onSelectNode={setSelectedNodeId}
              />
            )}
          </div>
        </div>

        <aside className="min-h-0 overflow-y-auto border-t border-border bg-panel p-4 lg:col-span-2 xl:col-span-1 xl:border-l xl:border-t-0">
          <div className="mb-4 grid gap-3">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-sm font-semibold text-ink">Node Details</h2>
            </div>
            {graph && (
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md border border-border bg-surface p-2">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
                    Params
                  </div>
                  <div
                    className="mt-1 truncate text-base font-semibold text-ink"
                    title={`${formatExactCount(graph.parameterCount)} parameters`}
                  >
                    {formatCompactCount(graph.parameterCount)}
                  </div>
                </div>
                <div className="rounded-md border border-border bg-surface p-2">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
                    Nodes
                  </div>
                  <div className="mt-1 text-base font-semibold text-ink">{graph.nodes.length}</div>
                </div>
                <div className="rounded-md border border-border bg-surface p-2">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
                    Edges
                  </div>
                  <div className="mt-1 text-base font-semibold text-ink">{graph.edges.length}</div>
                </div>
                <div className="rounded-md border border-border bg-surface p-2">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
                    View
                  </div>
                  <div className="mt-1 truncate text-base font-semibold capitalize text-ink">
                    {viewMode}
                  </div>
                </div>
              </div>
            )}
          </div>
          <SelectedNodeDetails node={selectedNode} />
        </aside>
      </section>
    </main>
  );
}
