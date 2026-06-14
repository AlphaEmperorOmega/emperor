import { useCallback, useMemo } from "react";
import { ChevronRight } from "lucide-react";
import { type InspectResponse } from "@/lib/api";
import {
  buildParameterFocusData,
  type ParameterFocusData,
  type ParameterFocusNodeSummary,
} from "@/lib/echarts/parameter-treemap-options";
import {
  formatCompactCount,
  formatExactCount,
  formatModelSize,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

function metricTitle(value: number, suffix: string) {
  return `${formatExactCount(value)} ${suffix}`;
}

function StatCell({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <div className="min-w-0 rounded-[7px] border border-white/[0.07] bg-black/20 px-2.5 py-2">
      <div className="text-[10px] font-bold uppercase tracking-[0.08em] text-ink-faint">
        {label}
      </div>
      <div
        title={title}
        className="mt-1 truncate font-mono text-[12px] font-semibold text-ink"
      >
        {value}
      </div>
    </div>
  );
}

function RolePill({ role }: { role: ParameterFocusNodeSummary["graphRole"] }) {
  const roleClassName =
    role === "architecture"
      ? "border-violet/35 bg-violet/15 text-violet-text"
      : role === "internal"
        ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100"
        : "border-amber-300/35 bg-amber-300/10 text-amber-100";

  return (
    <span
      className={cn(
        "inline-flex h-5 items-center rounded-[5px] border px-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.06em]",
        roleClassName,
      )}
    >
      {role}
    </span>
  );
}

function Breadcrumbs({
  ancestors,
  focusNodeId,
  focusedNode,
  onFocusNode,
}: {
  ancestors: ParameterFocusNodeSummary[];
  focusNodeId: string | null;
  focusedNode: ParameterFocusNodeSummary | null;
  onFocusNode: (nodeId: string | null) => void;
}) {
  const showFocusedCrumb = Boolean(focusNodeId && focusedNode);

  return (
    <nav
      aria-label="Parameter focus"
      className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto"
    >
      <button
        type="button"
        aria-current={focusNodeId ? undefined : "page"}
        onClick={() => onFocusNode(null)}
        className={cn(
          "shrink-0 rounded-[6px] px-2 py-1 font-mono text-[11px] font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
          focusNodeId
            ? "text-ink-dim hover:bg-white/[0.06] hover:text-ink"
            : "bg-white/[0.08] text-ink",
        )}
      >
        Root
      </button>
      {[...ancestors, ...(showFocusedCrumb && focusedNode ? [focusedNode] : [])].map(
        (node, index, nodes) => {
          const isCurrent = index === nodes.length - 1;
          return (
            <div key={node.id} className="flex min-w-0 shrink-0 items-center gap-1">
              <ChevronRight className="h-3 w-3 text-ink-faint" aria-hidden />
              <button
                type="button"
                aria-current={isCurrent ? "page" : undefined}
                onClick={() => onFocusNode(node.id)}
                className={cn(
                  "max-w-[180px] truncate rounded-[6px] px-2 py-1 font-mono text-[11px] font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                  isCurrent
                    ? "bg-white/[0.08] text-ink"
                    : "text-ink-dim hover:bg-white/[0.06] hover:text-ink",
                )}
              >
                {node.path}
              </button>
            </div>
          );
        },
      )}
    </nav>
  );
}

function ChildButton({
  child,
  selected,
  onSelect,
}: {
  child: ParameterFocusNodeSummary;
  selected: boolean;
  onSelect: (child: ParameterFocusNodeSummary) => void;
}) {
  return (
    <button
      type="button"
      aria-label={`select component ${child.path}`}
      onClick={() => onSelect(child)}
      className={cn(
        "grid min-w-0 gap-1 rounded-[7px] border px-2.5 py-2 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        selected
          ? "border-cyan-200/60 bg-cyan-300/[0.11] shadow-[0_0_20px_rgba(34,211,238,0.12)]"
          : "border-white/[0.07] bg-white/[0.035] hover:border-white/[0.16] hover:bg-white/[0.06]",
      )}
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span className="min-w-0 truncate text-[12px] font-bold text-ink">
          {child.label}
        </span>
        {child.childCount > 0 && (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-ink-dim" aria-hidden />
        )}
      </div>
      <div className="flex min-w-0 items-center gap-2">
        <RolePill role={child.graphRole} />
        <span className="min-w-0 truncate font-mono text-[11px] text-ink-dim">
          {child.typeName}
        </span>
      </div>
      <div className="flex min-w-0 items-center justify-between gap-2 font-mono text-[11px] text-ink-dim">
        <span
          title={metricTitle(child.parameterCount, "parameters")}
          className={cn(child.hasParameters && "text-cyan-100")}
        >
          {child.hasParameters ? formatCompactCount(child.parameterCount) : "0"}
        </span>
        {child.dimText && <span className="truncate">{child.dimText}</span>}
      </div>
    </button>
  );
}

function ChildSection({
  label,
  items,
  selectedNodeId,
  onSelectChild,
}: {
  label: string;
  items: ParameterFocusNodeSummary[];
  selectedNodeId: string | null;
  onSelectChild: (child: ParameterFocusNodeSummary) => void;
}) {
  if (items.length === 0) {
    return null;
  }

  return (
    <section className="grid gap-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-faint">
          {label}
        </h3>
        <span className="font-mono text-[10px] text-ink-faint">{items.length}</span>
      </div>
      <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
        {items.map((child) => (
          <ChildButton
            key={child.id}
            child={child}
            selected={child.id === selectedNodeId}
            onSelect={onSelectChild}
          />
        ))}
      </div>
    </section>
  );
}

function FocusDetails({
  data,
}: {
  data: ParameterFocusData;
}) {
  const focusedNode = data.focusedNode;
  const memoryText = formatModelSize(focusedNode?.parameterSizeBytes) ?? "0 MB";

  return (
    <section className="grid gap-2">
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-faint">
            Focus
          </div>
          <div
            data-testid="parameter-browser-focus"
            className="mt-1 truncate text-[14px] font-bold text-ink"
            title={focusedNode?.path}
          >
            {focusedNode?.label ?? "Root"}
          </div>
          <div className="mt-0.5 truncate font-mono text-[11px] text-ink-dim">
            {focusedNode?.path ?? "root"}
          </div>
        </div>
        {focusedNode && <RolePill role={focusedNode.graphRole} />}
      </div>
      <div className="grid grid-cols-2 gap-1.5 md:grid-cols-4">
        <StatCell
          label="Params"
          value={formatCompactCount(data.focusedParameterCount)}
          title={metricTitle(data.focusedParameterCount, "parameters")}
        />
        <StatCell label="Memory" value={memoryText} />
        <StatCell label="Type" value={focusedNode?.typeName ?? "Root"} />
        <StatCell label="Children" value={String(data.immediateChildren.length)} />
      </div>
      {focusedNode?.dimText && (
        <div className="rounded-[7px] border border-white/[0.07] bg-black/20 px-2.5 py-2 font-mono text-[11px] text-ink-dim">
          dims: <span className="text-ink">{focusedNode.dimText}</span>
        </div>
      )}
      {focusedNode && focusedNode.badges.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {focusedNode.badges.map((badge) => (
            <span
              key={`${badge.key}-${badge.value}`}
              className="rounded-[5px] border border-white/[0.08] bg-white/[0.04] px-1.5 py-1 font-mono text-[10px] text-ink-dim"
            >
              {badge.key}: <span className="text-ink">{badge.value}</span>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

export function ParameterBrowserPanel({
  graph,
  selectedNodeId,
  focusNodeId,
  onFocusNode,
  onRevealNode,
}: {
  graph: InspectResponse | undefined;
  selectedNodeId: string | null;
  focusNodeId: string | null;
  onFocusNode: (nodeId: string | null) => void;
  onRevealNode: (nodeId: string) => void;
}) {
  const data = useMemo(
    () => buildParameterFocusData(graph, focusNodeId),
    [focusNodeId, graph],
  );
  const parameterChildren = data.immediateChildren.filter((child) => child.hasParameters);
  const handleChild = useCallback((child: ParameterFocusNodeSummary) => {
    onRevealNode(child.id);
    if (child.childCount > 0) {
      onFocusNode(child.id);
    }
  }, [onFocusNode, onRevealNode]);

  return (
    <div
      data-testid="parameter-browser-panel"
      className="grid h-full min-h-0 w-full grid-rows-[auto_minmax(0,1fr)] overflow-hidden bg-[linear-gradient(180deg,rgba(13,15,24,0.88),rgba(6,7,12,0.98))]"
    >
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-white/[0.08] bg-black/[0.18] px-3 py-2 backdrop-blur">
        <Breadcrumbs
          ancestors={data.ancestors}
          focusNodeId={data.focusNodeId}
          focusedNode={data.focusedNode}
          onFocusNode={onFocusNode}
        />
        <div
          className="hidden shrink-0 font-mono text-[11px] text-ink-dim md:block"
          title={metricTitle(data.totalParameterCount, "parameters")}
        >
          {formatCompactCount(data.totalParameterCount)}
        </div>
      </div>
      <main
        data-testid="parameter-browser-inspector"
        className="min-h-0 overflow-y-auto bg-black/[0.14] p-3"
      >
        <div className="grid gap-4">
          <FocusDetails data={data} />
          <ChildSection
            label="Children"
            items={parameterChildren}
            selectedNodeId={selectedNodeId}
            onSelectChild={handleChild}
          />
          <ChildSection
            label="No params"
            items={data.zeroParameterChildren}
            selectedNodeId={selectedNodeId}
            onSelectChild={handleChild}
          />
        </div>
      </main>
    </div>
  );
}
