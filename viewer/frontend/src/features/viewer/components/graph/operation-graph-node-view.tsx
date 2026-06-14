import { memo } from "react";
import { ChevronRight } from "lucide-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { type OperationFlowNodeData } from "@/lib/graph";
import { cn } from "@/lib/utils";

function detailValueText(value: unknown) {
  if (Array.isArray(value)) {
    return value.join(" x ");
  }
  if (value === null || value === undefined) {
    return undefined;
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function DetailChip({ label, value }: { label: string; value: unknown }) {
  const text = detailValueText(value);
  if (!text) {
    return null;
  }
  return (
    <span
      title={`${label}: ${text}`}
      className="inline-flex max-w-full items-center gap-1 rounded-[6px] border border-white/10 bg-white/[0.04] px-2 py-1 font-mono text-[10.5px] font-semibold text-ink-dim"
    >
      <span className="uppercase text-ink-faint">{label}</span>
      <span className="truncate text-ink">{text}</span>
    </span>
  );
}

export const OperationGraphNodeView = memo(function OperationGraphNodeView({
  data,
  selected,
}: NodeProps<Node<OperationFlowNodeData>>) {
  if (data.kind === "group") {
    return (
      <div
        role="button"
        tabIndex={0}
        aria-label={`Expand operation group ${data.subtitle}`}
        onClick={data.onActivateNode}
        onKeyDown={(event) => {
          if (event.key !== "Enter" && event.key !== " ") {
            return;
          }
          event.preventDefault();
          data.onActivateNode();
        }}
        className={cn(
          "nodrag nopan edge flex h-full w-full flex-col justify-between overflow-hidden rounded-card px-5 py-4 shadow-[0_18px_40px_-28px_rgba(0,0,0,0.95)] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
          selected ? "edge-sel" : "hover:brightness-110",
        )}
      >
        <Handle type="target" position={Position.Left} />
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-[14px] font-extrabold text-ink">
              {data.label}
            </div>
            <div className="mt-1 truncate font-mono text-[11px] text-ink-dim">
              {data.subtitle}
            </div>
          </div>
          <span
            aria-hidden
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border border-cyan-200/25 bg-cyan-300/[0.08] text-cyan-100"
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </span>
        </div>
        <div className="font-mono text-[11px] font-semibold uppercase text-ink-faint">
          {data.operationCount} {data.operationCount === 1 ? "op" : "ops"}
        </div>
        <Handle type="source" position={Position.Right} />
      </div>
    );
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`Select operation ${data.nodeId}`}
      onClick={data.onActivateNode}
      onKeyDown={(event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }
        event.preventDefault();
        data.onActivateNode();
      }}
      className={cn(
        "nodrag nopan edge flex h-full w-full flex-col overflow-hidden rounded-card px-5 py-4 shadow-[0_18px_40px_-28px_rgba(0,0,0,0.95)] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        selected ? "edge-sel" : "hover:brightness-110",
      )}
    >
      <Handle type="target" position={Position.Left} />
      <div className="min-w-0">
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-[13px] font-extrabold text-ink">
              {data.label}
            </div>
            <div className="mt-1 truncate font-mono text-[11px] text-cyan-100/80">
              {data.opKind}
            </div>
          </div>
          <span className="shrink-0 rounded-[6px] border border-white/10 bg-white/[0.04] px-2 py-1 font-mono text-[10px] font-bold uppercase text-ink-dim">
            {data.nodeId}
          </span>
        </div>
        <div className="mt-3 line-clamp-2 break-all font-mono text-[11px] leading-snug text-ink-dim">
          {data.target}
        </div>
      </div>
      <div className="mt-auto grid gap-2 pt-3">
        {data.modulePath && (
          <div className="truncate font-mono text-[10.5px] text-ink-faint">
            {data.modulePath}
          </div>
        )}
        <div className="flex min-w-0 flex-wrap gap-1.5">
          <DetailChip label="shape" value={data.details.shape} />
          <DetailChip label="dtype" value={data.details.dtype} />
          <DetailChip label="kind" value={data.details.inputKind} />
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
});
