import { type ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { GraphChip } from "@/features/viewer/components/graph/graph-chip";
import { GraphParameterIndicators } from "@/features/viewer/components/graph/graph-parameter-indicators";
import { type ChildSummary, type ViewerNodeData } from "@/lib/graph";
import { cn } from "@/lib/utils";

export function GraphNodeChildSummaries({
  nodeId,
  summaries,
  monitorButton,
}: {
  nodeId: string;
  summaries: ViewerNodeData["childSummaries"];
  monitorButton?: ReactNode;
}) {
  const firstActivitySummaryIndex = summaries.findIndex(
    (summary) => summary.parameterActivity,
  );

  return (
    <div
      className="mt-2 grid min-h-6 shrink-0 content-start gap-2"
      data-testid={`child-summaries-${nodeId}`}
    >
      {summaries.map((summary, index) => (
        <GraphNodeChildSummaryRow
          key={`${summary.kind}-${summary.label}-${index}`}
          nodeId={nodeId}
          summary={summary}
          index={index}
          monitorButton={
            index === firstActivitySummaryIndex ? monitorButton : undefined
          }
        />
      ))}
    </div>
  );
}

function GraphNodeChildSummaryRow({
  nodeId,
  summary,
  index,
  monitorButton,
}: {
  nodeId: string;
  summary: ChildSummary;
  index: number;
  monitorButton?: ReactNode;
}) {
  const activity = summary.parameterActivity;
  const summaryLabel = summary.nestedLabel
    ? `${summary.label} ${summary.nestedLabel}`
    : summary.label;
  const summaryAccessibleLabel = summary.dims
    ? `${summaryLabel} ${summary.dims}`
    : summaryLabel;
  const summaryTitle = summary.title
    ? summary.dims
      ? `${summary.title} ${summary.dims}`
      : summary.title
    : summaryAccessibleLabel;

  return (
    <GraphChip
      aria-label={summaryAccessibleLabel}
      data-testid={`child-summary-${nodeId}-${index}`}
      title={summaryTitle}
      tone={summary.kind === "mechanism" ? "violet" : "default"}
      className={cn(
        "relative flex h-9 items-center gap-2 overflow-hidden rounded-[10px] px-3 text-[13px] font-medium",
        summary.kind === "mechanism"
          ? "bg-[linear-gradient(135deg,rgba(146,113,255,0.14),rgba(111,168,255,0.08))]"
          : summary.kind === "overflow"
            ? "bg-white/[0.035]"
            : undefined,
      )}
    >
      {summary.kind === "overflow" ? (
        <span className="w-full text-center tracking-[0.18em]">
          {summary.label}
        </span>
      ) : (
        <>
          {summary.nestedLabel ? (
            <span className="flex min-w-0 flex-1 items-center gap-1">
              <span className="shrink-0">{summary.label}</span>
              <ChevronRight
                className="h-3.5 w-3.5 shrink-0 text-ink-dim"
                aria-hidden
              />
              <span className="min-w-0 truncate">{summary.nestedLabel}</span>
            </span>
          ) : (
            <span className="min-w-0 flex-1 truncate">
              {summary.count ? `${summary.label} x${summary.count}` : summary.label}
            </span>
          )}
          {summary.dims && (
            <span className="shrink-0 whitespace-nowrap text-right font-mono text-[12px] font-semibold text-ink">
              {summary.dims}
            </span>
          )}
          {activity && (
            <span className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap">
              <GraphParameterIndicators activity={activity} />
              {monitorButton}
            </span>
          )}
        </>
      )}
    </GraphChip>
  );
}
