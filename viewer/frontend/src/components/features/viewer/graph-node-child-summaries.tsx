import { ChevronRight } from "lucide-react";
import { type ViewerNodeData } from "@/lib/graph";
import { cn } from "@/lib/utils";

export function GraphNodeChildSummaries({
  nodeId,
  summaries,
}: {
  nodeId: string;
  summaries: ViewerNodeData["childSummaries"];
}) {
  return (
    <div
      className="mt-2 grid min-h-6 shrink-0 content-start gap-2"
      data-testid={`child-summaries-${nodeId}`}
    >
      {summaries.map((summary, index) => {
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
          <div
            key={`${summary.kind}-${summary.label}-${index}`}
            aria-label={summaryAccessibleLabel}
            title={summaryTitle}
            className={cn(
              "relative flex h-9 items-center gap-2 overflow-hidden rounded-[10px] border px-3 text-[13px] font-medium leading-none",
              summary.kind === "mechanism"
                ? "border-violet/30 bg-[linear-gradient(135deg,rgba(146,113,255,0.14),rgba(111,168,255,0.08))] text-[#d7c9ff] shadow-[inset_0_-1px_0_rgba(146,113,255,0.24)]"
                : summary.kind === "overflow"
                  ? "border-line-soft bg-white/[0.035] text-ink-dim"
                  : "border-line-soft bg-white/[0.02] text-ink-dim",
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
                  <span className="shrink-0 text-right font-mono text-[12px] font-semibold text-ink">
                    {summary.dims}
                  </span>
                )}
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
