import {
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { CircleDot, Weight } from "lucide-react";
import {
  type GraphParameterActivityChannel,
  type GraphParameterActivity,
  type GraphParameterActivityStatus,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

const statusClassNames: Record<GraphParameterActivityStatus, string> = {
  updated: "border-ok/35 bg-ok/10 text-ok",
  unchanged: "border-danger-line bg-danger-soft text-[#fda4af]",
  mixed: "border-amber/40 bg-amber/[0.12] text-amber",
  missing: "border-line bg-white/[0.03] text-ink-faint",
  unknown: "border-line bg-white/[0.03] text-ink-faint",
};

const statusCopy: Record<GraphParameterActivityStatus, string> = {
  updated:
    "This parameter was logged and at least one sampled point showed update or value-change evidence.",
  unchanged:
    "This parameter was logged with enough samples, but no update or value-change evidence was observed.",
  mixed:
    "At least one historical run showed update evidence, but at least one other run did not.",
  missing:
    "No tags for this parameter were found in the source, usually because the parameter does not exist or was not logged.",
  unknown:
    "There is no readable source yet, or there are not enough samples to decide whether this parameter changed.",
};

const TOOLTIP_GAP = 6;
const TOOLTIP_MARGIN = 8;
const TOOLTIP_WIDTH = 288;
const ESTIMATED_TOOLTIP_HEIGHT = 116;

function formatStep(step: number | null | undefined) {
  return typeof step === "number" ? String(step) : "n/a";
}

function runCounts(channel: GraphParameterActivityChannel) {
  if (!channel.totalRuns) {
    return null;
  }
  return `${channel.updatedRuns ?? 0} updated / ${channel.unchangedRuns ?? 0} unchanged / ${
    channel.missingRuns ?? 0
  } missing / ${channel.unknownRuns ?? 0} unknown`;
}

function ParameterIndicator({
  label,
  channel,
  icon,
}: {
  label: "Weights" | "Bias";
  channel: GraphParameterActivityChannel;
  icon: ReactNode;
}) {
  const indicatorRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<{
    left: number;
    top: number;
  } | null>(null);
  const accessibleLabel = `${label} parameter activity: ${channel.status}`;
  const counts = runCounts(channel);
  const stopPropagation = (event: MouseEvent | KeyboardEvent) => {
    event.stopPropagation();
  };

  useLayoutEffect(() => {
    if (!tooltipVisible) {
      setTooltipPosition(null);
      return undefined;
    }

    const updateTooltipPosition = () => {
      const indicator = indicatorRef.current;
      if (!indicator || typeof window === "undefined") {
        return;
      }

      const rect = indicator.getBoundingClientRect();
      const viewportWidth =
        document.documentElement.clientWidth || window.innerWidth;
      const viewportHeight =
        document.documentElement.clientHeight || window.innerHeight;
      const tooltipWidth = Math.min(
        TOOLTIP_WIDTH,
        Math.max(0, viewportWidth - TOOLTIP_MARGIN * 2),
      );
      const tooltipHeight =
        tooltipRef.current?.offsetHeight ?? ESTIMATED_TOOLTIP_HEIGHT;
      const maxLeft = Math.max(
        TOOLTIP_MARGIN,
        viewportWidth - tooltipWidth - TOOLTIP_MARGIN,
      );
      const left = Math.min(
        Math.max(rect.right - tooltipWidth, TOOLTIP_MARGIN),
        maxLeft,
      );
      const belowTop = rect.bottom + TOOLTIP_GAP;
      const aboveTop = rect.top - tooltipHeight - TOOLTIP_GAP;
      const top =
        belowTop + tooltipHeight <= viewportHeight - TOOLTIP_MARGIN ||
        aboveTop < TOOLTIP_MARGIN
          ? belowTop
          : aboveTop;

      setTooltipPosition({ left, top });
    };

    updateTooltipPosition();
    window.addEventListener("resize", updateTooltipPosition);
    window.addEventListener("scroll", updateTooltipPosition, true);

    return () => {
      window.removeEventListener("resize", updateTooltipPosition);
      window.removeEventListener("scroll", updateTooltipPosition, true);
    };
  }, [tooltipVisible]);

  const tooltip =
    tooltipVisible && typeof document !== "undefined"
      ? createPortal(
          <span
            ref={tooltipRef}
            role="tooltip"
            style={{
              left: tooltipPosition?.left ?? 0,
              top: tooltipPosition?.top ?? 0,
              visibility: tooltipPosition ? "visible" : "hidden",
            }}
            className="pointer-events-none fixed z-[90] grid w-72 max-w-[calc(100vw-1rem)] gap-1 rounded-[8px] border border-line-soft bg-panel px-2.5 py-2 text-left font-sans text-[11px] font-semibold leading-4 text-ink shadow-panel"
          >
            <span className="text-ink">{label}</span>
            <span className="text-ink-dim">{statusCopy[channel.status]}</span>
            <span className="font-mono text-ink-faint">{channel.sourceLabel}</span>
            <span className="font-mono text-ink-faint">
              step: {formatStep(channel.lastStep)} - samples:{" "}
              {channel.observedPoints}
            </span>
            {counts && <span className="font-mono text-ink-faint">{counts}</span>}
          </span>,
          document.body,
        )
      : null;

  return (
    <span
      ref={indicatorRef}
      role="img"
      aria-label={accessibleLabel}
      tabIndex={0}
      onBlur={() => setTooltipVisible(false)}
      onClick={stopPropagation}
      onFocus={() => setTooltipVisible(true)}
      onKeyDown={stopPropagation}
      onMouseDown={stopPropagation}
      onMouseEnter={() => setTooltipVisible(true)}
      onMouseLeave={() => setTooltipVisible(false)}
      className={cn(
        "nodrag nopan relative inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        statusClassNames[channel.status],
      )}
    >
      {icon}
      {tooltip}
    </span>
  );
}

export function GraphParameterIndicators({
  activity,
}: {
  activity?: GraphParameterActivity;
}) {
  if (!activity) {
    return null;
  }

  return (
    <span className="flex shrink-0 items-center gap-1" data-testid="graph-parameter-indicators">
      <ParameterIndicator
        label="Weights"
        channel={activity.weights}
        icon={<Weight className="h-3.5 w-3.5" aria-hidden />}
      />
      {activity.bias && (
        <ParameterIndicator
          label="Bias"
          channel={activity.bias}
          icon={<CircleDot className="h-3.5 w-3.5" aria-hidden />}
        />
      )}
    </span>
  );
}
