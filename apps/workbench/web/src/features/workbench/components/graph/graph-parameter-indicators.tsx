import {
  useEffect,
  useId,
  useRef,
  useState,
  type FocusEvent,
  type KeyboardEvent,
  type MouseEvent,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import {
  type GraphParameterActivity,
  type GraphParameterActivityChannel,
  type GraphParameterActivityStatus,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

export const graphParameterActivityStatusClassNames: Readonly<
  Record<GraphParameterActivityStatus, string>
> = {
  loading: "border-cyan/40 bg-cyan/[0.11] text-cyan",
  updated: "border-ok/35 bg-ok/10 text-ok",
  unchanged: "border-danger-line bg-danger-soft text-danger-text",
  mixed: "border-amber/40 bg-amber/[0.12] text-amber",
  missing: "border-line bg-white/[0.03] text-ink-faint",
  unknown: "border-line bg-white/[0.03] text-ink-faint",
};

const graphParameterActivityTextClassNames: Readonly<
  Record<GraphParameterActivityStatus, string>
> = {
  loading: "text-cyan",
  updated: "text-ok",
  unchanged: "text-danger-text",
  mixed: "text-amber",
  missing: "text-ink-faint",
  unknown: "text-ink-faint",
};

const statusCopy: Record<GraphParameterActivityStatus, string> = {
  loading:
    "Parameter activity is being loaded for this graph target.",
  updated:
    "This parameter was logged and at least one sampled point showed update or value-change evidence.",
  unchanged:
    "This parameter was logged with enough samples, but no update or value-change evidence was observed.",
  mixed:
    "Historical runs have mixed or incomplete update evidence for this parameter.",
  missing:
    "No tags for this parameter were found in the source, usually because the parameter does not exist or was not logged.",
  unknown:
    "There is no readable source yet, or there are not enough samples to decide whether this parameter changed.",
};

const POPUP_GAP = 6;
const POPUP_MARGIN = 8;
const POPUP_WIDTH = 312;
const ESTIMATED_POPUP_HEIGHT = 220;

type ActivityChannelView = {
  key: "weights" | "bias";
  label: "Weights" | "Bias";
  shortLabel: "W" | "b";
  channel: GraphParameterActivityChannel;
};

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

function activityChannels(activity: GraphParameterActivity): ActivityChannelView[] {
  return [
    {
      key: "weights",
      label: "Weights",
      shortLabel: "W",
      channel: activity.weights,
    },
    ...(activity.bias
      ? [
          {
            key: "bias" as const,
            label: "Bias" as const,
            shortLabel: "b" as const,
            channel: activity.bias,
          },
        ]
      : []),
  ];
}

export function parameterActivityLabel(activity: GraphParameterActivity) {
  return activityChannels(activity)
    .map(({ label, channel }) => `${label.toLowerCase()} ${channel.status}`)
    .join(", ");
}

function GraphParameterActivityIcons({
  activity,
}: {
  activity: GraphParameterActivity;
}) {
  return (
    <>
      {activityChannels(activity).map(({ key, shortLabel, channel }) => (
        <span
          key={key}
          aria-hidden="true"
          data-testid={`graph-parameter-indicator-${key}`}
          className={cn(
            "inline-flex h-5 min-w-3 shrink-0 items-center justify-center font-mono type-meta font-bold leading-none",
            graphParameterActivityTextClassNames[channel.status],
          )}
        >
          {shortLabel}
        </span>
      ))}
    </>
  );
}

function GraphParameterActivityPopover({
  activity,
  anchorRef,
  id,
  open,
}: {
  activity: GraphParameterActivity;
  anchorRef: RefObject<HTMLElement | null>;
  id: string;
  open: boolean;
}) {
  const popupRef = useRef<HTMLDivElement>(null);
  const [popupPosition, setPopupPosition] = useState<{
    left: number;
    top: number;
  } | null>(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    let scheduledFrame: number | null = null;
    let scheduledWithAnimationFrame = false;

    const measurePopupPosition = () => {
      scheduledFrame = null;
      const anchor = anchorRef.current;
      if (!anchor || typeof window === "undefined") {
        return;
      }

      const rect = anchor.getBoundingClientRect();
      const viewportWidth =
        document.documentElement.clientWidth || window.innerWidth;
      const viewportHeight =
        document.documentElement.clientHeight || window.innerHeight;
      const popupWidth = Math.min(
        POPUP_WIDTH,
        Math.max(0, viewportWidth - POPUP_MARGIN * 2),
      );
      const popupHeight =
        popupRef.current?.offsetHeight ?? ESTIMATED_POPUP_HEIGHT;
      const maxLeft = Math.max(
        POPUP_MARGIN,
        viewportWidth - popupWidth - POPUP_MARGIN,
      );
      const left = Math.min(
        Math.max(rect.right - popupWidth, POPUP_MARGIN),
        maxLeft,
      );
      const belowTop = rect.bottom + POPUP_GAP;
      const aboveTop = rect.top - popupHeight - POPUP_GAP;
      const top =
        belowTop + popupHeight <= viewportHeight - POPUP_MARGIN ||
        aboveTop < POPUP_MARGIN
          ? belowTop
          : aboveTop;

      setPopupPosition((current) =>
        current?.left === left && current.top === top
          ? current
          : { left, top },
      );
    };

    const schedulePopupPosition = () => {
      if (scheduledFrame !== null) {
        return;
      }
      if (typeof window.requestAnimationFrame === "function") {
        scheduledWithAnimationFrame = true;
        scheduledFrame = window.requestAnimationFrame(measurePopupPosition);
      } else {
        scheduledWithAnimationFrame = false;
        scheduledFrame = window.setTimeout(measurePopupPosition, 0);
      }
    };

    schedulePopupPosition();
    const resizeObserver =
      typeof ResizeObserver === "function"
        ? new ResizeObserver(schedulePopupPosition)
        : null;
    const anchor = anchorRef.current;
    const popup = popupRef.current;
    if (anchor) {
      resizeObserver?.observe(anchor);
    }
    if (popup) {
      resizeObserver?.observe(popup);
    }
    window.addEventListener("resize", schedulePopupPosition);
    window.addEventListener("scroll", schedulePopupPosition, true);

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener("resize", schedulePopupPosition);
      window.removeEventListener("scroll", schedulePopupPosition, true);
      if (scheduledFrame !== null) {
        if (scheduledWithAnimationFrame) {
          window.cancelAnimationFrame(scheduledFrame);
        } else {
          window.clearTimeout(scheduledFrame);
        }
      }
    };
  }, [anchorRef, open]);

  if (!open || typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div
      id={id}
      ref={popupRef}
      role="tooltip"
      style={{
        left: popupPosition?.left ?? 0,
        top: popupPosition?.top ?? 0,
        visibility: popupPosition ? "visible" : "hidden",
      }}
      className="pointer-events-none fixed z-[90] grid w-[19.5rem] max-w-[calc(100vw-1rem)] gap-2 rounded-control-md border border-line-soft bg-panel px-2.5 py-2 text-left font-sans type-meta font-semibold leading-4 text-ink shadow-panel"
      data-testid="graph-parameter-activity-popover"
    >
      <span className="text-ink">Parameter Activity</span>
      {activityChannels(activity).map(({ key, label, shortLabel, channel }) => {
        const counts = runCounts(channel);

        return (
          <span key={key} className="grid gap-1">
            <span className="flex min-w-0 items-center gap-1.5">
              <span
                className={cn(
                  "inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-chip border px-1.5 font-mono type-caption font-bold leading-none",
                  graphParameterActivityStatusClassNames[channel.status],
                )}
              >
                {shortLabel}
              </span>
              <span className="text-ink">{label}</span>
              <span className="font-mono text-ink-faint">{channel.status}</span>
            </span>
            <span className="text-ink-dim">{statusCopy[channel.status]}</span>
            <span className="font-mono text-ink-faint">{channel.sourceLabel}</span>
            <span className="font-mono text-ink-faint">
              step: {formatStep(channel.lastStep)} · samples:{" "}
              {channel.observedPoints}
            </span>
            {counts && <span className="font-mono text-ink-faint">{counts}</span>}
          </span>
        );
      })}
    </div>,
    document.body,
  );
}

export function GraphParameterIndicators({
  activity,
}: {
  activity?: GraphParameterActivity;
}) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popupId = useId();
  const [isPopupOpen, setIsPopupOpen] = useState(false);

  if (!activity) {
    return null;
  }

  const stopActivityPropagation = (
    event:
      | FocusEvent<HTMLButtonElement>
      | KeyboardEvent<HTMLButtonElement>
      | MouseEvent<HTMLButtonElement>,
  ) => {
    event.stopPropagation();
  };
  const openPopup = () => setIsPopupOpen(true);
  const closePopup = () => setIsPopupOpen(false);
  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    stopActivityPropagation(event);
    openPopup();
  };
  const handleFocus = (event: FocusEvent<HTMLButtonElement>) => {
    stopActivityPropagation(event);
    openPopup();
  };
  const handleBlur = (event: FocusEvent<HTMLButtonElement>) => {
    stopActivityPropagation(event);
    closePopup();
  };
  const handleMouseLeave = (event: MouseEvent<HTMLButtonElement>) => {
    stopActivityPropagation(event);
    closePopup();
  };
  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    stopActivityPropagation(event);

    if (event.key === "Escape") {
      event.preventDefault();
      closePopup();
      return;
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPopup();
    }
  };

  return (
    <>
      <button
        type="button"
        ref={triggerRef}
        aria-describedby={isPopupOpen ? popupId : undefined}
        aria-label={`Parameter activity: ${parameterActivityLabel(activity)}`}
        data-testid="graph-parameter-indicators"
        onBlur={handleBlur}
        onClick={handleClick}
        onFocus={handleFocus}
        onKeyDown={handleKeyDown}
        onMouseDown={stopActivityPropagation}
        onMouseLeave={handleMouseLeave}
        className="nodrag nopan inline-flex shrink-0 cursor-help items-center gap-1 whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      >
        <GraphParameterActivityIcons activity={activity} />
      </button>
      <GraphParameterActivityPopover
        activity={activity}
        anchorRef={triggerRef}
        id={popupId}
        open={isPopupOpen}
      />
    </>
  );
}
