import { Terminal } from "lucide-react";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import { SurfacePanel } from "@/components/ui/surface-panel";

export const TRAINING_LOG_TAIL_LINE_LIMIT = 200;
export const TRAINING_LOG_TAIL_CHAR_LIMIT = 20_000;

const footerIconClass = "h-[15px] w-[15px] text-violet";

function lineCountLabel(count: number) {
  return `${count} line${count === 1 ? "" : "s"}`;
}

export function TrainingLogTailCard({ logTail = [] }: { logTail?: string[] }) {
  const logTailLines = logTail.length ? logTail : ["No log output yet"];
  const boundedLogTailLines = logTailLines.slice(-TRAINING_LOG_TAIL_LINE_LIMIT);
  const boundedLogTailText = boundedLogTailLines
    .join("\n")
    .slice(-TRAINING_LOG_TAIL_CHAR_LIMIT);
  const displayedLineCount = Math.min(
    logTail.length,
    TRAINING_LOG_TAIL_LINE_LIMIT,
  );

  return (
    <SurfacePanel
      icon={<Terminal className={footerIconClass} aria-hidden />}
      title="Log Tail"
      detail={<StatChip>{lineCountLabel(displayedLineCount)}</StatChip>}
    >
      <pre className="max-h-36 overflow-y-auto overflow-x-hidden whitespace-pre-wrap rounded-control border border-line bg-black/25 p-2 font-mono text-xs leading-5 text-ink-dim [overflow-wrap:anywhere]">
        {boundedLogTailText}
      </pre>
    </SurfacePanel>
  );
}
