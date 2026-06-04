import { type RefObject } from "react";
import { Database } from "lucide-react";
import { cn } from "@/lib/utils";

export function DatasetSelectorTrigger({
  datasetTriggerRef,
  datasetSelectorId,
  isOpen,
  disabled,
  datasetCount,
  onToggle,
}: {
  datasetTriggerRef: RefObject<HTMLButtonElement | null>;
  datasetSelectorId: string;
  isOpen: boolean;
  disabled: boolean;
  datasetCount: string;
  onToggle: () => void;
}) {
  return (
    <section className="grid gap-3">
      <button
        ref={datasetTriggerRef}
        type="button"
        className={cn(
          "grid min-h-[46px] w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 rounded-[12px] border px-3 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
          isOpen
            ? "border-violet/40 bg-[linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))]"
            : "border-line-soft bg-white/[0.018] hover:border-line hover:bg-white/[0.04]",
          disabled &&
            "cursor-not-allowed opacity-50 hover:border-line-soft hover:bg-white/[0.018]",
        )}
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        aria-controls={datasetSelectorId}
        onClick={onToggle}
      >
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-[9px] border border-line bg-white/[0.035]">
          <Database className="h-[15px] w-[15px] shrink-0 text-violet" aria-hidden />
        </span>
        <span className="min-w-0 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          Datasets
        </span>
        <span className="shrink-0 rounded-[7px] border border-line bg-white/[0.04] px-2 py-1 font-mono text-xs text-ink-dim">
          {datasetCount}
        </span>
      </button>
    </section>
  );
}
