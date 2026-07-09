import { type CSSProperties, type RefObject } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { IconButton } from "@/components/ui/icon-button";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import { type FixedPopupPosition } from "@/features/workbench/components/screen/fixed-popup";
import { type Dataset } from "@/lib/api";
import { cn } from "@/lib/utils";

export function DatasetSubmenu({
  id,
  submenuRef,
  datasets,
  selectedDatasets,
  position,
  onToggleDataset,
  onSelectAllDatasets,
  onSelectFirstDataset,
  onClose,
}: {
  id: string;
  submenuRef: RefObject<HTMLDivElement | null>;
  datasets: Dataset[];
  selectedDatasets: string[];
  position: FixedPopupPosition;
  onToggleDataset: (dataset: string) => void;
  onSelectAllDatasets: () => void;
  onSelectFirstDataset: () => void;
  onClose: () => void;
}) {
  const style = {
    top: position.top,
    left: position.left,
    width: position.width,
    maxHeight: position.maxHeight,
  } satisfies CSSProperties;

  return (
    <div
      id={id}
      ref={submenuRef}
      role="dialog"
      aria-label="Dataset selector"
      className="fixed z-[70] grid grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden rounded-[14px] border border-line bg-[linear-gradient(180deg,rgba(22,22,34,0.98),rgba(11,11,19,0.98))] shadow-[0_20px_60px_-24px_rgba(0,0,0,0.95),0_0_0_1px_rgba(255,255,255,0.03)] backdrop-blur"
      style={style}
    >
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-line-soft px-3 py-2.5">
        <div className="min-w-0">
          <h2 className="truncate text-[13px] font-bold text-ink">Dataset selector</h2>
          <p className="mt-0.5 font-mono text-xs text-ink-faint">
            {selectedDatasets.length} / {datasets.length}
          </p>
        </div>
        <IconButton
          label="Close dataset selector"
          onClick={onClose}
          size="sm"
          variant="edge"
          className="rounded-[9px] border-line bg-white/[0.025] text-ink-dim hover:border-white/15 hover:bg-white/[0.06] hover:text-ink"
          icon={<X className="h-4 w-4" aria-hidden />}
        />
      </div>
      <div className="grid min-h-0 gap-2 overflow-y-auto p-3">
        {datasets.map((dataset) => {
          const checked = selectedDatasets.includes(dataset.name);
          return (
            <label
              key={dataset.name}
              className={cn(
                "grid cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-[11px] rounded-[12px] border px-3 py-[11px] transition",
                checked
                  ? "border-violet/40 bg-[linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))]"
                  : "border-line-soft bg-white/[0.012] hover:border-line hover:bg-white/[0.03]",
              )}
            >
              <Checkbox
                checked={checked}
                onCheckedChange={() => onToggleDataset(dataset.name)}
                aria-label={`dataset ${dataset.name}`}
              />
              <span className="min-w-0">
                <span className="block truncate text-[13.5px] font-semibold text-ink">
                  {dataset.label}
                </span>
                <span className="mt-0.5 block truncate font-mono text-xs text-ink-dim">
                  {dataset.name}
                </span>
              </span>
              <StatChip className="shrink-0">
                {dataset.inputDim} {"->"} {dataset.outputDim}
              </StatChip>
            </label>
          );
        })}
      </div>
      <div className="grid grid-cols-2 gap-2 border-t border-line-soft p-3">
        <Button
          variant="secondary"
          onClick={onSelectAllDatasets}
          disabled={datasets.length === 0}
          className="h-9 text-[13px]"
        >
          All
        </Button>
        <Button
          variant="ghost"
          onClick={onSelectFirstDataset}
          disabled={datasets.length === 0}
          className="h-9 border border-line bg-white/[0.025] text-[13px]"
        >
          First
        </Button>
      </div>
    </div>
  );
}
