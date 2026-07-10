import { type RefObject } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import { FixedPopupShell } from "@/features/workbench/components/screen/fixed-popup-shell";
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
  return (
    <FixedPopupShell
      id={id}
      popupRef={submenuRef}
      ariaLabel="Dataset selector"
      title="Dataset selector"
      subtitle={
        <p className="mt-0.5 font-mono text-xs text-ink-faint">
          {selectedDatasets.length} / {datasets.length}
        </p>
      }
      position={position}
      closeLabel="Close dataset selector"
      onClose={onClose}
      bodyClassName="grid gap-2 p-3"
      footer={
        <>
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
        </>
      }
    >
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
    </FixedPopupShell>
  );
}
