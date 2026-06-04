import { Maximize2, SlidersHorizontal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useTargetConfig } from "@/components/features/viewer/providers/viewer-providers";

export function ConfigSummaryPanel({
  onOpenFullConfig,
}: {
  onOpenFullConfig: () => void;
}) {
  const {
    fieldCount,
    overrideCount,
    configSnapshotCount,
    selectedModel,
    selectedPreset,
    schemaQuery,
  } = useTargetConfig();
  const isLoading = schemaQuery.isLoading;
  const canOpenFullConfig = Boolean(
    selectedModel && selectedPreset && schemaQuery.isSuccess,
  );
  return (
    <section className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <SlidersHorizontal className="h-[15px] w-[15px] shrink-0 text-violet" aria-hidden />
          <h2 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
            Config
          </h2>
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-1.5">
          <span className="text-xs font-medium text-ink-dim">
            {overrideCount} overrides
          </span>
          {configSnapshotCount > 0 && (
            <Badge className="border-ok/30 bg-ok/10 text-ok">
              {configSnapshotCount} snapshots
            </Badge>
          )}
        </div>
      </div>

      <Button
        variant="primary"
        onClick={onOpenFullConfig}
        disabled={!canOpenFullConfig}
        className="h-11 w-full text-[13.5px]"
      >
        <Maximize2 className="h-4 w-4" aria-hidden />
        Open Full Config
      </Button>

      {isLoading ? (
        <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-3 text-sm text-ink-faint">
          Loading config schema...
        </div>
      ) : fieldCount === 0 && (
        <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-3 text-sm text-ink-faint">
          No config fields
        </div>
      )}
    </section>
  );
}
