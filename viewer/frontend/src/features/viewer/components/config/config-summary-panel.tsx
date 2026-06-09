import { Maximize2, SlidersHorizontal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import {
  useTargetConfigSummaryState,
} from "@/features/viewer/providers/viewer-providers";

export function ConfigSummaryPanel({
  onOpenFullConfig,
}: {
  onOpenFullConfig: () => void;
}) {
  const {
    fieldCount,
    overrideCount,
    configSnapshotCount,
    canOpenFullConfig,
    isSchemaLoading,
  } = useTargetConfigSummaryState();
  return (
    <section className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <SectionHeading
          as="h2"
          className="min-w-0"
          icon={<SlidersHorizontal className="h-[15px] w-[15px] shrink-0 text-violet" aria-hidden />}
          title="Config"
        />
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-1.5">
          <span className="text-xs font-medium text-ink-dim">
            {overrideCount} overrides
          </span>
          {configSnapshotCount > 0 && (
            <Badge variant="success">
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

      {isSchemaLoading ? (
        <InlineStatus compact>
          {viewerStatusCopy.loading.configSchema}
        </InlineStatus>
      ) : fieldCount === 0 && (
        <InlineStatus compact>
          {viewerStatusCopy.empty.configFields}
        </InlineStatus>
      )}
    </section>
  );
}
