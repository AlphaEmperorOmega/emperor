import { Maximize2, SlidersHorizontal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { SectionHeading } from "@/components/ui/section-heading";
import { workbenchStatusCopy } from "@/features/workbench/components/shared/status-copy";
import {
  useConfigSnapshotRecords,
  useModelPackageInspection,
} from "@/features/workbench/providers/workbench-providers";

export function ConfigSummaryPanel({
  onOpenFullConfig,
}: {
  onOpenFullConfig: () => void;
}) {
  const { records } = useConfigSnapshotRecords();
  const { target, browser, runtimeDefaults, status } =
    useModelPackageInspection();
  const fieldCount = runtimeDefaults.fieldCount;
  const overrideCount = runtimeDefaults.overrideCount;
  const configSnapshotCount = records.allCount;
  const canOpenFullConfig = Boolean(
    browser.selectedModel && browser.selectedPreset && status.schema.isReady,
  );
  const showFullConfigButton = target.kind !== "historical-run";
  const isSchemaLoading = status.schema.isLoading;
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

      {showFullConfigButton && (
        <Button
          variant="primary"
          onClick={onOpenFullConfig}
          disabled={!canOpenFullConfig}
          className="h-11 w-full type-compact"
        >
          <Maximize2 className="h-4 w-4" aria-hidden />
          Open Full Config
        </Button>
      )}

      {isSchemaLoading ? (
        <InlineStatus busy compact>
          {workbenchStatusCopy.loading.configSchema}
        </InlineStatus>
      ) : fieldCount === 0 && (
        <InlineStatus compact>
          {workbenchStatusCopy.empty.configFields}
        </InlineStatus>
      )}
    </section>
  );
}
