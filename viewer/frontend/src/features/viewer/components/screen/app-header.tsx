import {
  Layers,
  ListChecks,
  Lock,
  Plug,
  RotateCcw,
  SlidersHorizontal,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { StatusPill } from "@/features/viewer/components/status-pill";
import {
  useTargetHeaderState,
} from "@/features/viewer/providers/viewer-providers";
import { type ViewerWorkspace } from "@/types/viewer";

export function AppHeader({
  activeWorkspace,
  onOpenFeatureList,
  onOpenApiConnection,
  onOpenImportLogs,
}: {
  activeWorkspace: ViewerWorkspace;
  onOpenFeatureList: () => void;
  onOpenApiConnection: () => void;
  onOpenImportLogs: () => void;
}) {
  const {
    selectedModel,
    selectedPreset,
    apiOnline,
    overrideCount,
    presetOwnedFieldCount,
    resetOverrides: onResetOverrides,
  } = useTargetHeaderState();
  const canResetOverrides = activeWorkspace === "model" && Boolean(selectedModel);
  return (
    <header className="flex h-[60px] min-h-0 items-center justify-between gap-3 border-b border-line bg-[linear-gradient(180deg,rgba(16,14,28,0.7),rgba(8,8,14,0.5))] px-[22px] backdrop-blur-xl">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-[11px] bg-grad text-white shadow-primary">
          <Layers className="h-[19px] w-[19px]" aria-hidden />
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-base font-bold text-ink">Emperor Model Viewer</h1>
          <div className="mt-0.5 truncate font-mono text-xs text-ink-dim">
            {selectedModel || "No model"} {selectedPreset ? `/ ${selectedPreset}` : ""}
          </div>
        </div>
      </div>
      <div className="flex shrink-0 items-center justify-end gap-2">
        <StatusPill
          className="hidden sm:inline-flex"
          icon={<StatusDot online={apiOnline} />}
          label="API"
          value={apiOnline ? "online" : "offline"}
          tone={apiOnline ? "good" : "danger"}
        />
        <StatusPill
          className="hidden xl:inline-flex"
          icon={<SlidersHorizontal className="h-4 w-4" />}
          label="overrides"
          value={overrideCount}
          tone={overrideCount > 0 ? "warn" : "neutral"}
        />
        <StatusPill
          className="hidden xl:inline-flex"
          icon={<Lock className="h-4 w-4" />}
          label="presets"
          value={presetOwnedFieldCount}
          tone={presetOwnedFieldCount > 0 ? "warn" : "neutral"}
        />
        <div className="mx-1 hidden h-6 w-px bg-line xl:block" />
        <Button
          variant="secondary"
          aria-label="API connection settings"
          onClick={onOpenApiConnection}
          className="h-9 px-3 sm:h-9 sm:px-4"
        >
          <Plug className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden sm:inline">Connection</span>
        </Button>
        <Button
          variant="secondary"
          aria-label="Import logs"
          onClick={onOpenImportLogs}
          className="h-9 px-3 sm:h-9 sm:px-4"
        >
          <Upload className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden sm:inline">Import Logs</span>
        </Button>
        <Button
          variant="secondary"
          aria-label="Features"
          onClick={onOpenFeatureList}
          className="h-9 px-3 sm:h-9 sm:px-4"
        >
          <ListChecks className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden sm:inline">Features</span>
        </Button>
        <Button
          variant="secondary"
          onClick={onResetOverrides}
          disabled={!canResetOverrides}
          className="hidden xl:inline-flex"
        >
          <RotateCcw className="h-[15px] w-[15px]" aria-hidden />
          Reset Overrides
        </Button>
      </div>
    </header>
  );
}
