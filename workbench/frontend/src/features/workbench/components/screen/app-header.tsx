import {
  ListChecks,
  Lock,
  Plug,
  RotateCcw,
  SlidersHorizontal,
  Upload,
} from "lucide-react";
import {
  controlDisabledClassName,
  controlFocusClassName,
} from "@/components/ui/control-styles";
import { StatusDot } from "@/components/ui/status-dot";
import { StatusPill } from "@/features/workbench/components/status-pill";
import { WorkbenchWorkspaceNav } from "@/features/workbench/components/workbench-workspace-nav";
import {
  useModelPackageInspection,
} from "@/features/workbench/providers/workbench-providers";
import { useWorkbenchConnection } from "@/features/workbench/providers/workbench-connection-provider";
import { useActiveTrainingJob } from "@/features/workbench/providers/training-provider";
import { cn } from "@/lib/utils";
import { type WorkbenchWorkspace } from "@/types/workbench";

const headerActionButtonClassName = [
  "inline-flex h-touch min-w-touch items-center justify-center gap-1.5 rounded-control-sm px-2.5 md:h-control md:min-w-0",
  "border border-transparent bg-transparent type-body font-semibold text-ink-dim transition-[color,background-color,border-color,transform] duration-150 ease-out hover:border-line-soft hover:bg-control-hover hover:text-ink",
  "active:translate-y-px disabled:text-ink-faint disabled:hover:border-transparent disabled:hover:bg-transparent disabled:hover:text-ink-faint md:px-3",
  controlFocusClassName,
  controlDisabledClassName,
].join(" ");

function trainingStatusTone(status: string) {
  if (status === "completed") {
    return "good" as const;
  }
  if (status === "failed" || status === "cancelled") {
    return "danger" as const;
  }
  if (status === "queued" || status === "running") {
    return "warn" as const;
  }
  return "neutral" as const;
}

export function AppHeader({
  activeWorkspace,
  onChangeWorkspace,
  onOpenFeatureList,
  onOpenApiConnection,
  onOpenImportLogs,
}: {
  activeWorkspace: WorkbenchWorkspace;
  onChangeWorkspace: (workspace: WorkbenchWorkspace) => void;
  onOpenFeatureList: () => void;
  onOpenApiConnection: () => void;
  onOpenImportLogs: () => void;
}) {
  const { connection } = useWorkbenchConnection();
  const { browser, runtimeDefaults, actions } = useModelPackageInspection();
  const selectedModel = browser.selectedModel;
  const apiOnline = connection.isOnline;
  const overrideCount = runtimeDefaults.overrideCount;
  const presetOwnedFieldCount = runtimeDefaults.presetOwnedFieldCount;
  const onResetOverrides = actions.resetRuntimeDefaults;
  const { activeTrainingJob } = useActiveTrainingJob();
  const trainingStatus = activeTrainingJob
    ? {
        label: activeTrainingJob.status,
        tone: trainingStatusTone(activeTrainingJob.status),
      }
    : undefined;
  const canResetOverrides = activeWorkspace === "model" && Boolean(selectedModel);
  return (
    <header className="safe-header-inset flex h-full min-h-0 items-center justify-between gap-2 border-b border-line bg-panel/80 pt-[env(safe-area-inset-top)] shadow-divider backdrop-blur-xl lg:gap-region">
      <div className="flex min-w-0 items-center gap-region">
        <div className="hidden shrink-0 items-center gap-2.5 2xl:flex" aria-label="Emperor Workbench">
          <span
            className="grid h-control w-control place-items-center rounded-control-md border border-accent-line bg-accent-soft font-mono text-xs font-bold text-violet"
            aria-hidden
          >
            E/
          </span>
          <span className="grid leading-none">
            <span className="type-meta font-extrabold uppercase tracking-wordmark text-ink">
              Emperor
            </span>
            <span className="mt-1 font-mono type-micro uppercase tracking-wordmark text-ink-faint">
              Workbench
            </span>
          </span>
          <span className="ml-1 h-6 w-px bg-line" aria-hidden />
        </div>
        <WorkbenchWorkspaceNav
          activeWorkspace={activeWorkspace}
          onChange={onChangeWorkspace}
          trainingStatus={trainingStatus}
        />
      </div>
      <div className="flex shrink-0 items-center justify-end gap-0.5 sm:gap-1 lg:gap-2">
        <StatusPill
          className="hidden sm:inline-flex"
          icon={<StatusDot online={apiOnline} />}
          label="API"
          value={apiOnline ? "online" : "offline"}
          tone={apiOnline ? "good" : "danger"}
          live
        />
        <StatusPill
          className="hidden 2xl:inline-flex"
          icon={<SlidersHorizontal className="h-4 w-4" aria-hidden />}
          label="overrides"
          value={overrideCount}
          tone={overrideCount > 0 ? "warn" : "neutral"}
        />
        <StatusPill
          className="hidden 2xl:inline-flex"
          icon={<Lock className="h-4 w-4" aria-hidden />}
          label="presets"
          value={presetOwnedFieldCount}
          tone={presetOwnedFieldCount > 0 ? "warn" : "neutral"}
        />
        <div className="mx-1 hidden h-6 w-px bg-line 2xl:block" />
        <button
          type="button"
          aria-label="API connection settings"
          onClick={onOpenApiConnection}
          className={headerActionButtonClassName}
        >
          <Plug className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden lg:inline">Connection</span>
        </button>
        <button
          type="button"
          aria-label="Import logs"
          onClick={onOpenImportLogs}
          className={headerActionButtonClassName}
        >
          <Upload className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden lg:inline">Import Logs</span>
        </button>
        <button
          type="button"
          aria-label="Features"
          onClick={onOpenFeatureList}
          className={headerActionButtonClassName}
        >
          <ListChecks className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden lg:inline">Features</span>
        </button>
        <button
          type="button"
          onClick={onResetOverrides}
          disabled={!canResetOverrides}
          className={cn(headerActionButtonClassName, "hidden 2xl:inline-flex")}
        >
          <RotateCcw className="h-[15px] w-[15px]" aria-hidden />
          Reset Overrides
        </button>
      </div>
    </header>
  );
}
