import dynamic from "next/dynamic";
import { ListChecks, Plug, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { StatusPill } from "@/features/workbench/components/status-pill";
import { WorkbenchWorkspaceNav } from "@/features/workbench/components/workbench-workspace-nav";
import { useWorkbenchConnection } from "@/features/workbench/providers/workbench-connection-provider";
import { useActiveTrainingJob } from "@/features/workbench/providers/training-provider";
import { type WorkbenchWorkspace } from "@/types/workbench";

const headerActionButtonClassName =
  "min-w-touch gap-1.5 px-2.5 text-ink-dim hover:bg-control-hover disabled:hover:border-transparent disabled:hover:bg-transparent disabled:hover:text-ink-faint md:min-w-0 md:px-3";

function RuntimeDefaultsControlsLoading() {
  return (
    <span
      className="hidden h-control w-[18rem] shrink-0 2xl:inline-block"
      aria-hidden
    />
  );
}

const AppHeaderRuntimeDefaultsControls = dynamic(
  () =>
    import(
      "@/features/workbench/components/screen/app-header-runtime-defaults-controls"
    ).then((module) => module.AppHeaderRuntimeDefaultsControls),
  { ssr: false, loading: RuntimeDefaultsControlsLoading },
);

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
  workspaceHref,
  onOpenFeatureList,
  onOpenApiConnection,
  onOpenImportLogs,
}: {
  activeWorkspace: WorkbenchWorkspace;
  onChangeWorkspace: (workspace: WorkbenchWorkspace) => void;
  workspaceHref: (workspace: WorkbenchWorkspace) => string;
  onOpenFeatureList: () => void;
  onOpenApiConnection: () => void;
  onOpenImportLogs: () => void;
}) {
  const { connection } = useWorkbenchConnection();
  const apiOnline = connection.isOnline;
  const { activeTrainingJob } = useActiveTrainingJob();
  const trainingStatus = activeTrainingJob
    ? {
        label: activeTrainingJob.status,
        tone: trainingStatusTone(activeTrainingJob.status),
      }
    : undefined;
  return (
    <header className="safe-header-inset flex h-full min-h-0 items-center justify-between gap-2 border-b border-line bg-panel/80 pt-[env(safe-area-inset-top)] shadow-divider backdrop-blur-xl lg:gap-region">
      <div className="flex min-w-0 items-center gap-region">
        <div className="flex shrink-0 items-center gap-2.5">
          <span className="sr-only">Emperor Workbench</span>
          <span
            className="grid h-control-sm w-control-sm place-items-center rounded-control-md border border-accent-line bg-accent-soft font-mono type-caption font-bold text-violet shadow-control-accent md:h-control md:w-control"
            aria-hidden
          >
            E/
          </span>
          <span className="hidden leading-none 2xl:grid" aria-hidden>
            <span className="type-meta font-extrabold uppercase tracking-wordmark text-ink">
              Emperor
            </span>
            <span className="mt-1 font-mono type-micro uppercase tracking-wordmark text-ink-faint">
              Workbench
            </span>
          </span>
          <span className="ml-1 hidden h-6 w-px bg-line 2xl:block" aria-hidden />
        </div>
        <WorkbenchWorkspaceNav
          activeWorkspace={activeWorkspace}
          onChange={onChangeWorkspace}
          hrefForWorkspace={workspaceHref}
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
        <Button
          variant="ghost"
          aria-label="API connection settings"
          aria-describedby="app-header-api-status"
          onClick={onOpenApiConnection}
          className={`${headerActionButtonClassName} relative`}
        >
          <Plug className="h-[15px] w-[15px]" aria-hidden />
          <StatusDot
            online={apiOnline}
            className="absolute right-1.5 top-1.5 sm:hidden"
          />
          <span id="app-header-api-status" className="sr-only">
            API status: {apiOnline ? "online" : "offline"}
          </span>
          <span className="hidden lg:inline">Connection</span>
        </Button>
        <Button
          variant="ghost"
          aria-label="Import logs"
          onClick={onOpenImportLogs}
          className={headerActionButtonClassName}
        >
          <Upload className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden lg:inline">Import Logs</span>
        </Button>
        <Button
          variant="ghost"
          aria-label="Features"
          onClick={onOpenFeatureList}
          className={headerActionButtonClassName}
        >
          <ListChecks className="h-[15px] w-[15px]" aria-hidden />
          <span className="hidden lg:inline">Features</span>
        </Button>
        <AppHeaderRuntimeDefaultsControls
          activeWorkspace={activeWorkspace}
          actionButtonClassName={headerActionButtonClassName}
        />
      </div>
    </header>
  );
}
