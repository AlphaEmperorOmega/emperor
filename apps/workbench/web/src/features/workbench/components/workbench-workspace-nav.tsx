import { Activity, FileText, Layers } from "lucide-react";
import { type MouseEvent } from "react";
import { type WorkbenchWorkspace } from "@/types/workbench";
import { cn } from "@/lib/utils";

export type WorkbenchWorkspaceTrainingStatus = {
  label: string;
  tone: "neutral" | "good" | "warn" | "danger";
};

const workspaceItems: Array<{
  value: WorkbenchWorkspace;
  label: string;
  icon: typeof Layers;
}> = [
  { value: "model", label: "Model", icon: Layers },
  { value: "training", label: "Training", icon: Activity },
  { value: "logs", label: "Logs", icon: FileText },
];

export function WorkbenchWorkspaceNav({
  activeWorkspace,
  hrefForWorkspace,
  onChange,
  trainingStatus,
}: {
  activeWorkspace: WorkbenchWorkspace;
  hrefForWorkspace: (workspace: WorkbenchWorkspace) => string;
  onChange: (workspace: WorkbenchWorkspace) => void;
  trainingStatus?: WorkbenchWorkspaceTrainingStatus;
}) {
  return (
    <nav aria-label="Workspace" className="min-w-0 overflow-hidden">
      <ul className="m-0 flex h-touch min-w-0 list-none items-center gap-0.5 p-0 md:h-control md:gap-1">
        {workspaceItems.map((item) => {
          const active = activeWorkspace === item.value;
          const Icon = item.icon;
          return (
            <li key={item.value} className="min-w-0">
              <a
                href={hrefForWorkspace(item.value)}
                className={cn(
                  "relative inline-flex h-touch items-center justify-center gap-1.5 whitespace-nowrap rounded-control-sm px-2 type-body font-semibold transition-[color,background-color,border-color,box-shadow] duration-150 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 focus-visible:ring-offset-bg sm:px-2.5 md:h-control lg:px-3",
                  active
                    ? "bg-accent-soft text-ink shadow-control-selected"
                    : "text-ink-dim hover:bg-control-hover hover:text-ink",
                )}
                aria-current={active ? "page" : undefined}
                onClick={(event: MouseEvent<HTMLAnchorElement>) => {
                  if (
                    event.defaultPrevented ||
                    event.button !== 0 ||
                    event.metaKey ||
                    event.ctrlKey ||
                    event.shiftKey ||
                    event.altKey
                  ) {
                    return;
                  }
                  event.preventDefault();
                  onChange(item.value);
                }}
              >
                <Icon
                  className={cn(
                    "hidden h-4 w-4 shrink-0 sm:block",
                    active ? "text-accent" : "text-ink-faint",
                  )}
                  aria-hidden
                />
                <span>{item.label}</span>
                {item.value === "training" && trainingStatus && (
                  <span
                    className={cn(
                      "ml-0.5 hidden max-w-[6rem] truncate rounded-chip border px-1.5 py-0.5 font-mono type-caption font-bold uppercase leading-none xl:inline",
                      trainingStatus.tone === "good" &&
                        "border-ok/30 bg-ok/10 text-ok",
                      trainingStatus.tone === "warn" &&
                        "border-amber/40 bg-amber/[0.12] text-amber",
                      trainingStatus.tone === "danger" &&
                        "border-danger-line bg-danger-soft text-danger-text",
                      trainingStatus.tone === "neutral" &&
                        "border-line bg-white/[0.045] text-ink-dim",
                    )}
                  >
                    {trainingStatus.label}
                  </span>
                )}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
