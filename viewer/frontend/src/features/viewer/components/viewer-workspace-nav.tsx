import { Activity, FileText, GitCompare, Layers } from "lucide-react";
import { type ViewerWorkspace } from "@/types/viewer";
import { cn } from "@/lib/utils";

export type ViewerWorkspaceTrainingStatus = {
  label: string;
  tone: "neutral" | "good" | "warn" | "danger";
};

const workspaceItems: Array<{
  value: ViewerWorkspace;
  label: string;
  icon: typeof Layers;
}> = [
  { value: "model", label: "Model", icon: Layers },
  { value: "training", label: "Training", icon: Activity },
  { value: "logs", label: "Logs", icon: FileText },
  { value: "compare", label: "Compare", icon: GitCompare },
];

export function ViewerWorkspaceNav({
  activeWorkspace,
  onChange,
  trainingStatus,
}: {
  activeWorkspace: ViewerWorkspace;
  onChange: (workspace: ViewerWorkspace) => void;
  trainingStatus?: ViewerWorkspaceTrainingStatus;
}) {
  return (
    <nav aria-label="Workspace" className="min-w-0">
      <ul className="m-0 flex h-9 min-w-0 list-none items-center gap-1 p-0">
        {workspaceItems.map((item) => {
          const active = activeWorkspace === item.value;
          const Icon = item.icon;
          return (
            <li key={item.value} className="min-w-0">
              <button
                type="button"
                className={cn(
                  "relative inline-flex h-9 items-center justify-center gap-1.5 whitespace-nowrap rounded-control-sm px-2.5 text-sm font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus sm:px-3",
                  active
                    ? "bg-control-active text-ink"
                    : "text-ink-dim hover:bg-control-hover hover:text-ink",
                )}
                aria-current={active ? "page" : undefined}
                onClick={() => onChange(item.value)}
              >
                <Icon
                  className={cn(
                    "h-[15px] w-[15px] shrink-0",
                    active ? "text-accent" : "text-ink-faint",
                  )}
                  aria-hidden
                />
                <span>{item.label}</span>
                {item.value === "training" && trainingStatus && (
                  <span
                    className={cn(
                      "ml-0.5 hidden max-w-[6rem] truncate rounded-[6px] border px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase leading-none sm:inline",
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
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
