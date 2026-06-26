import { type ViewerWorkspace } from "@/types/viewer";

const workspaceItems: Array<{
  value: ViewerWorkspace;
  label: string;
}> = [
  { value: "model", label: "Model" },
  { value: "logs", label: "Logs" },
  { value: "compare", label: "Compare" },
];

export function ViewerWorkspaceNav({
  activeWorkspace,
  onChange,
}: {
  activeWorkspace: ViewerWorkspace;
  onChange: (workspace: ViewerWorkspace) => void;
}) {
  return (
    <nav className="grid grid-cols-3 gap-2 rounded-control-group border border-line-soft bg-control-muted p-1">
      {workspaceItems.map((item) => {
        const active = activeWorkspace === item.value;
        return (
          <button
            key={item.value}
            type="button"
            className={
              active
                ? "inline-flex h-9 items-center justify-center gap-2 rounded-control bg-grad px-3 text-sm font-bold text-white shadow-primary"
                : "inline-flex h-9 items-center justify-center gap-2 rounded-control px-3 text-sm font-semibold text-ink-dim transition hover:bg-control-active hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            }
            aria-pressed={active}
            onClick={() => onChange(item.value)}
          >
            {item.label}
          </button>
        );
      })}
    </nav>
  );
}
