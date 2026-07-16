import { useId, useSyncExternalStore } from "react";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";
import {
  getTrainingShellServerSnapshot,
  getTrainingShellSnapshot,
  setTrainingShell,
  subscribeTrainingShell,
  type TrainingShell,
} from "@/features/workbench/components/training/training-shell-store";

export type { TrainingShell } from "@/features/workbench/components/training/training-shell-store";

export function useTrainingShell() {
  const shell = useSyncExternalStore(
    subscribeTrainingShell,
    getTrainingShellSnapshot,
    getTrainingShellServerSnapshot,
  );
  return { shell, setShell: setTrainingShell };
}

export function TrainingShellSelector({
  shell,
  onChange,
}: {
  shell: TrainingShell;
  onChange: (shell: TrainingShell) => void;
}) {
  const titleId = useId();

  return (
    <SurfacePanel
      as="section"
      padding="compact"
      aria-labelledby={titleId}
      className="border-line-soft bg-control-subtle"
      title={
        <div className="min-w-0">
          <h3
            id={titleId}
            className="type-label font-bold uppercase tracking-label text-ink-dim"
          >
            Command Shell
          </h3>
          <p className="mt-0.5 text-pretty type-meta font-normal normal-case tracking-normal text-ink-faint">
            Choose the terminal that will run this command.
          </p>
        </div>
      }
      actions={
        <SegmentedControl aria-label="Training command shell">
          <ViewModeButton
            active={shell === "posix"}
            onClick={() => onChange("posix")}
          >
            <span translate="no">POSIX</span>
          </ViewModeButton>
          <ViewModeButton
            active={shell === "powershell"}
            onClick={() => onChange("powershell")}
          >
            <span translate="no">PowerShell</span>
          </ViewModeButton>
        </SegmentedControl>
      }
    />
  );
}

export function commandForShell(
  run: { command: string; commands?: { posix: string; powershell: string } },
  shell: TrainingShell,
) {
  return (
    run.commands?.[shell]?.trim() ||
    (typeof run.command === "string" ? run.command : "")
  );
}
