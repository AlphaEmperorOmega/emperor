import { useEffect, useId, useState } from "react";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";

export type TrainingShell = "posix" | "powershell";

const STORAGE_KEY = "emperor.workbench.trainingShell";
const CHANGE_EVENT = "emperor:training-shell-change";

function suggestedShell(): TrainingShell {
  if (typeof navigator === "undefined") {
    return "posix";
  }
  const navigatorWithUserAgentData = navigator as Navigator & {
    userAgentData?: { platform?: string };
  };
  const platform = [
    navigatorWithUserAgentData.userAgentData?.platform,
    navigator.platform,
    navigator.userAgent,
  ]
    .filter(Boolean)
    .join(" ");
  return /Windows|Win32|Win64/i.test(platform) ? "powershell" : "posix";
}

function storedShell(): TrainingShell | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value === "posix" || value === "powershell" ? value : null;
  } catch {
    return null;
  }
}

export function useTrainingShell() {
  const [shell, setShellState] = useState<TrainingShell>("posix");

  useEffect(() => {
    setShellState(storedShell() ?? suggestedShell());
    const sync = () => setShellState(storedShell() ?? suggestedShell());
    window.addEventListener(CHANGE_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  function setShell(nextShell: TrainingShell) {
    setShellState(nextShell);
    try {
      window.localStorage.setItem(STORAGE_KEY, nextShell);
      window.dispatchEvent(new Event(CHANGE_EVENT));
    } catch {
      // The selector still works for this dialog if storage is unavailable.
    }
  }

  return { shell, setShell };
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
