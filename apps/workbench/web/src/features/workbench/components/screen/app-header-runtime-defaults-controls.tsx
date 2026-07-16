import { Lock, RotateCcw, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StatusPill } from "@/features/workbench/components/status-pill";
import { useModelPackageInspection } from "@/features/workbench/providers/workbench-providers";
import { type WorkbenchWorkspace } from "@/types/workbench";

export function AppHeaderRuntimeDefaultsControls({
  activeWorkspace,
  actionButtonClassName,
}: {
  activeWorkspace: WorkbenchWorkspace;
  actionButtonClassName: string;
}) {
  const { browser, runtimeDefaults, actions } = useModelPackageInspection();
  const overrideCount = runtimeDefaults.overrideCount;
  const presetOwnedFieldCount = runtimeDefaults.presetOwnedFieldCount;
  const canResetOverrides =
    activeWorkspace === "model" && Boolean(browser.selectedModel);

  return (
    <>
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
      <Button
        variant="ghost"
        onClick={actions.resetRuntimeDefaults}
        disabled={!canResetOverrides}
        className={`${actionButtonClassName} hidden 2xl:inline-flex`}
      >
        <RotateCcw className="h-[15px] w-[15px]" aria-hidden />
        Reset Overrides
      </Button>
    </>
  );
}
