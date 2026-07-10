import { Info, X } from "lucide-react";
import { createPortal } from "react-dom";
import { IconButton } from "@/components/ui/icon-button";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { type GraphNode } from "@/lib/api";
import { configDetailText } from "@/lib/graph";

export type ComponentInfoNode = Pick<
  GraphNode,
  "typeName" | "description" | "path" | "config"
>;

export function ComponentInfoDialog({
  node,
  onClose,
}: {
  node: ComponentInfoNode;
  onClose: () => void;
}) {
  if (typeof document === "undefined") {
    return null;
  }

  const titleId = "component-info-dialog-title";
  const description = node.description?.trim() || "No description available";
  const configFields = node.config?.fields ?? [];

  return createPortal(
    <DialogShell
      titleId={titleId}
      size="md"
      panelVariant="surface"
      onClose={onClose}
      panelClassName="bg-[linear-gradient(180deg,rgba(20,23,33,0.98),rgba(12,14,21,0.98))]"
      header={
        <div className="flex min-w-0 items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <h2
              id={titleId}
              className="flex min-w-0 items-center gap-2 text-base font-semibold text-ink"
            >
              <Info className="h-4 w-4 shrink-0 text-violet" aria-hidden />
              <span className="truncate">Component info</span>
            </h2>
            <div className="mt-2 truncate text-sm font-semibold text-ink">
              {node.typeName}
            </div>
            <div className="mt-1 break-words font-mono text-xs text-ink-faint">
              {node.path}
            </div>
          </div>
          <IconButton
            label="Close component info"
            title="Close"
            icon={<X className="h-4 w-4" aria-hidden />}
            variant="edge"
            onClick={onClose}
          />
        </div>
      }
    >
      <div className="grid min-h-0 gap-4 overflow-y-auto p-4">
        <SurfacePanel padding="roomy" className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-dim">
            Role
          </div>
          <p className="mt-2 text-sm leading-6 text-ink">{description}</p>
        </SurfacePanel>

        <SurfacePanel padding="none" className="min-w-0 overflow-hidden">
          <div className="border-b border-line-soft px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-dim">
              Config
            </div>
            <div className="mt-1 font-mono text-xs text-ink-faint">
              {node.config?.typeName ?? "No config"}
            </div>
          </div>
          {configFields.length > 0 ? (
            <div className="divide-y divide-line-soft">
              {configFields.map((field) => (
                <div
                  key={field.key}
                  className="px-4 py-3"
                  data-testid={`component-info-config-field-${field.key}`}
                >
                  <div className="whitespace-normal break-words font-mono text-xs font-semibold leading-5 text-ink">
                    <span>{field.key}</span>
                    <span className="text-ink-dim"> - </span>
                    <span className="text-violet-text">{configDetailText(field.value)}</span>
                  </div>
                  <p className="mt-2 text-xs leading-5 text-ink-dim">
                    {field.description?.trim() || "No field description available"}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <div className="px-4 py-5 text-sm text-ink-dim">
              No config fields available
            </div>
          )}
        </SurfacePanel>
      </div>
    </DialogShell>,
    document.body,
  );
}
