import { type CSSProperties, type RefObject } from "react";
import { X } from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import { type FixedPopupPosition } from "@/features/viewer/components/screen/fixed-popup";

export function PresetDescriptionSubmenu({
  id,
  submenuRef,
  presetName,
  description,
  position,
  onClose,
}: {
  id: string;
  submenuRef: RefObject<HTMLDivElement | null>;
  presetName: string;
  description: string;
  position: FixedPopupPosition;
  onClose: () => void;
}) {
  const style = {
    top: position.top,
    left: position.left,
    width: position.width,
    maxHeight: position.maxHeight,
  } satisfies CSSProperties;

  return (
    <div
      id={id}
      ref={submenuRef}
      role="dialog"
      aria-label="Preset description"
      className="fixed z-[70] grid grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-[14px] border border-line bg-[linear-gradient(180deg,rgba(22,22,34,0.98),rgba(11,11,19,0.98))] shadow-[0_20px_60px_-24px_rgba(0,0,0,0.95),0_0_0_1px_rgba(255,255,255,0.03)] backdrop-blur"
      style={style}
    >
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-line-soft px-3 py-2.5">
        <div className="min-w-0">
          <h2 className="truncate text-[13px] font-bold text-ink">{presetName}</h2>
          <p className="mt-0.5 text-xs font-semibold uppercase tracking-[0.09em] text-ink-faint">
            Preset description
          </p>
        </div>
        <IconButton
          label="Close preset description"
          onClick={onClose}
          size="sm"
          variant="edge"
          className="rounded-[9px] border-line bg-white/[0.025] text-ink-dim hover:border-white/15 hover:bg-white/[0.06] hover:text-ink"
          icon={<X className="h-4 w-4" aria-hidden />}
        />
      </div>
      <div className="min-h-0 overflow-y-auto px-3 py-3">
        <p className="break-words text-xs leading-5 text-ink-dim">{description}</p>
      </div>
    </div>
  );
}
