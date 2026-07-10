import { type RefObject } from "react";
import { type FixedPopupPosition } from "@/features/workbench/components/screen/fixed-popup";
import { FixedPopupShell } from "@/features/workbench/components/screen/fixed-popup-shell";

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
  return (
    <FixedPopupShell
      id={id}
      popupRef={submenuRef}
      ariaLabel="Preset description"
      title={presetName}
      subtitle={
        <p className="mt-0.5 text-xs font-semibold uppercase tracking-[0.09em] text-ink-faint">
          Preset description
        </p>
      }
      position={position}
      closeLabel="Close preset description"
      onClose={onClose}
      bodyClassName="px-3 py-3"
    >
      <p className="break-words text-xs leading-5 text-ink-dim">{description}</p>
    </FixedPopupShell>
  );
}
