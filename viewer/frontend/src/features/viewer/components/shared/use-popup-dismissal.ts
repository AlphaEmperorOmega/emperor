import { type RefObject, useEffect } from "react";

export function usePopupDismissal<
  TTrigger extends HTMLElement,
  TPanel extends HTMLElement,
>({
  open,
  onClose,
  triggerRef,
  panelRef,
  enabled = true,
}: {
  open: boolean;
  onClose: () => void;
  triggerRef: RefObject<TTrigger | null>;
  panelRef: RefObject<TPanel | null>;
  enabled?: boolean;
}) {
  useEffect(() => {
    if (!enabled || !open) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (
        target &&
        (triggerRef.current?.contains(target) || panelRef.current?.contains(target))
      ) {
        return;
      }
      onClose();
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [enabled, onClose, open, panelRef, triggerRef]);
}
