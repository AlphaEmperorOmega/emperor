import {
  type RefObject,
  useCallback,
  useEffect,
  useState,
} from "react";
import { usePopupDismissal as usePointerPopupDismissal } from "@/components/features/viewer/shared/use-popup-dismissal";

export type FixedPopupPosition = {
  top: number;
  left: number;
  width: number;
  maxHeight: number;
};

type FixedPopupPositionOptions = {
  preferredWidth?: number;
  minWidth?: number;
  minimumHeight?: number;
};

function getFixedPopupPosition(
  trigger: HTMLElement,
  {
    preferredWidth = 320,
    minWidth = 260,
    minimumHeight = 220,
  }: FixedPopupPositionOptions = {},
): FixedPopupPosition {
  const rect = trigger.getBoundingClientRect();
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const viewportMargin = 12;
  const availableWidth = Math.max(160, viewportWidth - viewportMargin * 2);
  const opensRight =
    viewportWidth >= 1024 &&
    rect.right + viewportMargin + preferredWidth <= viewportWidth - viewportMargin;
  const width = opensRight
    ? preferredWidth
    : Math.min(Math.max(rect.width, minWidth), availableWidth);
  const left = opensRight
    ? rect.right + viewportMargin
    : Math.min(
        Math.max(rect.left, viewportMargin),
        viewportWidth - width - viewportMargin,
      );
  const desiredTop = opensRight ? rect.top : rect.bottom + 8;
  const top = Math.max(
    viewportMargin,
    Math.min(desiredTop, viewportHeight - viewportMargin - minimumHeight),
  );

  return {
    top,
    left,
    width,
    maxHeight: Math.max(180, viewportHeight - top - viewportMargin),
  };
}

export function useFixedPopupPosition<TTrigger extends HTMLElement>(
  triggerRef: RefObject<TTrigger | null>,
  isOpen: boolean,
  {
    preferredWidth = 320,
    minWidth = 260,
    minimumHeight = 220,
  }: FixedPopupPositionOptions = {},
) {
  const [position, setPosition] = useState<FixedPopupPosition | null>(null);

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger || typeof window === "undefined") {
      return;
    }

    setPosition(
      getFixedPopupPosition(trigger, {
        preferredWidth,
        minWidth,
        minimumHeight,
      }),
    );
  }, [minimumHeight, minWidth, preferredWidth, triggerRef]);

  useEffect(() => {
    if (!isOpen) {
      setPosition(null);
      return;
    }

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [isOpen, updatePosition]);

  return { position, updatePosition };
}

export function usePopupDismissal<
  TTrigger extends HTMLElement,
  TPopup extends HTMLElement,
>({
  isOpen,
  triggerRef,
  popupRef,
  onDismiss,
  onDismissWithFocus,
}: {
  isOpen: boolean;
  triggerRef: RefObject<TTrigger | null>;
  popupRef: RefObject<TPopup | null>;
  onDismiss: () => void;
  onDismissWithFocus: () => void;
}) {
  usePointerPopupDismissal({
    open: isOpen,
    onClose: onDismiss,
    triggerRef,
    panelRef: popupRef,
  });

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onDismissWithFocus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onDismissWithFocus]);
}
