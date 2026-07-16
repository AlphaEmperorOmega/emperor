import { useEffect, useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type DialogShellProps = {
  titleId?: string;
  labelledBy?: string;
  ariaLabel?: string;
  describedBy?: string;
  size?: "sm" | "md" | "lg" | "fullscreen";
  panelVariant?: "edge" | "surface";
  header?: ReactNode;
  footer?: ReactNode;
  overlayChildren?: ReactNode;
  children: ReactNode;
  onClose?: () => void;
  closeOnEscape?: boolean;
  className?: string;
  panelClassName?: string;
};

const dialogOverlayClassName =
  "safe-dialog-inset fixed inset-0 z-50 flex items-center justify-center overscroll-contain bg-black/80 backdrop-blur-md";

const dialogPanelClassNames: Record<NonNullable<DialogShellProps["size"]>, string> = {
  sm: "flex min-w-0 max-h-[calc(100dvh-1.5rem)] w-full max-w-full flex-col overflow-hidden sm:max-h-[calc(100dvh-3rem)] sm:max-w-lg",
  md: "flex min-w-0 max-h-[calc(100dvh-1.5rem)] w-full max-w-full flex-col overflow-hidden sm:max-h-[calc(100dvh-3rem)] sm:max-w-3xl",
  lg: "flex min-w-0 max-h-[calc(100dvh-1.5rem)] w-full max-w-full flex-col overflow-hidden sm:max-h-[calc(100dvh-3rem)] sm:max-w-4xl",
  fullscreen:
    "flex min-w-0 max-h-[calc(100dvh-1.5rem)] w-full max-w-full flex-col overflow-hidden sm:max-h-[calc(100dvh-3rem)] sm:max-w-[92rem]",
};

const dialogPanelVariantClassNames: Record<
  NonNullable<DialogShellProps["panelVariant"]>,
  string
> = {
  edge:
    "edge rounded-dialog shadow-dialog",
  surface:
    "rounded-dialog border border-line-hover bg-panel shadow-dialog",
};

const focusableSelector = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "details > summary:first-of-type",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

type ActiveDialog = {
  dialog: HTMLElement;
  root: HTMLElement;
};

const activeDialogs: ActiveDialog[] = [];

function isTopDialog(dialog: HTMLElement) {
  return topDialog()?.dialog === dialog;
}

function topDialog() {
  return activeDialogs.reduce<ActiveDialog | null>((currentTop, candidate) => {
    if (!currentTop) {
      return candidate;
    }
    if (currentTop.root.contains(candidate.root)) {
      return candidate;
    }
    if (candidate.root.contains(currentTop.root)) {
      return currentTop;
    }
    return currentTop.root.compareDocumentPosition(candidate.root) &
      Node.DOCUMENT_POSITION_FOLLOWING
      ? candidate
      : currentTop;
  }, null);
}

function visibleAndFocusable(element: HTMLElement) {
  if (element.getAttribute("aria-hidden") === "true") {
    return false;
  }
  if (element.tabIndex < 0) {
    return false;
  }
  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden";
}

function focusableElements(dialog: HTMLElement) {
  return Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelector))
    .filter(visibleAndFocusable);
}

function initialFocusTarget(dialog: HTMLElement) {
  const prefersDesktopFocus =
    typeof window.matchMedia !== "function" ||
    window.matchMedia("(min-width: 768px)").matches;
  if (prefersDesktopFocus) {
    const requestedTarget = dialog.querySelector<HTMLElement>(
      "[autofocus], [data-autofocus='true']",
    );
    if (requestedTarget && visibleAndFocusable(requestedTarget)) {
      return requestedTarget;
    }
  }
  return focusableElements(dialog)[0] ?? dialog;
}

function focusInsideDialog(dialog: HTMLElement) {
  initialFocusTarget(dialog).focus({ preventScroll: true });
}

export function DialogShell({
  titleId,
  labelledBy,
  ariaLabel,
  describedBy,
  size = "lg",
  panelVariant = "edge",
  header,
  footer,
  overlayChildren,
  children,
  onClose,
  closeOnEscape = true,
  className,
  panelClassName,
}: DialogShellProps) {
  const dialogRef = useRef<HTMLElement | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  const closeOnEscapeRef = useRef(closeOnEscape);

  useEffect(() => {
    onCloseRef.current = onClose;
    closeOnEscapeRef.current = closeOnEscape;
  });

  useEffect(() => {
    const dialog = dialogRef.current;
    const root = rootRef.current;
    if (!dialog || !root) {
      return;
    }
    const dialogElement: HTMLElement = dialog;
    const rootElement: HTMLElement = root;

    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const activeDialog = { dialog: dialogElement, root: rootElement };
    activeDialogs.push(activeDialog);

    if (isTopDialog(dialogElement) && !dialogElement.contains(document.activeElement)) {
      focusInsideDialog(dialogElement);
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (!isTopDialog(dialogElement)) {
        return;
      }

      const eventDialog =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>("[role='dialog']")
          : null;
      if (eventDialog && eventDialog !== dialogElement) {
        return;
      }

      if (event.key === "Escape" && closeOnEscapeRef.current && onCloseRef.current) {
        event.preventDefault();
        event.stopPropagation();
        onCloseRef.current();
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const focusable = focusableElements(dialogElement);
      if (focusable.length === 0) {
        event.preventDefault();
        dialogElement.focus({ preventScroll: true });
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const current = document.activeElement;

      if (event.shiftKey && current === first) {
        event.preventDefault();
        last.focus({ preventScroll: true });
        return;
      }

      if (!event.shiftKey && current === last) {
        event.preventDefault();
        first.focus({ preventScroll: true });
      }
    }

    function handleFocusIn(event: FocusEvent) {
      if (
        isTopDialog(dialogElement) &&
        event.target instanceof Node &&
        !dialogElement.contains(event.target)
      ) {
        focusInsideDialog(dialogElement);
      }
    }

    document.addEventListener("keydown", handleKeyDown, true);
    document.addEventListener("focusin", handleFocusIn);

    return () => {
      document.removeEventListener("keydown", handleKeyDown, true);
      document.removeEventListener("focusin", handleFocusIn);
      const activeIndex = activeDialogs.indexOf(activeDialog);
      if (activeIndex >= 0) {
        activeDialogs.splice(activeIndex, 1);
      }
      if (restoreFocusRef.current?.isConnected) {
        restoreFocusRef.current.focus({ preventScroll: true });
      }
    };
  }, []);

  return (
    <div ref={rootRef} className={cn(dialogOverlayClassName, className)}>
      <section
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        aria-labelledby={labelledBy ?? titleId}
        aria-describedby={describedBy}
        tabIndex={-1}
        className={cn(
          dialogPanelClassNames[size],
          dialogPanelVariantClassNames[panelVariant],
          panelClassName,
          "dialog-shell-panel",
        )}
      >
        {header}
        {children}
        {footer}
      </section>
      {overlayChildren}
    </div>
  );
}
