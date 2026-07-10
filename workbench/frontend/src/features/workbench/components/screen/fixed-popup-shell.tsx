import {
  type ReactNode,
  type RefObject,
} from "react";
import { X } from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import { type FixedPopupPosition } from "@/features/workbench/components/screen/fixed-popup";
import { cn } from "@/lib/utils";

export function FixedPopupShell({
  id,
  popupRef,
  ariaLabel,
  title,
  subtitle,
  position,
  closeLabel,
  onClose,
  children,
  bodyClassName,
  footer,
}: {
  id: string;
  popupRef: RefObject<HTMLDivElement | null>;
  ariaLabel: string;
  title: ReactNode;
  subtitle: ReactNode;
  position: FixedPopupPosition;
  closeLabel: string;
  onClose: () => void;
  children: ReactNode;
  bodyClassName?: string;
  footer?: ReactNode;
}) {
  return (
    <div
      id={id}
      ref={popupRef}
      role="dialog"
      aria-label={ariaLabel}
      className={cn(
        "fixed z-[70] grid overflow-hidden rounded-[14px] border border-line bg-[linear-gradient(180deg,rgba(22,22,34,0.98),rgba(11,11,19,0.98))] shadow-[0_20px_60px_-24px_rgba(0,0,0,0.95),0_0_0_1px_rgba(255,255,255,0.03)] backdrop-blur",
        footer
          ? "grid-rows-[auto_minmax(0,1fr)_auto]"
          : "grid-rows-[auto_minmax(0,1fr)]",
      )}
      style={{
        top: position.top,
        left: position.left,
        width: position.width,
        maxHeight: position.maxHeight,
      }}
    >
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-line-soft px-3 py-2.5">
        <div className="min-w-0">
          <h2 className="truncate text-[13px] font-bold text-ink">{title}</h2>
          {subtitle}
        </div>
        <IconButton
          label={closeLabel}
          onClick={onClose}
          size="sm"
          variant="edge"
          className="rounded-[9px] border-line bg-white/[0.025] text-ink-dim hover:border-white/15 hover:bg-white/[0.06] hover:text-ink"
          icon={<X className="h-4 w-4" aria-hidden />}
        />
      </div>
      <div className={cn("min-h-0 overflow-y-auto", bodyClassName)}>
        {children}
      </div>
      {footer && (
        <div className="grid grid-cols-2 gap-2 border-t border-line-soft p-3">
          {footer}
        </div>
      )}
    </div>
  );
}
