import { ChevronDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { type ConfigField } from "@/lib/api";
import {
  type OverrideValues,
  modifiedCount,
  presetOwnedCount,
  sectionCountLabel,
} from "@/lib/config";
import { cn } from "@/lib/utils";
import { ConfigFieldControl } from "@/features/viewer/components/config/config-field-control";
import { ConfigMetricBadge } from "@/features/viewer/components/config/config-metric-badge";

export function ConfigSectionAccordion({
  id,
  refCallback,
  title,
  fields,
  overrides,
  isOpen,
  onToggle,
  onFieldChange,
  onFieldReset,
}: {
  id: string;
  refCallback: (element: HTMLElement | null) => void;
  title: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  isOpen: boolean;
  onToggle: () => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const sectionModifiedCount = modifiedCount(fields, overrides);
  const sectionPresetOwnedCount = presetOwnedCount(fields);
  const hasOverride = sectionModifiedCount > 0;
  const hasPreset = sectionPresetOwnedCount > 0;
  const hasBoth = hasOverride && hasPreset;
  const panelId = `${id}-fields`;
  const stateContainerClass = hasBoth
    ? "border-amber/35 bg-[linear-gradient(135deg,rgba(255,209,102,0.075),rgba(167,139,250,0.105))] shadow-[0_20px_46px_-32px_rgba(167,139,250,0.35)] ring-1 ring-violet/25 hover:border-violet/45"
    : hasOverride
      ? "border-violet/35 bg-violet/[0.06] shadow-[0_18px_42px_-32px_rgba(167,139,250,0.35)] hover:border-violet/45"
      : hasPreset
        ? "border-amber/35 bg-amber/[0.045] shadow-[0_18px_42px_-32px_rgba(255,209,102,0.25)] hover:border-amber/45"
        : "";
  const stateHeaderClass = hasBoth
    ? "bg-[linear-gradient(90deg,rgba(255,209,102,0.12),rgba(167,139,250,0.13))] hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.16),rgba(167,139,250,0.17))]"
    : hasOverride
      ? "bg-violet/[0.08] hover:bg-violet/[0.12]"
      : hasPreset
        ? "bg-amber/[0.08] hover:bg-amber/[0.12]"
        : "";

  return (
    <section
      ref={refCallback}
      className={cn(
        "overflow-hidden rounded-[12px] border shadow-[0_16px_40px_-30px_rgba(0,0,0,0.95)] transition duration-150 hover:-translate-y-px hover:border-line hover:shadow-[0_20px_44px_-32px_rgba(0,0,0,0.95)] focus-within:-translate-y-px focus-within:ring-2 focus-within:ring-focus motion-reduce:transform-none",
        isOpen
          ? "border-line bg-panel/80"
          : "border-line-soft bg-panel/70 shadow-[0_10px_28px_-26px_rgba(0,0,0,0.9)]",
        stateContainerClass,
      )}
    >
      <h3>
        <button
          type="button"
          aria-expanded={isOpen}
          aria-controls={panelId}
          aria-label={`${title} section, ${sectionCountLabel(
            fields.length,
            "field",
          )}, ${sectionCountLabel(sectionModifiedCount, "override")}${
            hasPreset
              ? `, ${sectionCountLabel(sectionPresetOwnedCount, "preset")}`
              : ""
          }`}
          onClick={onToggle}
          className={cn(
            "flex min-h-12 w-full min-w-0 items-center justify-between gap-3 overflow-hidden px-3 py-2.5 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
            isOpen
              ? "bg-white/[0.055] hover:bg-white/[0.075]"
              : "bg-white/[0.025] hover:bg-white/[0.055]",
            stateHeaderClass,
          )}
        >
          <span className="flex min-w-0 items-center gap-2 overflow-hidden">
            <ChevronDown
              className={cn(
                "h-4 w-4 shrink-0 text-ink-faint transition-transform",
                !isOpen && "-rotate-90",
              )}
              aria-hidden
            />
            <span className="min-w-0 truncate text-sm font-semibold text-ink">{title}</span>
          </span>
          <span className="flex shrink-0 items-center justify-end gap-1">
            <ConfigMetricBadge count={fields.length} kind="fields" focusable={false} />
            <ConfigMetricBadge
              count={sectionModifiedCount}
              kind="overrides"
              variant={sectionModifiedCount > 0 ? "override" : "default"}
              focusable={false}
            />
            {hasPreset && (
              <Badge variant="preset" className="h-[23px] items-center px-1.5 py-0">
                {sectionPresetOwnedCount} preset
              </Badge>
            )}
          </span>
        </button>
      </h3>
      <div
        id={panelId}
        hidden={!isOpen}
        className="border-t border-line-soft bg-black/[0.08] px-3 py-3"
      >
        <div className="grid gap-x-3 gap-y-3 md:grid-cols-2 2xl:grid-cols-3">
          {isOpen &&
            fields.map((field) => (
              <ConfigFieldControl
                key={field.key}
                field={field}
                overrides={overrides}
                onChange={onFieldChange}
                onReset={onFieldReset}
                density="compact"
                idPrefix={`${id}-field`}
              />
            ))}
        </div>
      </div>
    </section>
  );
}

export function ConfigSectionFields({
  title,
  fields,
  overrides,
  onFieldChange,
  onFieldReset,
}: {
  title: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const sectionModifiedCount = modifiedCount(fields, overrides);

  return (
    <section className="grid gap-3 border-b border-line-soft pb-5 last:border-b-0 last:pb-0">
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
        <h3 className="min-w-0 truncate text-sm font-semibold text-ink">{title}</h3>
        <span className="flex shrink-0 items-center gap-1">
          {sectionModifiedCount > 0 && (
            <Badge variant="override">{sectionModifiedCount} set</Badge>
          )}
          <Badge>{fields.length} fields</Badge>
        </span>
      </div>
      <div className="grid gap-x-3 gap-y-4 md:grid-cols-2 2xl:grid-cols-3">
        {fields.map((field) => (
          <ConfigFieldControl
            key={field.key}
            field={field}
            overrides={overrides}
            onChange={onFieldChange}
            onReset={onFieldReset}
          />
        ))}
      </div>
    </section>
  );
}
