import { useEffect, useState } from "react";
import { ChevronDown, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  type RuntimeDefaultsFieldGroupPresentation,
  type RuntimeDefaultsInheritedFieldPresentation,
  type RuntimeDefaultsSectionPresentation,
} from "@/features/workbench/state/full-config/runtime-defaults-schema-presentation";
import { cn } from "@/lib/utils";
import { ConfigFieldControl } from "@/features/workbench/components/config/config-field-control";
import { ConfigMetricBadge } from "@/features/workbench/components/config/config-metric-badge";
import { surfacePanelClassName } from "@/components/ui/surface-panel";

function sectionCountLabel(
  count: number,
  noun: "field" | "override" | "preset",
) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function titlesFromSignature(signature: string) {
  return signature ? signature.split("\u0000") : [];
}

function SectionHeaderControl({
  presentation,
  controlId,
  onChange,
  onReset,
}: {
  presentation: NonNullable<RuntimeDefaultsSectionPresentation["controlField"]>;
  controlId: string;
  onChange: (key: string, value: string) => void;
  onReset: (key: string) => void;
}) {
  const field = presentation.schema;
  const isResetDisabled = !presentation.isModified || presentation.isPresetOwned;

  return (
    <span
      data-config-section-header-control={presentation.key}
      className="inline-flex min-w-0 items-center gap-2 rounded-control border border-line bg-black/25 px-2.5 py-1.5 shadow-control"
    >
      <Switch
        id={controlId}
        aria-label={presentation.label}
        disabled={presentation.isPresetOwned}
        checked={presentation.isEnabledValue}
        onCheckedChange={(checked) => onChange(presentation.key, String(checked))}
      />
      {presentation.modeLabel && (
        <span className="max-w-[13rem] truncate text-xs font-semibold text-ink-dim">
          {presentation.modeLabel}
        </span>
      )}
      {presentation.isModified && (
        <button
          type="button"
          aria-label={`Reset ${presentation.label}`}
          title={`Reset ${presentation.label}`}
          disabled={isResetDisabled}
          onClick={() => onReset(presentation.key)}
          className="flex h-touch w-touch shrink-0 items-center justify-center rounded-control-md border border-line bg-control text-ink-faint transition-colors hover:bg-control-hover hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-control disabled:hover:text-ink-faint md:h-control-sm md:w-control-sm"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden />
        </button>
      )}
      {presentation.isModified && (
        <Badge variant="override" className="px-1 py-0.5 text-xs">
          override
        </Badge>
      )}
      {presentation.isPresetOwned && (
        <Badge variant="preset" className="px-1 py-0.5 text-xs">
          preset
        </Badge>
      )}
      {field.lockedReason && <span className="sr-only">{field.lockedReason}</span>}
    </span>
  );
}

function InheritedConfigFieldRow({
  presentation,
}: {
  presentation: RuntimeDefaultsInheritedFieldPresentation;
}) {
  return (
    <div
      className="grid gap-1.5 rounded-control border-l-2 border-line bg-white/[0.025] px-2.5 py-2"
      data-config-inherited-field={presentation.field.key}
      title={presentation.title}
    >
      <div className="flex min-w-0 items-start justify-between gap-2">
        <span className="min-w-0 type-compact font-semibold leading-5 text-ink">
          {presentation.label}
        </span>
        <span className="flex shrink-0 items-center gap-1.5">
          {presentation.field.isModified && (
            <Badge variant="override" className="px-1 py-0.5 text-xs">
              override
            </Badge>
          )}
          <Badge
            variant="info"
            className="px-1 py-0.5 text-xs"
            aria-label={presentation.title}
          >
            From {presentation.sourceTitle}
          </Badge>
        </span>
      </div>
      <div className="flex h-10 min-w-0 items-center rounded-control border border-line-soft bg-black/25 px-3 type-compact font-medium text-ink-dim">
        <span className="min-w-0 truncate">{presentation.field.value}</span>
      </div>
    </div>
  );
}

function BoundaryModelGroupAccordion({
  group,
  isOpen,
  onToggle,
  onEnabledChange,
  onFieldChange,
  onFieldReset,
}: {
  group: RuntimeDefaultsFieldGroupPresentation;
  isOpen: boolean;
  onToggle: () => void;
  onEnabledChange: (checked: boolean) => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const { fieldCount, overrideCount, presetCount } = group.metrics;
  const hasPreset = presetCount > 0;
  const panelId = `${group.id}-fields`;

  return (
    <section
      className={cn(
        "overflow-visible rounded-control border border-line-soft bg-black/[0.12] transition",
        group.isEnabled
          ? "shadow-card"
          : "opacity-82",
        overrideCount > 0 && "border-violet/35 bg-violet/[0.045]",
        hasPreset && "border-amber/35 bg-amber/[0.035]",
      )}
      data-config-field-group={group.title}
    >
      <div className="grid min-h-12 grid-cols-[minmax(0,1fr)_auto] items-stretch overflow-hidden rounded-t-control">
        <h4 className="h-full min-w-0">
          <button
            type="button"
            aria-expanded={group.isEnabled && isOpen}
            aria-controls={panelId}
            aria-label={`${group.title} boundary model group, ${sectionCountLabel(
              fieldCount,
              "field",
            )}, ${sectionCountLabel(overrideCount, "override")}${
              hasPreset ? `, ${sectionCountLabel(presetCount, "preset")}` : ""
            }${
              !group.isEnabled && group.controlField
                ? `, select ${group.controlField.label} to open`
                : ""
            }`}
            disabled={!group.isEnabled}
            onClick={onToggle}
            className={cn(
              "flex h-full min-h-12 w-full min-w-0 items-center gap-3 overflow-hidden px-3 py-2.5 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed",
              group.isEnabled
                ? "bg-white/[0.035] hover:bg-white/[0.065]"
                : "bg-white/[0.018] hover:bg-white/[0.018]",
            )}
          >
            <span className="flex min-w-0 items-center gap-2 overflow-hidden">
              <ChevronDown
                className={cn(
                  "h-4 w-4 shrink-0 text-ink-faint transition-transform",
                  (!group.isEnabled || !isOpen) && "-rotate-90",
                )}
                aria-hidden
              />
              <span className="min-w-0 truncate text-sm font-semibold text-ink">
                {group.title}
              </span>
            </span>
          </button>
        </h4>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-1.5 border-l border-line-soft bg-black/15 px-3 py-2">
          {group.controlField && (
            <Switch
              aria-label={group.controlField.label}
              checked={group.isEnabled}
              disabled={group.isSwitchDisabled}
              onCheckedChange={onEnabledChange}
            />
          )}
          {group.stackHint && (
            <Badge
              variant="info"
              data-config-boundary-stack-hint={group.stackHint.sourceTitle}
              title={group.stackHint.title}
              aria-label={group.stackHint.title}
              className="h-[23px] items-center px-1.5 py-0"
            >
              {group.stackHint.label}
            </Badge>
          )}
          <ConfigMetricBadge count={fieldCount} kind="fields" focusable={false} />
          <ConfigMetricBadge
            count={overrideCount}
            kind="overrides"
            variant={overrideCount > 0 ? "override" : "default"}
            focusable={false}
          />
          {hasPreset && (
            <Badge variant="preset" className="h-[23px] items-center px-1.5 py-0">
              {presetCount} preset
            </Badge>
          )}
        </div>
      </div>
      <div
        id={panelId}
        hidden={!group.isEnabled || !isOpen}
        className={
          group.isEnabled && isOpen
            ? "grid gap-x-3 gap-y-3 border-t border-line-soft bg-black/[0.08] px-3 py-3 md:grid-cols-2 2xl:grid-cols-3"
            : "hidden"
        }
      >
        {group.isEnabled &&
          isOpen &&
          group.fields.map((field) => (
            <ConfigFieldControl
              key={field.key}
              presentation={field}
              onChange={onFieldChange}
              onReset={onFieldReset}
              density="compact"
              idPrefix={`${group.id}-field`}
            />
          ))}
      </div>
    </section>
  );
}

export function ConfigSectionAccordion({
  section,
  refCallback,
  isOpen,
  autoOpenKey,
  onToggle,
  onFieldChange,
  onFieldReset,
}: {
  section: RuntimeDefaultsSectionPresentation;
  refCallback: (element: HTMLElement | null) => void;
  isOpen: boolean;
  autoOpenKey?: string;
  onToggle: () => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const {
    id,
    displayTitle,
    displayDescription,
    bodyFields,
    fieldGroups,
    children,
    controlField,
    isDisabled,
    directMetrics,
    stackInheritanceHint,
    inheritedField,
    childTitleSignature,
    groupTitleSignature,
    enabledGroupTitleSignature,
  } = section;
  const childSectionTitles = titlesFromSignature(childTitleSignature);
  const [openChildSectionTitles, setOpenChildSectionTitles] = useState(
    () => new Set(childSectionTitles),
  );
  const [openBoundaryGroupTitles, setOpenBoundaryGroupTitles] = useState(
    () => new Set(titlesFromSignature(enabledGroupTitleSignature)),
  );
  const { fieldCount, overrideCount, presetCount, state } = directMetrics;
  const sectionHasOverride = overrideCount > 0;
  const hasPreset = presetCount > 0;
  const hasBoth = state === "override-and-preset";
  const panelId = `${id}-fields`;
  const controlFieldId = controlField ? `${id}-control-${controlField.key}` : undefined;
  const stateContainerClass = hasBoth
    ? "border-amber/35 bg-config-preset shadow-card-accent ring-1 ring-violet/25 hover:border-violet/45"
    : sectionHasOverride
      ? "border-violet/35 bg-violet/[0.06] shadow-card-accent hover:border-violet/45"
      : hasPreset
        ? "border-amber/35 bg-amber/[0.045] shadow-card-warning hover:border-amber/45"
        : "";
  const stateHeaderClass = hasBoth
    ? "bg-config-preset-header hover:bg-config-preset-header-hover"
    : sectionHasOverride
      ? "bg-violet/[0.08] hover:bg-violet/[0.12]"
      : hasPreset
        ? "bg-amber/[0.08] hover:bg-amber/[0.12]"
        : "";

  useEffect(() => {
    setOpenChildSectionTitles(new Set(titlesFromSignature(childTitleSignature)));
  }, [autoOpenKey, childTitleSignature]);

  useEffect(() => {
    if (!groupTitleSignature) {
      return;
    }
    const groupTitles = titlesFromSignature(groupTitleSignature);
    const enabledTitles = titlesFromSignature(enabledGroupTitleSignature);
    setOpenBoundaryGroupTitles((current) => {
      const next = new Set(current);
      const enabledTitleSet = new Set(enabledTitles);
      for (const groupTitle of groupTitles) {
        if (!enabledTitleSet.has(groupTitle)) {
          next.delete(groupTitle);
        }
      }
      for (const enabledTitle of enabledTitles) {
        next.add(enabledTitle);
      }
      return next;
    });
  }, [groupTitleSignature, enabledGroupTitleSignature]);

  function toggleChildSection(childTitle: string) {
    setOpenChildSectionTitles((current) => {
      const next = new Set(current);
      if (next.has(childTitle)) {
        next.delete(childTitle);
      } else {
        next.add(childTitle);
      }
      return next;
    });
  }

  function toggleBoundaryGroup(groupTitle: string) {
    setOpenBoundaryGroupTitles((current) => {
      const next = new Set(current);
      if (next.has(groupTitle)) {
        next.delete(groupTitle);
      } else {
        next.add(groupTitle);
      }
      return next;
    });
  }

  function setBoundaryGroupEnabled(
    group: RuntimeDefaultsFieldGroupPresentation,
    checked: boolean,
  ) {
    if (!group.controlField) {
      return;
    }
    if (!checked) {
      onFieldChange(group.controlField.key, "");
      setOpenBoundaryGroupTitles((current) => {
        const next = new Set(current);
        next.delete(group.title);
        return next;
      });
      return;
    }
    if (group.firstConcreteOption !== undefined) {
      onFieldChange(group.controlField.key, group.firstConcreteOption);
      setOpenBoundaryGroupTitles((current) => {
        const next = new Set(current);
        next.add(group.title);
        return next;
      });
    }
  }

  return (
    <section
      ref={refCallback}
      className={cn(
        surfacePanelClassName,
        "relative overflow-visible px-0 py-0 shadow-card transition duration-150 hover:-translate-y-px hover:border-line hover:shadow-card-hover focus-within:z-30 focus-within:-translate-y-px focus-within:ring-2 focus-within:ring-focus motion-reduce:transform-none",
        !isOpen &&
          "border-line-soft bg-white/[0.012] shadow-card-subtle",
        isDisabled && "hover:translate-y-0",
        stateContainerClass,
      )}
    >
      <div
        className={cn(
          "grid min-h-12 grid-cols-[minmax(0,1fr)_auto] items-stretch overflow-hidden transition",
          isOpen
            ? "bg-white/[0.055] hover:bg-white/[0.075]"
            : "bg-white/[0.025] hover:bg-white/[0.055]",
          isDisabled && "hover:bg-white/[0.025]",
          stateHeaderClass,
        )}
      >
        <h3 className="h-full min-w-0">
          <button
            type="button"
            aria-expanded={isOpen}
            aria-controls={panelId}
            aria-label={`${displayTitle} section, ${sectionCountLabel(
              fieldCount,
              "field",
            )}, ${sectionCountLabel(overrideCount, "override")}${
              hasPreset ? `, ${sectionCountLabel(presetCount, "preset")}` : ""
            }${
              isDisabled && controlField
                ? `, enable ${controlField.label} to open`
                : ""
            }`}
            disabled={isDisabled}
            onClick={onToggle}
            className={cn(
              "flex h-full min-h-12 w-full min-w-0 items-center gap-3 overflow-hidden px-3 py-2.5 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed",
              isOpen
                ? "bg-white/[0.055] hover:bg-white/[0.075]"
                : "bg-white/[0.025] hover:bg-white/[0.055]",
              isDisabled && "hover:bg-white/[0.025]",
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
              <span className="grid min-w-0 gap-0.5">
                <span className="min-w-0 truncate text-sm font-semibold text-ink">
                  {displayTitle}
                </span>
                {displayDescription && (
                  <span className="min-w-0 truncate text-xs font-medium text-ink-dim">
                    {displayDescription}
                  </span>
                )}
              </span>
            </span>
          </button>
        </h3>
        <div
          className={cn(
            controlField
              ? "flex min-w-0 shrink-0 flex-wrap items-center justify-end gap-1.5 px-3 py-2 max-w-[min(100%,34rem)]"
              : "flex shrink-0 flex-wrap items-center justify-end gap-1.5 px-3 py-2",
          )}
        >
          {controlField && controlFieldId && (
            <SectionHeaderControl
              presentation={controlField}
              onChange={onFieldChange}
              onReset={onFieldReset}
              controlId={controlFieldId}
            />
          )}
          {stackInheritanceHint && (
            <Badge
              variant={stackInheritanceHint.isCustom ? "success" : "info"}
              data-config-section-stack-hint={stackInheritanceHint.sourceTitle}
              title={stackInheritanceHint.title}
              aria-label={stackInheritanceHint.title}
              className="h-[23px] shrink-0 items-center px-1.5 py-0"
            >
              {stackInheritanceHint.label}
            </Badge>
          )}
          <ConfigMetricBadge count={fieldCount} kind="fields" focusable={false} />
          <ConfigMetricBadge
            count={overrideCount}
            kind="overrides"
            variant={overrideCount > 0 ? "override" : "default"}
            focusable={false}
          />
          {hasPreset && (
            <Badge variant="preset" className="h-[23px] items-center px-1.5 py-0">
              {presetCount} preset
            </Badge>
          )}
        </div>
      </div>
      <div
        id={panelId}
        hidden={!isOpen}
        className={
          isOpen
            ? "grid gap-3 border-t border-line-soft bg-black/[0.08] px-3 py-3"
            : "hidden"
        }
      >
        {bodyFields.length > 0 && fieldGroups && fieldGroups.length > 0 && (
          <div className="grid gap-2.5">
            {fieldGroups.map((group) => (
              <BoundaryModelGroupAccordion
                key={group.title}
                group={group}
                isOpen={openBoundaryGroupTitles.has(group.title)}
                onToggle={() => toggleBoundaryGroup(group.title)}
                onEnabledChange={(checked) => setBoundaryGroupEnabled(group, checked)}
                onFieldChange={onFieldChange}
                onFieldReset={onFieldReset}
              />
            ))}
          </div>
        )}
        {bodyFields.length > 0 && (!fieldGroups || fieldGroups.length === 0) && (
          <div className="grid gap-x-3 gap-y-3 md:grid-cols-2 2xl:grid-cols-3">
            {inheritedField && (
              <InheritedConfigFieldRow presentation={inheritedField} />
            )}
            {isOpen &&
              bodyFields.map((field) => (
                <ConfigFieldControl
                  key={field.key}
                  presentation={field}
                  onChange={onFieldChange}
                  onReset={onFieldReset}
                  density="compact"
                  idPrefix={`${id}-field`}
                />
              ))}
          </div>
        )}
        {isOpen && children.length > 0 && (
          <div className="grid gap-3">
            {children.map((childSection) => {
              const isChildOpen =
                !childSection.isDisabled &&
                openChildSectionTitles.has(childSection.title);
              return (
                <ConfigSectionAccordion
                  key={childSection.title}
                  section={childSection}
                  refCallback={() => undefined}
                  isOpen={isChildOpen}
                  autoOpenKey={autoOpenKey}
                  onToggle={() => toggleChildSection(childSection.title)}
                  onFieldChange={onFieldChange}
                  onFieldReset={onFieldReset}
                />
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
