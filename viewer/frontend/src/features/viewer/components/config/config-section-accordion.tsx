import { useEffect, useMemo, useState } from "react";
import { ChevronDown, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { type ConfigField } from "@/lib/api";
import {
  type ConfigFieldGroup,
  type ConfigSection,
  type OverrideValues,
  boundaryModelFieldGroups,
  concreteConfigOptionChoice,
  controlledSectionState,
  displayConfigFieldLabel,
  displayConfigSectionDescription,
  displayConfigSectionTitle,
  fieldValue,
  hasOverride as fieldHasOverride,
  inheritedHiddenModelFieldHint,
  inheritedStackSectionHint,
  modifiedCount,
  presetOwnedCount,
  sectionCountLabel,
  sectionElementId,
  sectionTitlesFromSignature,
} from "@/lib/config";
import { cn } from "@/lib/utils";
import { ConfigFieldControl } from "@/features/viewer/components/config/config-field-control";
import { ConfigMetricBadge } from "@/features/viewer/components/config/config-metric-badge";
import { surfacePanelClassName } from "@/features/viewer/components/shared/surface-panel";

const EMPTY_CONFIG_SECTIONS: ConfigSection[] = [];

function isEnabledConfigValue(value: string) {
  return ["true", "1", "yes", "on"].includes(value.trim().toLowerCase());
}

function SectionHeaderControl({
  field,
  displayLabel,
  overrides,
  controlId,
  onChange,
  onReset,
}: {
  field: ConfigField;
  displayLabel?: string;
  overrides: OverrideValues;
  controlId: string;
  onChange: (key: string, value: string) => void;
  onReset: (key: string) => void;
}) {
  const value = fieldValue(field, overrides);
  const label = displayLabel ?? field.label;
  const isEnabled = isEnabledConfigValue(value);
  const isModified = fieldHasOverride(overrides, field.key);
  const isLocked = Boolean(field.locked);
  const isResetDisabled = !isModified || isLocked;

  return (
    <span
      data-config-section-header-control={field.key}
      className="inline-flex min-w-0 items-center gap-2 rounded-[9px] border border-line bg-black/25 px-2.5 py-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.025)]"
    >
      <Switch
        id={controlId}
        aria-label={label}
        disabled={isLocked}
        checked={isEnabled}
        onCheckedChange={(checked) => onChange(field.key, String(checked))}
      />
      {isModified && (
        <button
          type="button"
          aria-label={`Reset ${label}`}
          title={`Reset ${label}`}
          disabled={isResetDisabled}
          onClick={() => onReset(field.key)}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border border-line bg-white/[0.035] text-ink-faint transition hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-white/[0.035] disabled:hover:text-ink-faint"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden />
        </button>
      )}
      {isModified && (
        <Badge variant="override" className="px-1 py-0.5 text-xs">
          override
        </Badge>
      )}
      {isLocked && (
        <Badge variant="preset" className="px-1 py-0.5 text-xs">
          preset
        </Badge>
      )}
      {field.lockedReason && <span className="sr-only">{field.lockedReason}</span>}
    </span>
  );
}

function descendantFieldKeys(sections: ConfigSection[]) {
  const keys = new Set<string>();

  function collect(section: ConfigSection) {
    for (const field of section.fields) {
      keys.add(field.key);
    }
    for (const child of section.children ?? []) {
      collect(child);
    }
  }

  for (const section of sections) {
    collect(section);
  }

  return keys;
}

function InheritedConfigFieldRow({
  hint,
  overrides,
}: {
  hint: NonNullable<ReturnType<typeof inheritedHiddenModelFieldHint>>;
  overrides: OverrideValues;
}) {
  const value = fieldValue(hint.sourceField, overrides);
  const isModified = fieldHasOverride(overrides, hint.sourceField.key);

  return (
    <div
      className="grid gap-1.5 rounded-[10px] border-l-2 border-line bg-white/[0.025] px-2.5 py-2"
      data-config-inherited-field={hint.sourceField.key}
      title={hint.title}
    >
      <div className="flex min-w-0 items-start justify-between gap-2">
        <span className="min-w-0 text-[13px] font-semibold leading-5 text-ink">
          {hint.label}
        </span>
        <span className="flex shrink-0 items-center gap-1.5">
          {isModified && (
            <Badge variant="override" className="px-1 py-0.5 text-xs">
              override
            </Badge>
          )}
          <Badge
            variant="info"
            className="px-1 py-0.5 text-xs"
            aria-label={hint.title}
          >
            From {hint.sourceTitle}
          </Badge>
        </span>
      </div>
      <div className="flex h-10 min-w-0 items-center rounded-control border border-line-soft bg-black/25 px-3 text-[13.5px] font-medium text-ink-dim">
        <span className="min-w-0 truncate">{value}</span>
      </div>
    </div>
  );
}

function BoundaryModelGroupAccordion({
  id,
  sectionTitle,
  group,
  overrides,
  isOpen,
  onToggle,
  onEnabledChange,
  onFieldChange,
  onFieldReset,
}: {
  id: string;
  sectionTitle: string;
  group: ConfigFieldGroup;
  overrides: OverrideValues;
  isOpen: boolean;
  onToggle: () => void;
  onEnabledChange: (checked: boolean) => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const isEnabled = group.isEnabled === true;
  const controlField = group.controlField;
  const controlFieldLabel = controlField
    ? displayConfigFieldLabel(controlField, sectionTitle, group.title)
    : undefined;
  const fieldCount = group.fields.length;
  const overrideCount = modifiedCount(group.fields, overrides);
  const presetCount = presetOwnedCount(group.fields);
  const hasPreset = presetCount > 0;
  const firstConcreteOption = controlField
    ? concreteConfigOptionChoice(controlField)
    : undefined;
  const isSwitchDisabled =
    !controlField ||
    Boolean(controlField.locked) ||
    (!isEnabled && firstConcreteOption === undefined);
  const panelId = `${id}-fields`;

  return (
    <section
      className={cn(
        "overflow-visible rounded-[10px] border border-line-soft bg-black/[0.12] transition",
        isEnabled
          ? "shadow-[0_16px_34px_-30px_rgba(0,0,0,0.95)]"
          : "opacity-82",
        overrideCount > 0 && "border-violet/35 bg-violet/[0.045]",
        hasPreset && "border-amber/35 bg-amber/[0.035]",
      )}
      data-config-field-group={group.title}
    >
      <div className="grid min-h-12 grid-cols-[minmax(0,1fr)_auto] items-stretch overflow-hidden rounded-t-[9px]">
        <h4 className="h-full min-w-0">
          <button
            type="button"
            aria-expanded={isEnabled && isOpen}
            aria-controls={panelId}
            aria-label={`${group.title} boundary model group, ${sectionCountLabel(
              fieldCount,
              "field",
            )}, ${sectionCountLabel(overrideCount, "override")}${
              hasPreset ? `, ${sectionCountLabel(presetCount, "preset")}` : ""
            }${!isEnabled && controlFieldLabel ? `, select ${controlFieldLabel} to open` : ""}`}
            disabled={!isEnabled}
            onClick={onToggle}
            className={cn(
              "flex h-full min-h-12 w-full min-w-0 items-center gap-3 overflow-hidden px-3 py-2.5 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed",
              isEnabled
                ? "bg-white/[0.035] hover:bg-white/[0.065]"
                : "bg-white/[0.018] hover:bg-white/[0.018]",
            )}
          >
            <span className="flex min-w-0 items-center gap-2 overflow-hidden">
              <ChevronDown
                className={cn(
                  "h-4 w-4 shrink-0 text-ink-faint transition-transform",
                  (!isEnabled || !isOpen) && "-rotate-90",
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
          {controlField && (
            <Switch
              aria-label={controlFieldLabel}
              checked={isEnabled}
              disabled={isSwitchDisabled}
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
        hidden={!isEnabled || !isOpen}
        className={
          isEnabled && isOpen
            ? "grid gap-x-3 gap-y-3 border-t border-line-soft bg-black/[0.08] px-3 py-3 md:grid-cols-2 2xl:grid-cols-3"
            : "hidden"
        }
      >
        {isEnabled &&
          isOpen &&
          group.fields.map((field) => (
            <ConfigFieldControl
              key={field.key}
              field={field}
              displayLabel={displayConfigFieldLabel(field, sectionTitle, group.title)}
              overrides={overrides}
              onChange={onFieldChange}
              onReset={onFieldReset}
              density="compact"
              idPrefix={`${id}-field`}
            />
          ))}
      </div>
    </section>
  );
}

export function ConfigSectionAccordion({
  id,
  refCallback,
  title,
  fields,
  allFields: allFieldsProp,
  showInheritedFields = true,
  childSections: childSectionsProp,
  overrides,
  isOpen,
  controlField,
  disabledReason,
  autoOpenKey,
  onToggle,
  onFieldChange,
  onFieldReset,
}: {
  id: string;
  refCallback: (element: HTMLElement | null) => void;
  title: string;
  fields: ConfigField[];
  allFields?: ConfigField[];
  showInheritedFields?: boolean;
  childSections?: ConfigSection[];
  overrides: OverrideValues;
  isOpen: boolean;
  controlField?: ConfigField;
  disabledReason?: string;
  autoOpenKey?: string;
  onToggle: () => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const childSections = childSectionsProp ?? EMPTY_CONFIG_SECTIONS;
  const allFields = allFieldsProp ?? fields;
  const sectionModifiedCount = modifiedCount(fields, overrides);
  const sectionPresetOwnedCount = presetOwnedCount(fields);
  const displayTitle = displayConfigSectionTitle(title);
  const displayDescription = displayConfigSectionDescription(title);
  const controlFieldId = controlField ? `${id}-control-${controlField.key}` : undefined;
  const controlFieldDisplayLabel = controlField
    ? displayConfigFieldLabel(controlField, title)
    : undefined;
  const childFieldKeys = useMemo(
    () => descendantFieldKeys(childSections),
    [childSections],
  );
  const bodyFields = fields.filter(
    (field) => field.key !== controlField?.key && !childFieldKeys.has(field.key),
  );
  const bodyFieldGroups = boundaryModelFieldGroups(title, bodyFields, overrides);
  const boundaryGroupTitleSignature = (
    bodyFieldGroups?.map((group) => group.title) ?? []
  ).join(String.fromCharCode(0));
  const enabledBoundaryGroupTitleSignature = (
    bodyFieldGroups
      ?.filter((group) => group.isEnabled)
      .map((group) => group.title) ?? []
  ).join(
    String.fromCharCode(0),
  );
  const childSectionTitles = useMemo(
    () => childSections.map((section) => section.title),
    [childSections],
  );
  const childSectionTitleSignature = childSectionTitles.join(String.fromCharCode(0));
  const [openChildSectionTitles, setOpenChildSectionTitles] = useState(
    () => new Set(childSectionTitles),
  );
  const [openBoundaryGroupTitles, setOpenBoundaryGroupTitles] = useState(
    () => new Set(sectionTitlesFromSignature(enabledBoundaryGroupTitleSignature)),
  );
  const sectionHasOverride = sectionModifiedCount > 0;
  const hasPreset = sectionPresetOwnedCount > 0;
  const hasBoth = sectionHasOverride && hasPreset;
  const isDisabled = Boolean(disabledReason);
  const panelId = `${id}-fields`;
  const stackInheritanceHint = inheritedStackSectionHint(
    { title, fields, controlFieldKey: controlField?.key },
    overrides,
  );
  const inheritedHiddenFieldHint = showInheritedFields
    ? inheritedHiddenModelFieldHint(
        { title, fields },
        allFields,
      )
    : undefined;
  const stateContainerClass = hasBoth
    ? "border-amber/35 bg-[linear-gradient(135deg,rgba(255,209,102,0.075),rgba(167,139,250,0.105))] shadow-[0_20px_46px_-32px_rgba(167,139,250,0.35)] ring-1 ring-violet/25 hover:border-violet/45"
    : sectionHasOverride
      ? "border-violet/35 bg-violet/[0.06] shadow-[0_18px_42px_-32px_rgba(167,139,250,0.35)] hover:border-violet/45"
      : hasPreset
        ? "border-amber/35 bg-amber/[0.045] shadow-[0_18px_42px_-32px_rgba(255,209,102,0.25)] hover:border-amber/45"
        : "";
  const stateHeaderClass = hasBoth
    ? "bg-[linear-gradient(90deg,rgba(255,209,102,0.12),rgba(167,139,250,0.13))] hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.16),rgba(167,139,250,0.17))]"
    : sectionHasOverride
      ? "bg-violet/[0.08] hover:bg-violet/[0.12]"
      : hasPreset
        ? "bg-amber/[0.08] hover:bg-amber/[0.12]"
    : "";

  useEffect(() => {
    setOpenChildSectionTitles(
      new Set(sectionTitlesFromSignature(childSectionTitleSignature)),
    );
  }, [autoOpenKey, childSectionTitleSignature]);

  useEffect(() => {
    if (!boundaryGroupTitleSignature) {
      return;
    }
    const groupTitles = sectionTitlesFromSignature(boundaryGroupTitleSignature);
    const enabledTitles = sectionTitlesFromSignature(
      enabledBoundaryGroupTitleSignature,
    );
    setOpenBoundaryGroupTitles((current) => {
      const next = new Set(current);
      const enabledTitleSet = new Set(enabledTitles);
      for (const title of groupTitles) {
        if (!enabledTitleSet.has(title)) {
          next.delete(title);
        }
      }
      for (const title of enabledTitles) {
        next.add(title);
      }
      return next;
    });
  }, [boundaryGroupTitleSignature, enabledBoundaryGroupTitleSignature]);

  function toggleChildSection(title: string) {
    setOpenChildSectionTitles((current) => {
      const next = new Set(current);
      if (next.has(title)) {
        next.delete(title);
      } else {
        next.add(title);
      }
      return next;
    });
  }

  function toggleBoundaryGroup(title: string) {
    setOpenBoundaryGroupTitles((current) => {
      const next = new Set(current);
      if (next.has(title)) {
        next.delete(title);
      } else {
        next.add(title);
      }
      return next;
    });
  }

  function setBoundaryGroupEnabled(group: ConfigFieldGroup, checked: boolean) {
    const controlField = group.controlField;
    if (!controlField) {
      return;
    }
    if (!checked) {
      onFieldChange(controlField.key, "");
      setOpenBoundaryGroupTitles((current) => {
        const next = new Set(current);
        next.delete(group.title);
        return next;
      });
      return;
    }

    const firstConcreteOption = concreteConfigOptionChoice(controlField);
    if (firstConcreteOption !== undefined) {
      onFieldChange(controlField.key, firstConcreteOption);
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
        "relative overflow-visible px-0 py-0 shadow-[0_16px_40px_-30px_rgba(0,0,0,0.95)] transition duration-150 hover:-translate-y-px hover:border-line hover:shadow-[0_20px_44px_-32px_rgba(0,0,0,0.95)] focus-within:z-30 focus-within:-translate-y-px focus-within:ring-2 focus-within:ring-focus motion-reduce:transform-none",
        !isOpen &&
          "border-line-soft bg-white/[0.012] shadow-[0_10px_28px_-26px_rgba(0,0,0,0.9)]",
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
              fields.length,
              "field",
            )}, ${sectionCountLabel(sectionModifiedCount, "override")}${
              hasPreset
                ? `, ${sectionCountLabel(sectionPresetOwnedCount, "preset")}`
                : ""
            }${isDisabled && controlFieldDisplayLabel ? `, enable ${controlFieldDisplayLabel} to open` : ""}`}
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
              field={controlField}
              displayLabel={controlFieldDisplayLabel}
              overrides={overrides}
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
        {bodyFields.length > 0 && bodyFieldGroups && bodyFieldGroups.length > 0 && (
          <div className="grid gap-2.5">
            {bodyFieldGroups.map((group, index) => (
              <BoundaryModelGroupAccordion
                key={group.title}
                id={sectionElementId(index, group.title, `${id}-boundary-group`)}
                sectionTitle={title}
                group={group}
                overrides={overrides}
                isOpen={openBoundaryGroupTitles.has(group.title)}
                onToggle={() => toggleBoundaryGroup(group.title)}
                onEnabledChange={(checked) => setBoundaryGroupEnabled(group, checked)}
                onFieldChange={onFieldChange}
                onFieldReset={onFieldReset}
              />
            ))}
          </div>
        )}
        {bodyFields.length > 0 && (!bodyFieldGroups || bodyFieldGroups.length === 0) && (
          <div className="grid gap-x-3 gap-y-3 md:grid-cols-2 2xl:grid-cols-3">
            {inheritedHiddenFieldHint && (
              <InheritedConfigFieldRow
                hint={inheritedHiddenFieldHint}
                overrides={overrides}
              />
            )}
            {isOpen &&
              bodyFields.map((field) => (
                <ConfigFieldControl
                  key={field.key}
                  field={field}
                  displayLabel={displayConfigFieldLabel(field, title)}
                  overrides={overrides}
                  onChange={onFieldChange}
                  onReset={onFieldReset}
                  density="compact"
                  idPrefix={`${id}-field`}
                />
              ))}
          </div>
        )}
        {isOpen && childSections.length > 0 && (
          <div className="grid gap-3">
            {childSections.map((childSection, index) => {
              const childSectionId = sectionElementId(
                index,
                childSection.title,
                `${id}-nested-section`,
              );
              const childState = controlledSectionState(childSection, overrides);
              const childDisabledReason = disabledReason ?? childState.disabledReason;
              const isChildOpen =
                !childDisabledReason && openChildSectionTitles.has(childSection.title);

              return (
                <ConfigSectionAccordion
                  key={childSection.title}
                  id={childSectionId}
                  refCallback={() => undefined}
                  title={childSection.title}
                  fields={childSection.fields}
                  allFields={allFields}
                  showInheritedFields={showInheritedFields}
                  childSections={childSection.children}
                  overrides={overrides}
                  isOpen={isChildOpen}
                  controlField={childState.controlField}
                  disabledReason={childDisabledReason}
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
  const displayTitle = displayConfigSectionTitle(title);

  return (
    <section className="grid gap-3 border-b border-line-soft pb-5 last:border-b-0 last:pb-0">
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
        <h3 className="min-w-0 truncate text-sm font-semibold text-ink">
          {displayTitle}
        </h3>
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
            displayLabel={displayConfigFieldLabel(field, title)}
            overrides={overrides}
            onChange={onFieldChange}
            onReset={onFieldReset}
          />
        ))}
      </div>
    </section>
  );
}
