import { RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { IconButton } from "@/components/ui/icon-button";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { type ConfigField } from "@/lib/api";
import {
  type OverrideValues,
  configFieldSelectOptions,
  fieldValue,
  hasOverride,
} from "@/lib/config";
import { cn } from "@/lib/utils";

type ConfigFieldControlDensity = "comfortable" | "compact";

const booleanOptions = [
  { value: "true", label: "On" },
  { value: "false", label: "Off" },
] as const;

function BooleanSegmentedControl({
  id,
  label,
  value,
  onChange,
  disabled,
  isCompact,
  className,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
  isCompact: boolean;
  className?: string;
}) {
  return (
    <SegmentedControl
      id={id}
      aria-label={label}
      aria-disabled={disabled || undefined}
      className={cn(
        "grid w-full min-w-0 grid-cols-2 overflow-hidden rounded-control border border-line bg-control-chrome p-0 font-sans font-semibold text-ink transition",
        isCompact ? "h-10 text-[12.5px]" : "h-10 text-[13px]",
        disabled && "opacity-60",
        className,
      )}
    >
      {booleanOptions.map((option) => {
        const isSelected =
          option.value === "true" ? value === "true" : value !== "true";
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={isSelected}
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={cn(
              "min-w-0 border-r border-line-soft px-2 text-center transition last:border-r-0 focus:outline-none focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed",
              isSelected
                ? "bg-control-selected text-violet"
                : "bg-transparent text-ink-faint hover:bg-control-active hover:text-ink",
            )}
          >
            <span className="block min-w-0 truncate">{option.label}</span>
          </button>
        );
      })}
    </SegmentedControl>
  );
}

export function ConfigFieldValueEditor({
  field,
  overrides,
  onChange,
  onReset,
  controlId,
  controlLabel,
  resetLabel,
  resetTitle,
  density = "comfortable",
  disabled = false,
  className,
}: {
  field: ConfigField;
  overrides: OverrideValues;
  onChange: (key: string, value: string) => void;
  onReset: (key: string) => void;
  controlId: string;
  controlLabel?: string;
  resetLabel?: string;
  resetTitle?: string;
  density?: ConfigFieldControlDensity;
  disabled?: boolean;
  className?: string;
}) {
  const value = fieldValue(field, overrides);
  const choices = field.choices;
  const isModified = hasOverride(overrides, field.key);
  const isLocked = field.locked;
  const isControlDisabled = isLocked || disabled;
  const isCompact = density === "compact";
  const showResetButton = isModified;
  const resolvedResetLabel = resetLabel ?? "Reset field override";
  const resetInsetClassName = isCompact ? "right-1" : "right-0.5";
  const resetPaddingClassName = showResetButton ? "pr-11" : undefined;
  const selectTriggerClassName = isCompact
    ? cn("h-10 px-3 py-2 text-[13.5px]", resetPaddingClassName)
    : resetPaddingClassName;
  const inputClassName = cn(
    isCompact ? "h-10 px-3 py-2 text-[13.5px]" : undefined,
    resetPaddingClassName,
  );

  return (
    <div
      className={cn(
        "relative grid min-w-0 items-center",
        isCompact ? "mt-1" : "mt-2",
        className,
      )}
    >
      {field.type === "bool" ? (
        <BooleanSegmentedControl
          id={controlId}
          label={controlLabel ?? field.label}
          value={value}
          disabled={isControlDisabled}
          isCompact={isCompact}
          onChange={(nextValue) => onChange(field.key, nextValue)}
          className={resetPaddingClassName}
        />
      ) : choices.length > 0 ? (
        <SelectOnlyDropdown
          id={controlId}
          label={controlLabel ?? field.label}
          value={value}
          options={configFieldSelectOptions(field, overrides)}
          disabled={isControlDisabled}
          onChange={(nextValue) => onChange(field.key, nextValue)}
          placeholder="None"
          triggerClassName={selectTriggerClassName}
        />
      ) : field.type === "int" || field.type === "float" ? (
        <Input
          id={controlId}
          name={field.key}
          aria-label={controlLabel}
          type="text"
          inputMode={field.type === "float" ? "decimal" : "numeric"}
          autoComplete="off"
          value={value}
          disabled={isControlDisabled}
          onChange={(event) => onChange(field.key, event.target.value)}
          className={inputClassName}
        />
      ) : (
        <Input
          id={controlId}
          name={field.key}
          aria-label={controlLabel}
          autoComplete="off"
          value={value}
          disabled={isControlDisabled}
          onChange={(event) => onChange(field.key, event.target.value)}
          className={inputClassName}
        />
      )}
      {showResetButton && (
        <IconButton
          label={resolvedResetLabel}
          title={resetTitle ?? resolvedResetLabel}
          size="sm"
          variant="edge"
          disabled={isControlDisabled}
          onClick={() => onReset(field.key)}
          icon={<RotateCcw className="h-3.5 w-3.5" aria-hidden />}
          className={cn(
            "absolute top-1/2 z-40 -translate-y-1/2 rounded-[8px] border-line bg-white/[0.055] text-ink-faint shadow-[0_8px_18px_-14px_rgba(0,0,0,0.95)] hover:bg-white/[0.09] hover:text-ink",
            resetInsetClassName,
          )}
        />
      )}
    </div>
  );
}

export function ConfigFieldControl({
  field,
  overrides,
  onChange,
  onReset,
  density = "comfortable",
  idPrefix = "field",
  disabled = false,
  disabledReason,
}: {
  field: ConfigField;
  overrides: OverrideValues;
  onChange: (key: string, value: string) => void;
  onReset: (key: string) => void;
  density?: ConfigFieldControlDensity;
  idPrefix?: string;
  disabled?: boolean;
  disabledReason?: string;
}) {
  const id = `${idPrefix}-${field.key}`;
  const isModified = hasOverride(overrides, field.key);
  const isLocked = field.locked;
  const isControlDisabled = disabled && !isLocked;
  const isCompact = density === "compact";
  const overrideBadgeClassName = isCompact ? "px-1 py-0.5 text-xs" : undefined;

  return (
    <div
      className={cn(
        "grid w-full transition",
        isCompact ? "gap-1.5 py-1.5" : "gap-2 py-1.5",
        isModified && "border-l-2 border-violet/40 pl-2",
        isLocked &&
          "rounded-[10px] border-l-2 border-amber/55 bg-amber/[0.055] pl-2 pr-2 shadow-[inset_0_0_0_1px_rgba(255,209,102,0.04)]",
        isControlDisabled &&
          "rounded-[10px] border-l-2 border-line bg-white/[0.025] pl-2 pr-2 opacity-65",
      )}
    >
      <label className={cn("grid", isCompact ? "gap-1" : "gap-1.5")} htmlFor={id}>
        <span className="flex min-w-0 items-start justify-between gap-2">
          <span className="min-w-0 text-[13px] font-semibold leading-5 text-ink [overflow-wrap:anywhere]">
            {field.label}
          </span>
        </span>
        {(isModified || isLocked || isControlDisabled) && (
          <span className="flex min-w-0 flex-wrap items-center gap-1.5 text-xs font-medium text-ink-faint">
            {isModified && (
              <Badge
                className={cn(
                  "border-violet/30 bg-violet/15 text-violet",
                  overrideBadgeClassName,
                )}
              >
                override
              </Badge>
            )}
            {isLocked && (
              <Badge variant="preset" className={overrideBadgeClassName}>
                preset
              </Badge>
            )}
            {isLocked && field.lockedReason && (
              <span className="min-w-0 text-xs leading-4 text-ink-dim">
                {field.lockedReason}
              </span>
            )}
            {isControlDisabled && disabledReason && (
              <span className="min-w-0 text-xs leading-4 text-ink-dim">
                {disabledReason}
              </span>
            )}
          </span>
        )}
      </label>
      <ConfigFieldValueEditor
        field={field}
        overrides={overrides}
        onChange={onChange}
        onReset={onReset}
        controlId={id}
        density={density}
        disabled={disabled}
        resetTitle={`Reset ${field.label}`}
      />
    </div>
  );
}
