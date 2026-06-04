import { RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { type ConfigField } from "@/lib/api";
import {
  type OverrideValues,
  defaultLabel,
  fieldValue,
  hasOverride,
} from "@/lib/config";
import { cn } from "@/lib/utils";

type ConfigFieldControlDensity = "comfortable" | "compact";

export function ConfigFieldValueEditor({
  field,
  overrides,
  onChange,
  onReset,
  controlId,
  controlLabel,
  resetLabel = "Reset field override",
  resetTitle,
  density = "comfortable",
  hideDisabledReset = true,
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
  hideDisabledReset?: boolean;
  className?: string;
}) {
  const value = fieldValue(field, overrides);
  const choices = field.choices;
  const isModified = hasOverride(overrides, field.key);
  const isLocked = field.locked;
  const isCompact = density === "compact";
  const isResetDisabled = !isModified || isLocked;

  return (
    <div
      className={cn(
        "grid items-center gap-2",
        isCompact ? "mt-1 grid-cols-[minmax(0,1fr)_40px]" : "mt-2 grid-cols-[minmax(0,1fr)_32px]",
        className,
      )}
    >
      {field.type === "bool" ? (
        <div
          className={cn(
            "flex items-center justify-between rounded-[10px] border border-line bg-black/25",
            isCompact ? "h-10 px-3" : "h-9 px-2.5",
          )}
        >
          <span className={cn("text-ink", isCompact ? "text-[13px]" : "text-sm")}>
            {value === "true" ? "Enabled" : "Off"}
          </span>
          <Switch
            id={controlId}
            aria-label={controlLabel ?? field.label}
            disabled={isLocked}
            checked={value === "true"}
            onCheckedChange={(checked) => onChange(field.key, String(checked))}
          />
        </div>
      ) : choices.length > 0 ? (
        <Select
          id={controlId}
          name={field.key}
          aria-label={controlLabel}
          autoComplete="off"
          value={value}
          disabled={isLocked}
          onChange={(event) => onChange(field.key, event.target.value)}
          className={isCompact ? "h-10 px-3 py-2 text-[13.5px]" : undefined}
        >
          {field.nullable && <option value="">None</option>}
          {choices.map((choice) => (
            <option key={String(choice)} value={String(choice)}>
              {String(choice)}
            </option>
          ))}
        </Select>
      ) : field.type === "int" || field.type === "float" ? (
        <Input
          id={controlId}
          name={field.key}
          aria-label={controlLabel}
          type="number"
          step={field.type === "float" ? "any" : "1"}
          autoComplete="off"
          value={value}
          disabled={isLocked}
          onChange={(event) => onChange(field.key, event.target.value)}
          className={isCompact ? "h-10 px-3 py-2 text-[13.5px]" : undefined}
        />
      ) : (
        <Input
          id={controlId}
          name={field.key}
          aria-label={controlLabel}
          autoComplete="off"
          value={value}
          disabled={isLocked}
          onChange={(event) => onChange(field.key, event.target.value)}
          className={isCompact ? "h-10 px-3 py-2 text-[13.5px]" : undefined}
        />
      )}
      <button
        type="button"
        aria-label={resetLabel}
        title={resetTitle ?? resetLabel}
        disabled={isResetDisabled}
        onClick={() => onReset(field.key)}
        className={cn(
          "flex items-center justify-center rounded-[10px] border border-line bg-white/[0.035] text-ink-faint transition hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:hover:bg-white/[0.035] disabled:hover:text-ink-faint",
          isCompact ? "h-10 w-10" : "h-8 w-8",
          hideDisabledReset && isResetDisabled
            ? "pointer-events-none opacity-0"
            : isResetDisabled && "opacity-45",
        )}
      >
        <RotateCcw className="h-3.5 w-3.5" aria-hidden />
      </button>
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
}: {
  field: ConfigField;
  overrides: OverrideValues;
  onChange: (key: string, value: string) => void;
  onReset: (key: string) => void;
  density?: ConfigFieldControlDensity;
  idPrefix?: string;
}) {
  const id = `${idPrefix}-${field.key}`;
  const isModified = hasOverride(overrides, field.key);
  const isLocked = field.locked;
  const isCompact = density === "compact";
  const defaultBadgeClassName = isCompact
    ? "rounded-none border-0 bg-transparent px-0 py-0 font-mono text-xs leading-4"
    : undefined;
  const overrideBadgeClassName = isCompact ? "px-1 py-0.5 text-xs" : undefined;

  return (
    <div
      className={cn(
        "grid w-full transition",
        isCompact ? "gap-1.5 py-1.5" : "gap-2 py-1.5",
        isModified && "border-l-2 border-violet/40 pl-2",
        isLocked &&
          "rounded-[10px] border-l-2 border-amber/55 bg-amber/[0.055] pl-2 pr-2 shadow-[inset_0_0_0_1px_rgba(255,209,102,0.04)]",
      )}
    >
      <label className={cn("grid", isCompact ? "gap-1" : "gap-1.5")} htmlFor={id}>
        <span className="flex min-w-0 items-start justify-between gap-2">
          <span className="min-w-0 text-[13px] font-semibold leading-5 text-ink [overflow-wrap:anywhere]">
            {field.label}
          </span>
          <span
            className={cn(
              "flex max-w-[62%] shrink-0 flex-wrap items-start justify-end",
              isCompact ? "gap-x-2 gap-y-1" : "gap-1",
            )}
          >
            <span
              className={cn(
                "max-w-full truncate font-mono text-xs text-ink-dim",
                isCompact
                  ? "leading-4"
                  : "rounded-[7px] border border-line bg-white/[0.035] px-1.5 py-1 leading-tight",
              )}
            >
              {field.flag}
            </span>
            {isCompact && (
              <span aria-hidden className="font-mono text-xs leading-4 text-ink-dim">
                |
              </span>
            )}
            <Badge className={defaultBadgeClassName}>default {defaultLabel(field)}</Badge>
          </span>
        </span>
        {(isModified || isLocked) && (
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
        resetTitle={`Reset ${field.label}`}
      />
    </div>
  );
}
