import { ChevronRight, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { type ConfigField } from "@/lib/api";
import { cn } from "@/lib/utils";

export type OverrideValues = Record<string, string>;

function fieldValue(field: ConfigField, overrides: OverrideValues) {
  const value = overrides[field.key] ?? field.default;
  return value === null || value === undefined ? "" : String(value);
}

function configSectionId(title: string) {
  return `config-section-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
}

export function ConfigSectionAccordion({
  title,
  fields,
  isOpen,
  overrides,
  onToggle,
  onFieldChange,
  onFieldReset,
}: {
  title: string;
  fields: ConfigField[];
  isOpen: boolean;
  overrides: OverrideValues;
  onToggle: () => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const controlsId = configSectionId(title);
  const modifiedCount = fields.filter((field) =>
    Object.prototype.hasOwnProperty.call(overrides, field.key),
  ).length;

  return (
    <section
      className={cn(
        "overflow-hidden rounded-md border bg-panel shadow-panel transition",
        modifiedCount > 0 ? "border-accent-line" : "border-border",
      )}
    >
      <h2>
        <button
          type="button"
          aria-expanded={isOpen}
          aria-controls={controlsId}
          onClick={onToggle}
          className="flex min-h-11 w-full items-center gap-2 px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.08em] text-ink transition hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
        >
          <ChevronRight
            className={cn("h-4 w-4 shrink-0 text-muted transition-transform", isOpen && "rotate-90")}
            aria-hidden
          />
          <span className="min-w-0 flex-1 truncate">{title}</span>
          <span className="flex shrink-0 items-center gap-1">
            {modifiedCount > 0 && <Badge className="border-accent-line bg-accent-soft text-accent">{modifiedCount} set</Badge>}
            <Badge>{fields.length} fields</Badge>
          </span>
        </button>
      </h2>
      {isOpen && (
        <div id={controlsId} className="grid gap-2 border-t border-border bg-[#fbfcfb] p-2">
          {fields.map((field) => (
            <ConfigControl
              key={field.key}
              field={field}
              overrides={overrides}
              onChange={onFieldChange}
              onReset={onFieldReset}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function ConfigControl({
  field,
  overrides,
  onChange,
  onReset,
}: {
  field: ConfigField;
  overrides: OverrideValues;
  onChange: (key: string, value: string) => void;
  onReset: (key: string) => void;
}) {
  const value = fieldValue(field, overrides);
  const choices = field.choices.length > 0 ? field.choices : field.searchChoices;
  const id = `field-${field.key}`;
  const isModified = Object.prototype.hasOwnProperty.call(overrides, field.key);
  const defaultLabel =
    field.default === null || field.default === undefined ? "None" : String(field.default);

  return (
    <div
      className={cn(
        "rounded-md border bg-panel p-2.5 transition",
        isModified ? "border-accent-line shadow-[inset_3px_0_0_#15705f]" : "border-subtle",
      )}
    >
      <label className="grid gap-1.5" htmlFor={id}>
        <span className="text-[13px] font-semibold leading-5 text-ink [overflow-wrap:anywhere]">
          {field.label}
        </span>
        <span className="flex min-w-0 flex-wrap items-center gap-1.5 text-xs font-medium text-muted">
          <span className="max-w-full min-w-0 rounded border border-border bg-surface px-1.5 py-1 font-mono text-[10px] leading-tight [overflow-wrap:anywhere]">
            {field.flag}
          </span>
          <Badge>default {defaultLabel}</Badge>
          {isModified && (
            <Badge className="border-accent-line bg-accent-soft text-accent">override</Badge>
          )}
        </span>
      </label>
      <div className="mt-2 grid grid-cols-[minmax(0,1fr)_32px] items-center gap-2">
        {field.type === "bool" ? (
          <div className="flex h-9 items-center justify-between rounded-md border border-border bg-surface px-2.5">
            <span className="text-sm text-ink">{value === "true" ? "Enabled" : "Off"}</span>
            <Switch
              id={id}
              aria-label={field.label}
              checked={value === "true"}
              onCheckedChange={(checked) => onChange(field.key, String(checked))}
            />
          </div>
        ) : choices.length > 0 ? (
          <Select
            id={id}
            name={field.key}
            autoComplete="off"
            value={value}
            onChange={(event) => onChange(field.key, event.target.value)}
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
            id={id}
            name={field.key}
            type="number"
            step={field.type === "float" ? "any" : "1"}
            autoComplete="off"
            value={value}
            onChange={(event) => onChange(field.key, event.target.value)}
          />
        ) : (
          <Input
            id={id}
            name={field.key}
            autoComplete="off"
            value={value}
            onChange={(event) => onChange(field.key, event.target.value)}
          />
        )}
        <button
          type="button"
          aria-label="Reset field override"
          title={`Reset ${field.label}`}
          disabled={!isModified}
          onClick={() => onReset(field.key)}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-md border border-border bg-panel text-muted transition hover:bg-surface hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-0",
            !isModified && "pointer-events-none",
          )}
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden />
        </button>
      </div>
    </div>
  );
}
