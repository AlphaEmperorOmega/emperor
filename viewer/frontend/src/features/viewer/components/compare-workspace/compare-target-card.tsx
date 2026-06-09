import {
  Activity,
  ArrowRight,
  Database,
  Network,
  Settings2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EdgeCard } from "@/components/ui/edge-card";
import { Select } from "@/components/ui/select";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import { errorMessage } from "@/lib/utils";
import {
  formatInteger,
  monitorSummary,
  type CompareEntry,
  type CompareEntryData,
  type CompareModelOption,
} from "./derive";

export function CompareTargetCard({
  entryData,
  index,
  modelOptions,
  baselineParameterCount,
  canRemove,
  onRemove,
  onUpdate,
  onApply,
}: {
  entryData: CompareEntryData;
  index: number;
  modelOptions: CompareModelOption[];
  baselineParameterCount: number | undefined;
  canRemove: boolean;
  onRemove: (id: string) => void;
  onUpdate: (id: string, patch: Partial<CompareEntry>) => void;
  onApply: (entry: CompareEntry) => void;
}) {
  const { entry } = entryData;
  const parameterDelta =
    entryData.inspection && baselineParameterCount !== undefined
      ? entryData.inspection.parameterCount - baselineParameterCount
      : undefined;

  return (
    <EdgeCard className="min-w-0 rounded-card p-4">
      <div className="grid gap-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-faint">
              Target {index + 1}
            </div>
            <div className="mt-1 truncate font-mono text-sm font-semibold text-ink">
              {entry.model || "No model"}
            </div>
          </div>
          <Button
            variant="ghost"
            className="h-8 w-8 px-0"
            aria-label={`Remove comparison target ${index + 1}`}
            onClick={() => onRemove(entry.id)}
            disabled={!canRemove}
          >
            <X className="h-4 w-4" aria-hidden />
          </Button>
        </div>

        <label className="grid min-w-0 gap-1.5">
          <span className="text-xs font-semibold text-ink-dim">Model</span>
          <Select
            className="min-w-0"
            value={entry.model}
            onChange={(event) =>
              onUpdate(entry.id, {
                model: event.target.value,
                preset: "",
              })
            }
          >
            {modelOptions.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </Select>
        </label>

        <label className="grid min-w-0 gap-1.5">
          <span className="text-xs font-semibold text-ink-dim">Preset</span>
          <Select
            className="min-w-0"
            value={entry.preset}
            onChange={(event) => onUpdate(entry.id, { preset: event.target.value })}
            disabled={entryData.presets.length === 0}
          >
            {entryData.presets.length === 0 ? (
              <option value="">No presets</option>
            ) : (
              entryData.presets.map((preset) => (
                <option key={preset.name} value={preset.name}>
                  {preset.name}
                </option>
              ))
            )}
          </Select>
        </label>

        {Boolean(entryData.error) && (
          <InlineStatus tone="danger" compact role="alert">
            {errorMessage(entryData.error)}
          </InlineStatus>
        )}
        {!entryData.error && entryData.isLoading && (
          <InlineStatus busy compact>
            {viewerStatusCopy.loading.targetData}
          </InlineStatus>
        )}

        <div className="grid min-w-0 grid-cols-2 gap-2">
          <Badge variant="violet" className="min-w-0 truncate whitespace-nowrap">
            <Network className="mr-1 h-3.5 w-3.5" aria-hidden />
            {formatInteger(entryData.inspection?.parameterCount)} params
          </Badge>
          <Badge className="min-w-0 truncate whitespace-nowrap">
            <Activity className="mr-1 h-3.5 w-3.5" aria-hidden />
            {formatInteger(entryData.inspection?.nodes.length)} nodes
          </Badge>
          <Badge className="min-w-0 truncate whitespace-nowrap">
            <Settings2 className="mr-1 h-3.5 w-3.5" aria-hidden />
            {formatInteger(entryData.fields.length)} fields
          </Badge>
          <Badge className="min-w-0 truncate whitespace-nowrap">
            <Database className="mr-1 h-3.5 w-3.5" aria-hidden />
            {entryData.dataset || "No dataset"}
          </Badge>
        </div>

        {parameterDelta !== undefined && index > 0 && (
          <div
            className={
              parameterDelta === 0
                ? "rounded-[10px] border border-line-soft bg-white/[0.025] px-3 py-2 text-xs font-semibold text-ink-dim"
                : "rounded-[10px] border border-amber/35 bg-amber/[0.08] px-3 py-2 text-xs font-semibold text-amber"
            }
          >
            Delta from target 1: {parameterDelta > 0 ? "+" : ""}
            {formatInteger(parameterDelta)}
          </div>
        )}

        <div className="grid min-w-0 gap-1.5 rounded-[10px] border border-line-soft bg-black/16 p-3 text-xs">
          <div className="grid min-w-0 grid-cols-[76px_minmax(0,1fr)] gap-3">
            <span className="text-ink-faint">Monitors</span>
            <span
              className="min-w-0 truncate text-right font-semibold text-ink-dim"
              title={entryData.monitors.map((monitor) => monitor.label).join(", ")}
            >
              {monitorSummary(entryData.monitors)}
            </span>
          </div>
          <div className="grid min-w-0 grid-cols-[76px_minmax(0,1fr)] gap-3">
            <span className="text-ink-faint">Datasets</span>
            <span className="min-w-0 text-right font-mono text-ink-dim">
              {formatInteger(entryData.datasets.length)}
            </span>
          </div>
          <div className="grid min-w-0 grid-cols-[76px_minmax(0,1fr)] gap-3">
            <span className="text-ink-faint">Locked</span>
            <span className="min-w-0 text-right font-mono text-ink-dim">
              {formatInteger(entryData.fields.filter((field) => field.locked).length)}
            </span>
          </div>
        </div>

        <Button
          variant="secondary"
          onClick={() => onApply(entry)}
          disabled={!entry.model || !entry.preset}
        >
          <ArrowRight className="h-4 w-4" aria-hidden />
          Use as Target
        </Button>
      </div>
    </EdgeCard>
  );
}
