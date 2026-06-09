import { Maximize2, RotateCcw, SlidersHorizontal, Target } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrainingTargetDatasetPanel } from "@/features/viewer/components/training/training-target-dataset-panel";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { type Dataset, type Preset } from "@/lib/api";
import {
  type ConfigSection,
  type OverrideValues,
} from "@/lib/config";

export function TrainingExperimentSetup({
  models,
  presets,
  datasetOptions,
  configSections,
  selectedModel,
  selectedPreset,
  selectedDatasets,
  overrides,
  onSelectModel,
  onSelectPreset,
  onToggleDataset,
  onSelectAllDatasets,
  onSelectFirstDataset,
  onResetOverrides,
  onOpenFullConfig,
  canOpenFullConfig,
}: {
  models: string[];
  presets: Preset[];
  datasetOptions: Dataset[];
  configSections: ConfigSection[];
  selectedModel: string;
  selectedPreset: string;
  selectedDatasets: string[];
  overrides: OverrideValues;
  onSelectModel: (model: string) => void;
  onSelectPreset: (preset: string) => void;
  onToggleDataset: (dataset: string) => void;
  onSelectAllDatasets: () => void;
  onSelectFirstDataset: () => void;
  onResetOverrides: () => void;
  onOpenFullConfig: () => void;
  canOpenFullConfig: boolean;
}) {
  const modelOptions = models.map((model) => ({ value: model, label: model }));
  const presetOptions = presets.map((preset) => ({
    value: preset.name,
    label: preset.name,
  }));
  const overrideCount = Object.keys(overrides).length;
  const fieldCount = configSections.reduce(
    (total, section) => total + section.fields.length,
    0,
  );

  return (
    <section className="edge grid gap-3 rounded-card p-3">
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          <Target className="h-[15px] w-[15px] text-violet" aria-hidden />
          Experiment Setup
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-1.5">
          <Badge>{fieldCount} fields</Badge>
          <Badge className={overrideCount > 0 ? "border-violet/30 bg-violet/15 text-violet" : undefined}>
            {overrideCount} overrides
          </Badge>
        </div>
      </div>

      <div className="grid gap-3 xl:grid-cols-[minmax(220px,0.72fr)_minmax(340px,1fr)] xl:items-stretch">
        <TrainingTargetDatasetPanel
          modelOptions={modelOptions}
          presetOptions={presetOptions}
          selectedModel={selectedModel}
          selectedPreset={selectedPreset}
          datasetOptions={datasetOptions}
          selectedDatasets={selectedDatasets}
          onSelectModel={onSelectModel}
          onSelectPreset={onSelectPreset}
          onToggleDataset={onToggleDataset}
          onSelectAllDatasets={onSelectAllDatasets}
          onSelectFirstDataset={onSelectFirstDataset}
        />

        <div className="grid min-w-0 content-start gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
              <SlidersHorizontal className="h-[15px] w-[15px] text-violet" aria-hidden />
              Overrides
            </div>
            <div className="flex shrink-0 flex-wrap items-center gap-1.5">
              <Badge
                className={
                  overrideCount > 0
                    ? "border-violet/30 bg-violet/15 text-violet"
                    : undefined
                }
              >
                {overrideCount} overrides
              </Badge>
              <Button
                variant="ghost"
                onClick={onResetOverrides}
                disabled={overrideCount === 0}
                className="h-8 border border-line bg-white/[0.025] px-2.5 text-xs"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden />
                Reset
              </Button>
            </div>
          </div>

          <div className="grid gap-2 rounded-[10px] border border-line bg-white/[0.018] p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
              <span className="font-medium text-ink">Config fields</span>
              <Badge>{fieldCount} fields</Badge>
            </div>
            <Button
              variant="primary"
              onClick={onOpenFullConfig}
              disabled={!canOpenFullConfig}
              className="h-10 w-full text-[13.5px]"
            >
              <Maximize2 className="h-4 w-4" aria-hidden />
              Open Full Config
            </Button>
          </div>

          {configSections.length === 0 && (
            <InlineStatus compact>
              Select a model and preset to load config fields
            </InlineStatus>
          )}
        </div>
      </div>
    </section>
  );
}
