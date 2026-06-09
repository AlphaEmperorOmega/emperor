import { useMemo } from "react";
import { type MultiSelectDropdownOption } from "@/features/viewer/components/screen/multi-select-dropdown";
import { type MonitorOption, type Preset } from "@/lib/api";

type SelectOption = {
  value: string;
  label: string;
};

export function buildTrainingModelOptions(models: string[]): SelectOption[] {
  return models.map((model) => ({ value: model, label: model }));
}

export function buildTrainingPresetOptions(presets: Preset[]): SelectOption[] {
  return presets.map((preset) => ({ value: preset.name, label: preset.name }));
}

export function buildTrainingMonitorOptions(
  monitorOptions: MonitorOption[],
): MultiSelectDropdownOption[] {
  return monitorOptions.map((monitor) => ({
    value: monitor.name,
    label: monitor.label,
    description: monitor.description,
    meta:
      monitor.kinds.length > 0 ? (
        <span>{monitor.kinds.join(" / ")}</span>
      ) : undefined,
  }));
}

export function useTrainingPanelOptions({
  models,
  presets,
  monitorOptions,
}: {
  models: string[];
  presets: Preset[];
  monitorOptions: MonitorOption[];
}) {
  const modelOptions = useMemo(() => buildTrainingModelOptions(models), [models]);
  const presetOptions = useMemo(() => buildTrainingPresetOptions(presets), [presets]);
  const trainingMonitorOptions = useMemo(
    () => buildTrainingMonitorOptions(monitorOptions),
    [monitorOptions],
  );

  return {
    modelOptions,
    presetOptions,
    trainingMonitorOptions,
  };
}
