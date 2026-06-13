import { useMemo } from "react";
import { type MultiSelectDropdownOption } from "@/features/viewer/components/screen/multi-select-dropdown";
import { type MonitorOption, type Preset } from "@/lib/api";
import {
  modelNameForId,
  modelsForType,
  modelTypeOptions as createModelTypeOptions,
} from "@/lib/selection";

type SelectOption = {
  value: string;
  label: string;
};

export function buildTrainingModelTypeOptions(models: string[]): SelectOption[] {
  return createModelTypeOptions(models);
}

export function buildTrainingModelOptions(
  models: string[],
  selectedModelType = "",
): SelectOption[] {
  return modelsForType(models, selectedModelType).map((model) => ({
    value: model,
    label: modelNameForId(model),
  }));
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
  selectedModelType,
  presets,
  monitorOptions,
}: {
  models: string[];
  selectedModelType: string;
  presets: Preset[];
  monitorOptions: MonitorOption[];
}) {
  const modelTypeOptions = useMemo(
    () => buildTrainingModelTypeOptions(models),
    [models],
  );
  const modelOptions = useMemo(
    () => buildTrainingModelOptions(models, selectedModelType),
    [models, selectedModelType],
  );
  const presetOptions = useMemo(() => buildTrainingPresetOptions(presets), [presets]);
  const trainingMonitorOptions = useMemo(
    () => buildTrainingMonitorOptions(monitorOptions),
    [monitorOptions],
  );

  return {
    modelTypeOptions,
    modelOptions,
    presetOptions,
    trainingMonitorOptions,
  };
}
