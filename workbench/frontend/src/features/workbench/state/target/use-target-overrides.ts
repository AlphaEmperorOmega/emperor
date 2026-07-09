import { useCallback, useState } from "react";
import { type OverrideValues } from "@/lib/config";

type TargetOverridesInitialState = {
  selectedModel?: string;
  selectedPreset?: string;
  overrides?: OverrideValues;
  presetOverrides?: OverrideValues;
};

export function useTargetOverridesState(
  initialState: TargetOverridesInitialState = {},
) {
  const [selectedModel, setSelectedModel] = useState(
    initialState.selectedModel ?? "",
  );
  const [selectedPreset, setSelectedPreset] = useState(
    initialState.selectedPreset ?? "",
  );
  const [presetOverrides, setPresetOverrides] = useState<OverrideValues>(
    initialState.presetOverrides ?? initialState.overrides ?? {},
  );

  const selectModel = useCallback((model: string) => {
    setSelectedModel(model);
    setSelectedPreset("");
    setPresetOverrides({});
  }, []);

  const selectPreset = useCallback((preset: string) => {
    setSelectedPreset(preset);
  }, []);

  const updatePresetOverride = useCallback((key: string, value: string) => {
    setPresetOverrides((current) => ({ ...current, [key]: value }));
  }, []);

  const clearPresetOverride = useCallback((key: string) => {
    setPresetOverrides((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  }, []);

  const clearPresetOverrides = useCallback(() => {
    setPresetOverrides({});
  }, []);

  return {
    selectedModel,
    setSelectedModel,
    selectedPreset,
    setSelectedPreset,
    presetOverrides,
    setPresetOverrides,
    overrides: presetOverrides,
    setOverrides: setPresetOverrides,
    selectModel,
    selectPreset,
    updatePresetOverride,
    clearPresetOverride,
    clearPresetOverrides,
    updateOverride: updatePresetOverride,
    clearOverride: clearPresetOverride,
    clearOverrides: clearPresetOverrides,
  };
}
