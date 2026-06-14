import { useCallback, useState } from "react";
import { type OverrideValues } from "@/lib/config";

type TargetOverridesInitialState = {
  selectedModel?: string;
  selectedPreset?: string;
  overrides?: OverrideValues;
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
  const [overrides, setOverrides] = useState<OverrideValues>(
    initialState.overrides ?? {},
  );

  const selectModel = useCallback((model: string) => {
    setSelectedModel(model);
    setSelectedPreset("");
    setOverrides({});
  }, []);

  const selectPreset = useCallback((preset: string) => {
    setSelectedPreset(preset);
    setOverrides({});
  }, []);

  const updateOverride = useCallback((key: string, value: string) => {
    setOverrides((current) => ({ ...current, [key]: value }));
  }, []);

  const clearOverride = useCallback((key: string) => {
    setOverrides((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  }, []);

  const clearOverrides = useCallback(() => {
    setOverrides({});
  }, []);

  return {
    selectedModel,
    setSelectedModel,
    selectedPreset,
    setSelectedPreset,
    overrides,
    setOverrides,
    selectModel,
    selectPreset,
    updateOverride,
    clearOverride,
    clearOverrides,
  };
}
