import { useCallback, useState } from "react";
import { type OverrideValues } from "@/lib/config";

export function useTargetOverridesState() {
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedPreset, setSelectedPreset] = useState("");
  const [overrides, setOverrides] = useState<OverrideValues>({});

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
