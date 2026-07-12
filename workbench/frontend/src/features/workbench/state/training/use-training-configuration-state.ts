import { useMemo } from "react";
import {
  useTrainingDraftState,
  type TrainingDraftSeed,
  type TrainingDraftState,
} from "@/features/workbench/state/training/use-training-draft-state";
import { type ModelIdentity } from "@/lib/api";
import { type WorkbenchWorkspace } from "@/types/workbench";

export function useTrainingConfigurationState({
  activeWorkspace,
  models,
  protectedReadsEnabled,
  seed,
}: {
  activeWorkspace: WorkbenchWorkspace;
  models: ModelIdentity[];
  protectedReadsEnabled: boolean;
  seed: TrainingDraftSeed;
}) {
  const draft = useTrainingDraftState({
    activeWorkspace,
    models,
    protectedReadsEnabled,
    seed,
  });
  const modelSetup = draft.setup.model;
  const variantSetup = draft.setup.variants;
  const monitorSetup = draft.setup.monitors;
  const runtimeDefaults = draft.runtimeDefaults;
  const configuration = useMemo(
    () => ({
      selectedModelType: modelSetup.selectedType,
      selectedModel: modelSetup.selected,
      selectedPrimaryPreset: variantSetup.primaryPreset,
      selectedSnapshotIds: variantSetup.selectedSnapshotIds,
      selectedMonitors: monitorSetup.selected,
      configSections: runtimeDefaults.sections,
      fieldCount: runtimeDefaults.fieldCount,
      bulkOverrides: runtimeDefaults.active,
      inactiveLockedOverrideCount: runtimeDefaults.inactiveLockedCount,
      schemaLoading: draft.status.schemaLoading,
      includeSnapshot: variantSetup.includeSnapshot,
      excludeSnapshot: variantSetup.excludeSnapshot,
      updateOverride: runtimeDefaults.edit,
      clearOverride: runtimeDefaults.clear,
      resetOverrides: runtimeDefaults.reset,
    }),
    [
      draft.status.schemaLoading,
      modelSetup.selected,
      modelSetup.selectedType,
      monitorSetup.selected,
      runtimeDefaults,
      variantSetup.excludeSnapshot,
      variantSetup.includeSnapshot,
      variantSetup.primaryPreset,
      variantSetup.selectedSnapshotIds,
    ],
  );

  return useMemo(
    () => ({
      clearForConnectionChange: draft.clearForConnectionChange,
      configuration,
      draft,
    }),
    [configuration, draft],
  );
}

export type TrainingConfiguration = ReturnType<
  typeof useTrainingConfigurationState
>["configuration"];
export type { TrainingDraftState };
