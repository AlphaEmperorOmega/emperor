"use client";

import { type ReactNode, useRef } from "react";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import { type TrainingWorkspace } from "@/features/workbench/state/training/use-training-workspace-state";
import {
  type TrainingConfiguration,
  type TrainingDraftState,
} from "@/features/workbench/state/training/use-training-configuration-state";

const [TrainingWorkspaceProvider, useTrainingWorkspace] =
  createWorkbenchContext<TrainingWorkspace>("TrainingWorkspaceContext");
const [TrainingConfigurationProvider, useTrainingConfiguration] =
  createWorkbenchContext<TrainingConfiguration>(
    "TrainingConfigurationContext",
  );
const [TrainingDraftProvider, useTrainingDraft] =
  createWorkbenchContext<TrainingDraftState>("TrainingDraftContext");

export {
  useTrainingConfiguration,
  useTrainingDraft,
  useTrainingWorkspace,
};

function useShallowStableValue<Target extends Record<PropertyKey, unknown>>(
  value: Target,
) {
  const stableValue = useRef(value);
  const keys = Reflect.ownKeys(value);
  if (
    keys.length !== Reflect.ownKeys(stableValue.current).length ||
    keys.some((key) => !Object.is(value[key], stableValue.current[key]))
  ) {
    stableValue.current = value;
  }
  return stableValue.current;
}

export function TrainingWorkspaceContextProvider({
  value,
  children,
}: {
  value: TrainingWorkspace;
  children: ReactNode;
}) {
  const stableValue = useShallowStableValue(value);
  return (
    <TrainingWorkspaceProvider value={stableValue}>
      {children}
    </TrainingWorkspaceProvider>
  );
}

export function TrainingConfigurationContextProvider({
  value,
  children,
}: {
  value: TrainingConfiguration;
  children: ReactNode;
}) {
  const stableValue = useShallowStableValue(value);
  return (
    <TrainingConfigurationProvider value={stableValue}>
      {children}
    </TrainingConfigurationProvider>
  );
}

export function TrainingDraftContextProvider({
  value,
  children,
}: {
  value: TrainingDraftState;
  children: ReactNode;
}) {
  const stableValue = useShallowStableValue(value);
  return (
    <TrainingDraftProvider value={stableValue}>{children}</TrainingDraftProvider>
  );
}
