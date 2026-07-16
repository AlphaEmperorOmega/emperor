"use client";

import { type ReactNode } from "react";
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

export function TrainingWorkspaceContextProvider({
  value,
  children,
}: {
  value: TrainingWorkspace;
  children: ReactNode;
}) {
  return (
    <TrainingWorkspaceProvider value={value}>
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
  return (
    <TrainingConfigurationProvider value={value}>
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
  return (
    <TrainingDraftProvider value={value}>{children}</TrainingDraftProvider>
  );
}
