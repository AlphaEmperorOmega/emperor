import {
  type ReactNode,
  useMemo,
  useRef,
} from "react";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  isWorkbenchProtectedAccessReady,
  useRegisterWorkbenchConnectionReset,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import {
  useTrainingJobPolling,
  type TrainingJobPolling,
} from "@/features/workbench/state/training/use-training-job-polling";
import { type TrainingWorkspace } from "@/features/workbench/state/training/use-training-workspace-state";
import {
  type TrainingConfiguration,
  type TrainingDraftState,
} from "@/features/workbench/state/training/use-training-configuration-state";
import { type TrainingJob } from "@/lib/api";

const [TrainingWorkspaceProvider, useTrainingWorkspace] =
  createWorkbenchContext<TrainingWorkspace>("TrainingWorkspaceContext");
const [TrainingConfigurationProvider, useTrainingConfiguration] =
  createWorkbenchContext<TrainingConfiguration>(
    "TrainingConfigurationContext",
  );
const [TrainingDraftProvider, useTrainingDraft] =
  createWorkbenchContext<TrainingDraftState>("TrainingDraftContext");
const [TrainingPollingContextProvider, useTrainingPolling] =
  createWorkbenchContext<TrainingJobPolling>("TrainingPollingContext");
type ActiveTrainingJobProjection = Readonly<{
  activeTrainingJob: TrainingJob | undefined;
}>;
const [ActiveTrainingJobProvider, useActiveTrainingJob] =
  createWorkbenchContext<ActiveTrainingJobProjection>(
    "ActiveTrainingJobContext",
  );

export {
  useActiveTrainingJob,
  useTrainingConfiguration,
  useTrainingDraft,
  useTrainingPolling,
  useTrainingWorkspace,
};

export function TrainingLifecycleProvider({
  onJobStarted,
  children,
}: {
  onJobStarted?: (logFolder: string) => void;
  children: ReactNode;
}) {
  const workbenchConnection = useWorkbenchConnection();
  const polling = useTrainingJobPolling({
    enabled: isWorkbenchProtectedAccessReady(workbenchConnection),
    onJobStarted,
  });
  useRegisterWorkbenchConnectionReset(polling.clearForConnectionChange);
  const projection = useMemo<ActiveTrainingJobProjection>(
    () => ({
      activeTrainingJob: polling.job,
    }),
    [polling.job],
  );

  return (
    <TrainingPollingContextProvider value={polling}>
      <ActiveTrainingJobProvider value={projection}>
        {children}
      </ActiveTrainingJobProvider>
    </TrainingPollingContextProvider>
  );
}

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
