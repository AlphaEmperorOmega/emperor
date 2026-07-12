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
  useTrainingJobLifecycle,
  type TrainingJobLifecycle,
} from "@/features/workbench/state/training/use-training-job-lifecycle";
import {
  type TrainingConfiguration,
  type TrainingWorkspace,
} from "@/features/workbench/state/training/use-training-workspace-state";
import { type TrainingJob } from "@/lib/api";

const [TrainingWorkspaceProvider, useTrainingWorkspace] =
  createWorkbenchContext<TrainingWorkspace>("TrainingWorkspaceContext");
const [TrainingConfigurationProvider, useTrainingConfiguration] =
  createWorkbenchContext<TrainingConfiguration>(
    "TrainingConfigurationContext",
  );
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
  useTrainingWorkspace,
};

export function TrainingLifecycleProvider({
  onJobStarted,
  children,
}: {
  onJobStarted?: (logFolder: string) => void;
  children: (lifecycle: TrainingJobLifecycle) => ReactNode;
}) {
  const workbenchConnection = useWorkbenchConnection();
  const lifecycle = useTrainingJobLifecycle({
    enabled: isWorkbenchProtectedAccessReady(workbenchConnection),
    onJobStarted,
  });
  useRegisterWorkbenchConnectionReset(lifecycle.clearForConnectionChange);
  const projection = useMemo<ActiveTrainingJobProjection>(
    () => ({
      activeTrainingJob: lifecycle.job,
    }),
    [lifecycle.job],
  );

  return (
    <ActiveTrainingJobProvider value={projection}>
      {children(lifecycle)}
    </ActiveTrainingJobProvider>
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
