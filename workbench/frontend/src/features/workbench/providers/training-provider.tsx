import {
  type ReactNode,
  useCallback,
  useMemo,
  useRef,
  useState,
} from "react";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  isWorkbenchProtectedAccessReady,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import {
  useActiveTrainingJobProgress,
  type ActiveTrainingJobProgress,
} from "@/features/workbench/state/training/use-training-job-controller";
import {
  type TrainingConfiguration,
  type TrainingWorkspace,
} from "@/features/workbench/state/training/use-training-workspace-state";
import { type TrainingJob } from "@/lib/api";

const noop = () => {};
const terminalJobStatuses = new Set(["completed", "failed", "cancelled"]);

function useActiveTrainingJobState({
  onJobStarted,
}: {
  onJobStarted?: (logFolder: string) => void;
} = {}) {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeTrainingJob, setActiveTrainingJob] = useState<TrainingJob>();
  const onJobChange = useCallback(
    (job: TrainingJob | undefined) => {
      setActiveTrainingJob((current) => {
        if (
          current &&
          job &&
          current.id === job.id &&
          terminalJobStatuses.has(current.status) &&
          !terminalJobStatuses.has(job.status)
        ) {
          return current;
        }
        return job;
      });
      if (job?.logFolder) {
        onJobStarted?.(job.logFolder);
      }
    },
    [onJobStarted],
  );
  const clearActiveTrainingJob = useCallback(() => {
    setActiveJobId(null);
    setActiveTrainingJob(undefined);
  }, []);

  return useMemo(
    () => ({
      activeJobId,
      setActiveJobId,
      activeTrainingJob,
      onJobChange,
      clearActiveTrainingJob,
    }),
    [
      activeJobId,
      activeTrainingJob,
      clearActiveTrainingJob,
      onJobChange,
    ],
  );
}

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

type TrainingLifecycleImplementation = Pick<
  ReturnType<typeof useActiveTrainingJobState>,
  "activeTrainingJob" | "setActiveJobId" | "onJobChange"
> &
  Pick<ActiveTrainingJobProgress, "progressError"> & {
    clearForConnectionChange: () => void;
    registerDraftConnectionClear: (clear: () => void) => () => void;
  };
const [TrainingLifecycleImplementationProvider, useTrainingLifecycleImplementation] =
  createWorkbenchContext<TrainingLifecycleImplementation>(
    "TrainingLifecycleImplementationContext",
  );

export {
  useActiveTrainingJob,
  useTrainingConfiguration,
  useTrainingLifecycleImplementation,
  useTrainingWorkspace,
};

export function TrainingLifecycleProvider({
  onJobStarted,
  children,
}: {
  onJobStarted?: (logFolder: string) => void;
  children: ReactNode;
}) {
  const lifecycle = useActiveTrainingJobState({ onJobStarted });
  const workbenchConnection = useWorkbenchConnection();
  const progress = useActiveTrainingJobProgress({
    activeJobId: lifecycle.activeJobId,
    onJobChange: lifecycle.onJobChange,
    enabled: isWorkbenchProtectedAccessReady(workbenchConnection),
  });
  const draftConnectionClearRef = useRef<() => void>(noop);
  const registerDraftConnectionClear = useCallback((clear: () => void) => {
    draftConnectionClearRef.current = clear;
    return () => {
      if (draftConnectionClearRef.current === clear) {
        draftConnectionClearRef.current = noop;
      }
    };
  }, []);
  const clearActiveTrainingJob = lifecycle.clearActiveTrainingJob;
  const clearForConnectionChange = useCallback(() => {
    draftConnectionClearRef.current();
    clearActiveTrainingJob();
  }, [clearActiveTrainingJob]);
  const projection = useMemo<ActiveTrainingJobProjection>(
    () => ({
      activeTrainingJob: lifecycle.activeTrainingJob,
    }),
    [lifecycle.activeTrainingJob],
  );
  const implementation = useMemo<TrainingLifecycleImplementation>(
    () => ({
      activeTrainingJob: lifecycle.activeTrainingJob,
      setActiveJobId: lifecycle.setActiveJobId,
      onJobChange: lifecycle.onJobChange,
      progressError: progress.progressError,
      clearForConnectionChange,
      registerDraftConnectionClear,
    }),
    [
      clearForConnectionChange,
      lifecycle.activeTrainingJob,
      lifecycle.onJobChange,
      lifecycle.setActiveJobId,
      progress.progressError,
      registerDraftConnectionClear,
    ],
  );

  return (
    <TrainingLifecycleImplementationProvider value={implementation}>
      <ActiveTrainingJobProvider value={projection}>
        {children}
      </ActiveTrainingJobProvider>
    </TrainingLifecycleImplementationProvider>
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
