import {
  type ReactNode,
  useMemo,
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
import type { TrainingJob } from "@/lib/api/training-jobs";

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
  useTrainingPolling,
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
