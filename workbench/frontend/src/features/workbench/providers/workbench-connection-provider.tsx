import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  type Capabilities,
  fetchCapabilities,
  fetchHealth,
  fetchModels,
  isUnauthorizedApiError,
} from "@/lib/api";
import {
  clearWorkbenchAuthenticationObservation,
  isWorkbenchAuthenticationVerified,
  loadWorkbenchConnectionRuntime,
  observeWorkbenchAuthMode,
  workbenchConnectionRuntimeSnapshot,
  type WorkbenchConnectionRuntimeSnapshot,
} from "@/lib/api/_connection-runtime";
import { workbenchQueryKeys } from "@/lib/query-keys";
import { type WorkbenchConnectionActionEnvironment } from "@/features/workbench/providers/_workbench-connection-actions";

const loadWorkbenchConnectionActions = () =>
  import("@/features/workbench/providers/_workbench-connection-actions");

export type WorkbenchFeatureCapabilities = Omit<Capabilities, "authMode">;

export const DEFAULT_WORKBENCH_CAPABILITIES: WorkbenchFeatureCapabilities = {
  trainingEnabled: true,
  trainingCancellationCapability: "unsupported",
  logDeletionEnabled: true,
  configSnapshotsEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: true,
  maxUploadSize: null,
  maxActiveTrainingJobs: 2,
  trainingJobMemoryLimitBytes: 16 * 1024 ** 3,
  trainingJobCpuLimit: 8,
  trainingJobProcessLimit: 512,
};

export type WorkbenchAuthenticationState =
  | "capability-checking"
  | "capability-failed"
  | "disabled"
  | "unauthenticated"
  | "checking"
  | "authenticated"
  | "rejected"
  | "protected-read-failed";

export type WorkbenchConnectionActionResult =
  | { ok: true }
  | { ok: false; message: string };

export type WorkbenchConnection = Readonly<{
  connection: Readonly<{
    apiBaseUrl: string;
    isOnline: boolean;
    isChanging: boolean;
    configurationError: string | null;
  }>;
  authentication: Readonly<{
    mode: "unknown" | Capabilities["authMode"];
    state: WorkbenchAuthenticationState;
    hasToken: boolean;
  }>;
  storage: Readonly<{
    apiBaseUrl: WorkbenchConnectionRuntimeSnapshot["storage"]["apiBaseUrl"];
    sessionToken: WorkbenchConnectionRuntimeSnapshot["storage"]["sessionToken"];
    message: string | null;
  }>;
  actions: Readonly<{
    useApiBaseUrl: (url: string) => Promise<WorkbenchConnectionActionResult>;
    resetApiBaseUrl: () => Promise<WorkbenchConnectionActionResult>;
    signIn: (token: string) => Promise<WorkbenchConnectionActionResult>;
    logout: () => Promise<WorkbenchConnectionActionResult>;
    retry: () => Promise<void>;
  }>;
}>;

export type WorkbenchCapabilities = Readonly<{
  capabilities: WorkbenchFeatureCapabilities;
}>;

function featureCapabilities({
  trainingEnabled,
  trainingCancellationCapability,
  logDeletionEnabled,
  configSnapshotsEnabled,
  historicalLogsEnabled,
  liveMonitorDataEnabled,
  historicalMonitorDataEnabled,
  uploadsEnabled,
  maxUploadSize,
  maxActiveTrainingJobs,
  trainingJobMemoryLimitBytes,
  trainingJobCpuLimit,
  trainingJobProcessLimit,
}: Capabilities): WorkbenchFeatureCapabilities {
  return {
    trainingEnabled,
    trainingCancellationCapability,
    logDeletionEnabled,
    configSnapshotsEnabled,
    historicalLogsEnabled,
    liveMonitorDataEnabled,
    historicalMonitorDataEnabled,
    uploadsEnabled,
    maxUploadSize,
    maxActiveTrainingJobs,
    trainingJobMemoryLimitBytes,
    trainingJobCpuLimit,
    trainingJobProcessLimit,
  };
}

export function isWorkbenchProtectedAccessReady(
  workbenchConnection: WorkbenchConnection,
) {
  return (
    !workbenchConnection.connection.isChanging &&
    (workbenchConnection.authentication.state === "disabled" ||
      workbenchConnection.authentication.state === "authenticated")
  );
}

type ConnectionReset = () => void;
type RegisterConnectionReset = (reset: ConnectionReset) => () => void;

const [WorkbenchConnectionContextProvider, useWorkbenchConnection] =
  createWorkbenchContext<WorkbenchConnection>("WorkbenchConnectionContext");
const [WorkbenchCapabilitiesContextProvider, useWorkbenchCapabilities] =
  createWorkbenchContext<WorkbenchCapabilities>("WorkbenchCapabilitiesContext");
const [ConnectionResetRegistryProvider, useConnectionResetRegistry] =
  createWorkbenchContext<RegisterConnectionReset>(
    "WorkbenchConnectionResetRegistryContext",
  );

export { useWorkbenchCapabilities, useWorkbenchConnection };

/** Composition-only registration; callers consume only the two public projections. */
export function useRegisterWorkbenchConnectionReset(reset: ConnectionReset) {
  const register = useConnectionResetRegistry();
  useEffect(() => register(reset), [register, reset]);
}

function authenticationState({
  runtime,
  capabilitiesQuery,
  modelsQuery,
  authenticationVerified,
}: {
  runtime: WorkbenchConnectionRuntimeSnapshot;
  capabilitiesQuery: ReturnType<typeof useQuery<Capabilities>>;
  modelsQuery: ReturnType<typeof useQuery<Awaited<ReturnType<typeof fetchModels>>>>;
  authenticationVerified: boolean;
}) {
  if (capabilitiesQuery.isPending) {
    return {
      mode: "unknown" as const,
      state: "capability-checking" as const,
    };
  }
  if (capabilitiesQuery.isError) {
    return {
      mode: "unknown" as const,
      state: "capability-failed" as const,
    };
  }
  if (capabilitiesQuery.data.authMode === "none") {
    return { mode: "none" as const, state: "disabled" as const };
  }
  if (!runtime.hasAuthToken) {
    return {
      mode: "bearer" as const,
      state: "unauthenticated" as const,
    };
  }
  if (modelsQuery.isPending || modelsQuery.isFetching) {
    return {
      mode: "bearer" as const,
      state: "checking" as const,
    };
  }
  if (modelsQuery.isError) {
    return {
      mode: "bearer" as const,
      state: isUnauthorizedApiError(modelsQuery.error)
        ? ("rejected" as const)
        : ("protected-read-failed" as const),
    };
  }
  if (!authenticationVerified) {
    return {
      mode: "bearer" as const,
      state: "checking" as const,
    };
  }
  return {
    mode: "bearer" as const,
    state: "authenticated" as const,
  };
}

export function WorkbenchConnectionProvider({
  children,
}: {
  children: ReactNode;
}) {
  const queryClient = useQueryClient();
  const [runtime, setRuntime] = useState(loadWorkbenchConnectionRuntime);
  const [isChanging, setIsChanging] = useState(false);
  const resetRegistryRef = useRef(new Set<ConnectionReset>());
  const transitionQueueRef = useRef<Promise<void>>(Promise.resolve());
  const canContactBackend = !isChanging && !runtime.configurationError;

  const healthQuery = useQuery({
    queryKey: workbenchQueryKeys.health(),
    queryFn: ({ signal }) => fetchHealth({ signal }),
    enabled: canContactBackend,
    retry: false,
    refetchInterval: 10_000,
  });
  const capabilitiesQuery = useQuery({
    queryKey: workbenchQueryKeys.capabilities(),
    queryFn: ({ signal }) => fetchCapabilities({ signal }),
    enabled: canContactBackend,
    retry: false,
  });
  if (capabilitiesQuery.isError) {
    clearWorkbenchAuthenticationObservation();
  } else if (capabilitiesQuery.isSuccess) {
    observeWorkbenchAuthMode(capabilitiesQuery.data.authMode);
  }
  const modelsQuery = useQuery({
    queryKey: workbenchQueryKeys.models(),
    queryFn: ({ signal }) => fetchModels({ signal }),
    enabled:
      canContactBackend &&
      capabilitiesQuery.isSuccess &&
      (capabilitiesQuery.data.authMode === "none" || runtime.hasAuthToken),
    retry: false,
    staleTime: 5 * 60_000,
    refetchOnMount: "always",
  });
  const authenticationVerified = isWorkbenchAuthenticationVerified();
  useEffect(() => {
    if (
      capabilitiesQuery.isSuccess &&
      capabilitiesQuery.data.authMode === "bearer" &&
      runtime.hasAuthToken &&
      !authenticationVerified &&
      modelsQuery.isSuccess &&
      !modelsQuery.isFetching
    ) {
      void queryClient.invalidateQueries({
        queryKey: workbenchQueryKeys.models(),
      });
    }
  }, [
    authenticationVerified,
    capabilitiesQuery.data,
    capabilitiesQuery.isSuccess,
    modelsQuery.isFetching,
    modelsQuery.isSuccess,
    queryClient,
    runtime.hasAuthToken,
  ]);

  const registerReset = useCallback<RegisterConnectionReset>((reset) => {
    resetRegistryRef.current.add(reset);
    return () => resetRegistryRef.current.delete(reset);
  }, []);

  const serialize = useCallback(
    <Result,>(transition: () => Promise<Result>) => {
      const result = transitionQueueRef.current.then(transition, transition);
      transitionQueueRef.current = result.then(
        () => undefined,
        () => undefined,
      );
      return result;
    },
    [],
  );

  const publishRuntime = useCallback(() => {
    setRuntime(workbenchConnectionRuntimeSnapshot());
  }, []);

  const resetProtectedState = useCallback(() => {
    for (const reset of Array.from(resetRegistryRef.current)) {
      reset();
    }
  }, []);
  const actionEnvironment = useMemo<WorkbenchConnectionActionEnvironment>(
    () => ({
      queryClient,
      publishRuntime,
      resetProtectedState,
      setIsChanging,
    }),
    [publishRuntime, queryClient, resetProtectedState],
  );

  const useApiBaseUrl = useCallback(
    (url: string) =>
      serialize(async () => {
        const actions = await loadWorkbenchConnectionActions();
        return actions.useApiBaseUrl(actionEnvironment, url);
      }),
    [actionEnvironment, serialize],
  );

  const resetApiBaseUrl = useCallback(
    () =>
      serialize(async () => {
        const actions = await loadWorkbenchConnectionActions();
        return actions.resetApiBaseUrl(actionEnvironment);
      }),
    [actionEnvironment, serialize],
  );

  const signIn = useCallback(
    (rawToken: string) =>
      serialize(async () => {
        const actions = await loadWorkbenchConnectionActions();
        return actions.signIn(actionEnvironment, rawToken);
      }),
    [actionEnvironment, serialize],
  );

  const logout = useCallback(
    () =>
      serialize(async () => {
        const actions = await loadWorkbenchConnectionActions();
        return actions.logout(actionEnvironment);
      }),
    [actionEnvironment, serialize],
  );

  const retry = useCallback(async () => {
    const actions = await loadWorkbenchConnectionActions();
    await actions.retry(actionEnvironment);
  }, [actionEnvironment]);

  const auth = authenticationState({
    runtime,
    capabilitiesQuery,
    modelsQuery,
    authenticationVerified,
  });
  const storageMessage =
    runtime.storageMessages.sessionToken ?? runtime.storageMessages.apiBaseUrl;
  const connectionValue = useMemo<WorkbenchConnection>(
    () => ({
      connection: {
        apiBaseUrl: runtime.apiBaseUrl,
        isOnline: healthQuery.data?.status === "ok",
        isChanging,
        configurationError: runtime.configurationError,
      },
      authentication: {
        mode: auth.mode,
        state: auth.state,
        hasToken: runtime.hasAuthToken,
      },
      storage: {
        ...runtime.storage,
        message: storageMessage,
      },
      actions: {
        useApiBaseUrl,
        resetApiBaseUrl,
        signIn,
        logout,
        retry,
      },
    }),
    [
      auth.mode,
      auth.state,
      healthQuery.data?.status,
      isChanging,
      logout,
      resetApiBaseUrl,
      retry,
      runtime,
      signIn,
      storageMessage,
      useApiBaseUrl,
    ],
  );
  const capabilityValue = useMemo<WorkbenchCapabilities>(
    () => ({
      capabilities: capabilitiesQuery.data
        ? featureCapabilities(capabilitiesQuery.data)
        : DEFAULT_WORKBENCH_CAPABILITIES,
    }),
    [capabilitiesQuery.data],
  );

  return (
    <ConnectionResetRegistryProvider value={registerReset}>
      <WorkbenchConnectionContextProvider value={connectionValue}>
        <WorkbenchCapabilitiesContextProvider value={capabilityValue}>
          {children}
        </WorkbenchCapabilitiesContextProvider>
      </WorkbenchConnectionContextProvider>
    </ConnectionResetRegistryProvider>
  );
}
