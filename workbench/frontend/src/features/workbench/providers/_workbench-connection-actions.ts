import { type QueryClient } from "@tanstack/react-query";
import {
  validateWorkbenchApiBaseUrl,
  workbenchConnectionRuntimeSnapshot,
  WORKBENCH_API_BASE_URL,
} from "@/lib/api/_connection-runtime";
import {
  beginWorkbenchConnectionTransition,
  commitWorkbenchApiBaseUrl,
  commitWorkbenchAuthToken,
  finishWorkbenchConnectionTransition,
  persistClearedWorkbenchAuthToken,
  persistDefaultWorkbenchApiBaseUrl,
  persistWorkbenchApiBaseUrl,
  persistWorkbenchAuthToken,
} from "@/lib/api/_connection-runtime-actions";
import { workbenchQueryKeys } from "@/lib/query-keys";

export type WorkbenchConnectionActionResult =
  | { ok: true }
  | { ok: false; message: string };

export type WorkbenchConnectionActionEnvironment = Readonly<{
  queryClient: QueryClient;
  publishRuntime: () => void;
  resetProtectedState: () => void;
  setIsChanging: (isChanging: boolean) => void;
}>;

function failAction(
  environment: WorkbenchConnectionActionEnvironment,
  message: string,
): WorkbenchConnectionActionResult {
  environment.publishRuntime();
  return { ok: false, message };
}

async function applyTransition(
  environment: WorkbenchConnectionActionEnvironment,
  commit: () => void,
  scope: "all" | "protected",
): Promise<WorkbenchConnectionActionResult> {
  environment.setIsChanging(true);
  beginWorkbenchConnectionTransition();
  try {
    await environment.queryClient.cancelQueries();
    commit();
    environment.queryClient.getMutationCache().clear();
    if (scope === "all") {
      environment.queryClient.getQueryCache().clear();
    } else {
      for (const query of environment.queryClient.getQueryCache().getAll()) {
        const root = query.queryKey[0];
        if (root !== "health" && root !== "capabilities") {
          environment.queryClient.getQueryCache().remove(query);
        }
      }
    }
    finishWorkbenchConnectionTransition();
    environment.resetProtectedState();
    environment.publishRuntime();
    return { ok: true };
  } finally {
    finishWorkbenchConnectionTransition();
    environment.setIsChanging(false);
  }
}

export async function useApiBaseUrl(
  environment: WorkbenchConnectionActionEnvironment,
  url: string,
) {
  const validation = validateWorkbenchApiBaseUrl(url);
  if (!validation.ok) {
    return failAction(environment, validation.message);
  }
  if (validation.value === workbenchConnectionRuntimeSnapshot().apiBaseUrl) {
    return { ok: true } as const;
  }
  const persisted = persistWorkbenchApiBaseUrl(validation.value);
  if (!persisted.ok) {
    return failAction(environment, persisted.message);
  }
  return applyTransition(
    environment,
    () => commitWorkbenchApiBaseUrl(validation.value),
    "all",
  );
}

export async function resetApiBaseUrl(
  environment: WorkbenchConnectionActionEnvironment,
) {
  const persisted = persistDefaultWorkbenchApiBaseUrl();
  if (!persisted.ok) {
    return failAction(environment, persisted.message);
  }
  const current = workbenchConnectionRuntimeSnapshot();
  if (current.apiBaseUrl === WORKBENCH_API_BASE_URL) {
    environment.publishRuntime();
    return { ok: true } as const;
  }
  return applyTransition(
    environment,
    () => commitWorkbenchApiBaseUrl(WORKBENCH_API_BASE_URL),
    "all",
  );
}

export async function signIn(
  environment: WorkbenchConnectionActionEnvironment,
  rawToken: string,
) {
  const token = rawToken.trim();
  if (!token) {
    return failAction(
      environment,
      "Enter the bearer token supplied by the API operator.",
    );
  }
  const persisted = persistWorkbenchAuthToken(token);
  if (!persisted.ok) {
    return failAction(environment, persisted.message);
  }
  return applyTransition(
    environment,
    () => commitWorkbenchAuthToken(token),
    "protected",
  );
}

export async function logout(
  environment: WorkbenchConnectionActionEnvironment,
) {
  const persisted = persistClearedWorkbenchAuthToken();
  if (!persisted.ok) {
    return failAction(environment, persisted.message);
  }
  return applyTransition(
    environment,
    () => commitWorkbenchAuthToken(null),
    "protected",
  );
}

export async function retry(
  environment: WorkbenchConnectionActionEnvironment,
) {
  await Promise.all([
    environment.queryClient.invalidateQueries({
      queryKey: workbenchQueryKeys.health(),
    }),
    environment.queryClient.invalidateQueries({
      queryKey: workbenchQueryKeys.capabilities(),
    }),
  ]);
  if (
    environment.queryClient.getQueryState(workbenchQueryKeys.capabilities())
      ?.status === "success"
  ) {
    await environment.queryClient.invalidateQueries({
      queryKey: workbenchQueryKeys.models(),
    });
  }
}
