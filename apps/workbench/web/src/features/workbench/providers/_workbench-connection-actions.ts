import { type QueryClient } from "@tanstack/react-query";
import {
  abortWorkbenchConnectionRuntimeTransition,
  beginWorkbenchConnectionRuntimeTransition,
  completeWorkbenchConnectionRuntimeTransition,
  getWorkbenchConnectionRuntimeSnapshot,
  reportWorkbenchConnectionStorage,
  validateWorkbenchApiBaseUrl,
  WORKBENCH_API_BASE_URL,
  WORKBENCH_API_BASE_URL_STORAGE_KEY,
  WORKBENCH_AUTH_TOKEN_STORAGE_KEY,
} from "@/lib/api/_connection-runtime";
import { persistConnectionStorageValue } from "@/lib/api/_connection-storage";
import { workbenchQueryKeys } from "@/lib/query-keys";

type WorkbenchConnectionTransitionRequest =
  | { kind: "use-api-base-url"; url: string }
  | { kind: "reset-api-base-url" }
  | { kind: "sign-in"; token: string }
  | { kind: "logout" };

type WorkbenchConnectionTransitionScope = "all" | "protected";

type WorkbenchConnectionTransitionPlan = Readonly<{
  area: "apiBaseUrl" | "sessionToken";
  identity: string | null;
  persist: () =>
    | { ok: true; rollback: () => string | null }
    | { ok: false; message: string };
}>;

type PlannedWorkbenchConnectionTransition =
  | {
      ok: true;
      plan: WorkbenchConnectionTransitionPlan;
      scope: WorkbenchConnectionTransitionScope;
    }
  | { ok: false; message: string };

const API_BASE_URL_ROLLBACK_FAILURE =
  "This browser could not restore the previous API base URL after the connection transition failed.";
const AUTH_TOKEN_ROLLBACK_FAILURE =
  "This browser could not restore the previous session token after the connection transition failed.";

function apiBaseUrlPlan({
  identity,
  storageValue,
  failureMessage,
}: {
  identity: string;
  storageValue: string | null;
  failureMessage: string;
}): PlannedWorkbenchConnectionTransition {
  return {
    ok: true,
    plan: {
      area: "apiBaseUrl",
      identity,
      persist: () =>
        persistConnectionStorageValue({
          kind: "localStorage",
          key: WORKBENCH_API_BASE_URL_STORAGE_KEY,
          value: storageValue,
          failureMessage,
          rollbackFailureMessage: API_BASE_URL_ROLLBACK_FAILURE,
        }),
    },
    scope: "all",
  };
}

function authTokenPlan({
  identity,
  failureMessage,
}: {
  identity: string | null;
  failureMessage: string;
}): PlannedWorkbenchConnectionTransition {
  return {
    ok: true,
    plan: {
      area: "sessionToken",
      identity,
      persist: () =>
        persistConnectionStorageValue({
          kind: "sessionStorage",
          key: WORKBENCH_AUTH_TOKEN_STORAGE_KEY,
          value: identity,
          failureMessage,
          rollbackFailureMessage: AUTH_TOKEN_ROLLBACK_FAILURE,
        }),
    },
    scope: "protected",
  };
}

function connectionTransitionPlan(
  request: WorkbenchConnectionTransitionRequest,
): PlannedWorkbenchConnectionTransition {
  if (request.kind === "use-api-base-url") {
    const validation = validateWorkbenchApiBaseUrl(request.url);
    return validation.ok
      ? apiBaseUrlPlan({
          identity: validation.value,
          storageValue: validation.value,
          failureMessage:
            "This browser could not persist the API base URL. Check browser storage permissions.",
        })
      : validation;
  }
  if (request.kind === "reset-api-base-url") {
    return apiBaseUrlPlan({
      identity: WORKBENCH_API_BASE_URL,
      storageValue: null,
      failureMessage:
        "This browser could not clear the saved API base URL. Check browser storage permissions.",
    });
  }
  if (request.kind === "sign-in") {
    const token = request.token.trim();
    return token
      ? authTokenPlan({
          identity: token,
          failureMessage:
            "This browser could not store a session token. Check browser storage permissions.",
        })
      : {
          ok: false,
          message: "Enter the bearer token supplied by the API operator.",
        };
  }
  return authTokenPlan({
    identity: null,
    failureMessage:
      "This browser could not clear the session token. Check browser storage permissions.",
  });
}

export type WorkbenchConnectionActionResult =
  | { ok: true }
  | { ok: false; message: string };

export type WorkbenchConnectionActionEnvironment = Readonly<{
  queryClient: QueryClient;
  resetProtectedState: () => void;
}>;

async function applyTransition(
  environment: WorkbenchConnectionActionEnvironment,
  request: WorkbenchConnectionTransitionRequest,
): Promise<WorkbenchConnectionActionResult> {
  const planned = connectionTransitionPlan(request);
  if (!planned.ok) {
    return planned;
  }

  const persisted = planned.plan.persist();
  if (!persisted.ok) {
    reportWorkbenchConnectionStorage(
      planned.plan.area,
      "unavailable",
      persisted.message,
    );
    return persisted;
  }

  reportWorkbenchConnectionStorage(planned.plan.area, "available", null);
  if (
    planned.plan.area === "apiBaseUrl" &&
    getWorkbenchConnectionRuntimeSnapshot().apiBaseUrl === planned.plan.identity
  ) {
    return { ok: true };
  }

  beginWorkbenchConnectionRuntimeTransition(planned.plan.area);
  try {
    await environment.queryClient.cancelQueries();
    environment.queryClient.getMutationCache().clear();
    if (planned.scope === "all") {
      environment.queryClient.getQueryCache().clear();
    } else {
      for (const query of environment.queryClient.getQueryCache().getAll()) {
        const root = query.queryKey[0];
        if (root !== "health" && root !== "capabilities") {
          environment.queryClient.getQueryCache().remove(query);
        }
      }
    }
    environment.resetProtectedState();
    completeWorkbenchConnectionRuntimeTransition(
      planned.plan.area,
      planned.plan.identity,
    );
    return { ok: true };
  } catch (error) {
    const rollbackMessage = persisted.rollback();
    abortWorkbenchConnectionRuntimeTransition(
      planned.plan.area,
      rollbackMessage,
    );
    throw error;
  }
}

export async function useApiBaseUrl(
  environment: WorkbenchConnectionActionEnvironment,
  url: string,
) {
  return applyTransition(environment, { kind: "use-api-base-url", url });
}

export async function resetApiBaseUrl(
  environment: WorkbenchConnectionActionEnvironment,
) {
  return applyTransition(environment, { kind: "reset-api-base-url" });
}

export async function signIn(
  environment: WorkbenchConnectionActionEnvironment,
  rawToken: string,
) {
  return applyTransition(environment, { kind: "sign-in", token: rawToken });
}

export async function logout(
  environment: WorkbenchConnectionActionEnvironment,
) {
  return applyTransition(environment, { kind: "logout" });
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
