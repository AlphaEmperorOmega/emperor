import { type QueryClient } from "@tanstack/react-query";
import {
  validateWorkbenchApiBaseUrl,
  withWorkbenchConnectionRuntimeTransition,
  WORKBENCH_API_BASE_URL,
  WORKBENCH_API_BASE_URL_STORAGE_KEY,
  WORKBENCH_AUTH_TOKEN_STORAGE_KEY,
} from "@/lib/api/_connection-runtime";
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

function storageAdapter(kind: "localStorage" | "sessionStorage") {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const storage = window[kind];
    return typeof storage?.getItem === "function" ? storage : null;
  } catch {
    return null;
  }
}

function writeStorageValue(
  storage: Storage,
  key: string,
  value: string | null,
) {
  if (value === null) {
    storage.removeItem(key);
  } else {
    storage.setItem(key, value);
  }
  if (storage.getItem(key) !== value) {
    throw new Error("Browser storage did not retain the requested value.");
  }
}

function persistStorageValue(
  kind: "localStorage" | "sessionStorage",
  key: string,
  value: string | null,
  failureMessage: string,
  rollbackFailureMessage: string,
):
  | { ok: true; rollback: () => string | null }
  | { ok: false; message: string } {
  const storage = storageAdapter(kind);
  if (!storage) {
    return { ok: false, message: failureMessage };
  }
  let previousValue: string | null = null;
  let previousValueRead = false;
  try {
    previousValue = storage.getItem(key);
    previousValueRead = true;
    writeStorageValue(storage, key, value);
  } catch {
    if (previousValueRead) {
      try {
        writeStorageValue(storage, key, previousValue);
      } catch {
        // The runtime records that persistence is unavailable.
      }
    }
    return { ok: false, message: failureMessage };
  }
  return {
    ok: true,
    rollback: () => {
      try {
        writeStorageValue(storage, key, previousValue);
        return null;
      } catch {
        return rollbackFailureMessage;
      }
    },
  };
}

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
        persistStorageValue(
          "localStorage",
          WORKBENCH_API_BASE_URL_STORAGE_KEY,
          storageValue,
          failureMessage,
          API_BASE_URL_ROLLBACK_FAILURE,
        ),
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
        persistStorageValue(
          "sessionStorage",
          WORKBENCH_AUTH_TOKEN_STORAGE_KEY,
          identity,
          failureMessage,
          AUTH_TOKEN_ROLLBACK_FAILURE,
        ),
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
  publishRuntime: () => void;
  resetProtectedState: () => void;
  setIsChanging: (isChanging: boolean) => void;
}>;

async function applyTransition(
  environment: WorkbenchConnectionActionEnvironment,
  request: WorkbenchConnectionTransitionRequest,
): Promise<WorkbenchConnectionActionResult> {
  const planned = connectionTransitionPlan(request);
  if (!planned.ok) {
    return planned;
  }
  let changing = false;
  try {
    const outcome = await withWorkbenchConnectionRuntimeTransition(
      async (current) => {
        const persisted = planned.plan.persist();
        if (!persisted.ok) {
          current.storage[planned.plan.area] = "unavailable";
          current.storageMessages[planned.plan.area] = persisted.message;
          return persisted;
        }
        current.storage[planned.plan.area] = "available";
        current.storageMessages[planned.plan.area] = null;
        if (
          planned.plan.area === "apiBaseUrl" &&
          current.apiBaseUrl === planned.plan.identity
        ) {
          return { ok: true } as const;
        }

        current.revision += 1;
        current.transitioning = true;
        current.authenticationProbeGeneration += 1;
        changing = true;
        try {
          environment.setIsChanging(true);
          await environment.queryClient.cancelQueries();
          environment.queryClient.getMutationCache().clear();
          if (planned.scope === "all") {
            environment.queryClient.getQueryCache().clear();
          } else {
            for (const query of environment.queryClient
              .getQueryCache()
              .getAll()) {
              const root = query.queryKey[0];
              if (root !== "health" && root !== "capabilities") {
                environment.queryClient.getQueryCache().remove(query);
              }
            }
          }
          environment.resetProtectedState();
          if (planned.plan.area === "apiBaseUrl") {
            current.apiBaseUrl = planned.plan.identity!;
          } else {
            current.authToken = planned.plan.identity;
          }
          current.authMode = "unknown";
          current.verifiedRevision = null;
          return { ok: true } as const;
        } catch (error) {
          const rollbackMessage = persisted.rollback();
          current.storage[planned.plan.area] = rollbackMessage
            ? "unavailable"
            : "available";
          current.storageMessages[planned.plan.area] = rollbackMessage;
          throw error;
        } finally {
          current.transitioning = false;
        }
      },
    );
    environment.publishRuntime();
    return outcome;
  } catch (error) {
    environment.publishRuntime();
    throw error;
  } finally {
    if (changing) {
      environment.setIsChanging(false);
    }
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
