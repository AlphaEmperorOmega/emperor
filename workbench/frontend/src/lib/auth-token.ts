const AUTH_TOKEN_STORAGE_KEY = "emperor.workbench.authToken";

function getSessionStorage() {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch {
    // Storage access can throw in locked-down browser contexts.
    return null;
  }
}

export function getSessionAuthToken() {
  const storage = getSessionStorage();
  if (!storage) {
    return null;
  }
  try {
    return storage.getItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    // Token lookup should never break rendering or API setup.
    return null;
  }
}

export function setSessionAuthToken(token: string) {
  const storage = getSessionStorage();
  if (!storage) {
    return;
  }
  try {
    storage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
  } catch {
    // Ignore storage failures; callers can continue without a session token.
  }
}

export function clearSessionAuthToken() {
  const storage = getSessionStorage();
  if (!storage) {
    return;
  }
  try {
    storage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    // Ignore storage failures; clearing is best-effort.
  }
}
