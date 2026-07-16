import {
  parseWorkbenchWorkspace,
  type WorkbenchWorkspace,
} from "@/types/workbench";

export type DeferredWorkbenchWorkspace = Extract<
  WorkbenchWorkspace,
  "logs" | "training"
>;

export type WorkbenchLocationSnapshot = Readonly<{
  href: string;
  workspace: WorkbenchWorkspace;
  revision: number;
  deferredWorkspaceOrder: readonly DeferredWorkbenchWorkspace[];
}>;

const listeners = new Set<() => void>();
let snapshot: WorkbenchLocationSnapshot | null = null;
let restoreHistoryObservation: (() => void) | null = null;

function workspaceFromHref(href: string) {
  return parseWorkbenchWorkspace(new URL(href).searchParams.get("workspace"));
}

function appendDeferredWorkspace(
  order: readonly DeferredWorkbenchWorkspace[],
  workspace: WorkbenchWorkspace,
) {
  if (
    (workspace !== "logs" && workspace !== "training") ||
    order.includes(workspace)
  ) {
    return order;
  }
  return [...order, workspace];
}

function browserHref() {
  return typeof window === "undefined"
    ? "http://workbench.local/"
    : window.location.href;
}

function createSnapshot(
  href: string,
  revision: number,
  previousOrder: readonly DeferredWorkbenchWorkspace[] = [],
): WorkbenchLocationSnapshot {
  const workspace = workspaceFromHref(href);
  return Object.freeze({
    href,
    workspace,
    revision,
    deferredWorkspaceOrder: Object.freeze(
      appendDeferredWorkspace(previousOrder, workspace),
    ),
  });
}

function currentSnapshot() {
  snapshot ??= createSnapshot(browserHref(), 0);
  return snapshot;
}

function publishBrowserLocation() {
  const current = currentSnapshot();
  const href = browserHref();
  if (href === current.href) {
    return;
  }
  snapshot = createSnapshot(
    href,
    current.revision + 1,
    current.deferredWorkspaceOrder,
  );
  for (const listener of listeners) {
    listener();
  }
}

function observeHistory() {
  if (typeof window === "undefined" || restoreHistoryObservation) {
    return;
  }
  const { history } = window;
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;
  const wrap =
    (implementation: History["pushState"]) =>
    (...args: Parameters<History["pushState"]>) => {
      implementation.apply(history, args);
      publishBrowserLocation();
    };
  history.pushState = wrap(originalPushState);
  history.replaceState = wrap(originalReplaceState);
  window.addEventListener("popstate", publishBrowserLocation);
  window.addEventListener("hashchange", publishBrowserLocation);
  restoreHistoryObservation = () => {
    history.pushState = originalPushState;
    history.replaceState = originalReplaceState;
    window.removeEventListener("popstate", publishBrowserLocation);
    window.removeEventListener("hashchange", publishBrowserLocation);
    restoreHistoryObservation = null;
  };
}

export function subscribeWorkbenchLocation(
  listener: () => void,
  initialWorkspace: WorkbenchWorkspace = "model",
) {
  if (listeners.size === 0) {
    const initialOrder =
      initialWorkspace === "logs" || initialWorkspace === "training"
        ? [initialWorkspace]
        : [];
    snapshot = createSnapshot(browserHref(), 0, initialOrder);
  }
  listeners.add(listener);
  observeHistory();
  publishBrowserLocation();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      restoreHistoryObservation?.();
    }
  };
}

export function getWorkbenchLocationSnapshot() {
  const current = currentSnapshot();
  const href = browserHref();
  if (href !== current.href) {
    snapshot = createSnapshot(
      href,
      current.revision + 1,
      current.deferredWorkspaceOrder,
    );
  }
  return snapshot!;
}

export function createWorkbenchLocationServerSnapshot(
  initialWorkspace: WorkbenchWorkspace,
) {
  const url = new URL("http://workbench.local/");
  if (initialWorkspace !== "model") {
    url.searchParams.set("workspace", initialWorkspace);
  }
  return createSnapshot(url.href, 0);
}

export function workspaceHref(
  href: string,
  workspace: WorkbenchWorkspace,
) {
  const url = new URL(href, "http://workbench.local");
  url.searchParams.set("workspace", workspace);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function navigateToWorkbenchWorkspace(workspace: WorkbenchWorkspace) {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set("workspace", workspace);
  window.history.pushState(window.history.state, "", url);
  publishBrowserLocation();
}
