import { type ReactNode } from "react";
import { createViewerContext } from "@/features/viewer/providers/create-context";
import { type LogsWorkspaceState } from "@/features/viewer/state/logs/use-logs-workspace-state";

const [LogsWorkspaceProviderBase, useLogsWorkspace] =
  createViewerContext<LogsWorkspaceState>("LogsWorkspaceContext");

export { useLogsWorkspace };

export function LogsWorkspaceProvider({
  state,
  children,
}: {
  state: LogsWorkspaceState;
  children: ReactNode;
}) {
  return <LogsWorkspaceProviderBase value={state}>{children}</LogsWorkspaceProviderBase>;
}
