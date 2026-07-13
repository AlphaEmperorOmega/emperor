import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const workspaceSource = readFileSync(
  new URL("./_use-logs-workspace-state.ts", import.meta.url),
  "utf8",
);
const chartsSource = readFileSync(
  new URL("./_logs-chart-state.ts", import.meta.url),
  "utf8",
);
const providerSource = readFileSync(
  new URL("../../providers/logs-workspace-provider.tsx", import.meta.url),
  "utf8",
);

function interfaceBlock(start: string, end: string) {
  return workspaceSource.slice(
    workspaceSource.indexOf(start),
    workspaceSource.indexOf(end),
  );
}

describe("Logs Charts and Run detail architecture", () => {
  it("keeps query results, setters, and low-level toggles out of both Interfaces", () => {
    const chartsInterface = interfaceBlock(
      "export type LogsChartsInput",
      "export type LogsRunDetail",
    );
    const detailInterface = interfaceBlock(
      "export type LogsRunDetail",
      "export type LogsDeletion",
    );

    for (const forbidden of [
      "UseQueryResult",
      "runsQuery",
      "tagsQuery",
      "setSelected",
      "toggleMetricGroup",
      "toggleTag",
    ]) {
      expect(chartsInterface).not.toContain(forbidden);
      expect(detailInterface).not.toContain(forbidden);
    }
    expect(chartsSource).not.toContain("state.runsQuery");
    expect(chartsSource).not.toContain("state.tagsQuery");
  });

  it("keeps chart and Run Artifact hooks private behind Activity-safe providers", () => {
    expect(providerSource).toContain("function LogsChartsProvider");
    expect(providerSource).toContain(
      "<LogsChartsProvider input={workspace.charts}>",
    );
    expect(providerSource).not.toContain("enabled ? (");
    expect(providerSource).not.toContain("LogsChartSourceProvider");
    expect(providerSource).not.toContain("useLogRunArtifactsQuery");
    expect(providerSource).toContain(
      "createWorkbenchContext<LogsRunDetail>",
    );
    expect(workspaceSource).toContain("useLogRunArtifactsQuery({");
  });
});
