type LogsApi = typeof import("@/lib/api/logs");

import { createLazyFunction } from "@/lib/lazy-value";

export const fetchLogCheckpoints = createLazyFunction(() =>
  import("@/lib/api/logs").then((module) => module.fetchLogCheckpoints),
);
export const fetchLogExperiments = createLazyFunction(() =>
  import("@/lib/api/logs").then((module) => module.fetchLogExperiments),
);
export const fetchLogMedia = createLazyFunction(() =>
  import("@/lib/api/logs").then((module) => module.fetchLogMedia),
);
export const fetchLogRuns = createLazyFunction(() =>
  import("@/lib/api/logs").then((module) => module.fetchLogRuns),
);
export const fetchLogRunArtifacts = createLazyFunction(() =>
  import("@/lib/api/logs").then((module) => module.fetchLogRunArtifacts),
);
export const fetchLogScalars = createLazyFunction(() =>
  import("@/lib/api/logs").then((module) => module.fetchLogScalars),
);
export const fetchLogTags = createLazyFunction(() =>
  import("@/lib/api/logs").then((module) => module.fetchLogTags),
);

fetchLogCheckpoints satisfies LogsApi["fetchLogCheckpoints"];
fetchLogExperiments satisfies LogsApi["fetchLogExperiments"];
fetchLogMedia satisfies LogsApi["fetchLogMedia"];
fetchLogRuns satisfies LogsApi["fetchLogRuns"];
fetchLogRunArtifacts satisfies LogsApi["fetchLogRunArtifacts"];
fetchLogScalars satisfies LogsApi["fetchLogScalars"];
fetchLogTags satisfies LogsApi["fetchLogTags"];
