type ModelMetadataApi = typeof import("@/lib/api/models");

import { createLazyFunction } from "@/lib/lazy-value";

export const fetchConfigSchema = createLazyFunction(() =>
  import("@/lib/api/models").then((module) => module.fetchConfigSchema),
);
export const fetchDatasets = createLazyFunction(() =>
  import("@/lib/api/models").then((module) => module.fetchDatasets),
);
export const fetchMonitors = createLazyFunction(() =>
  import("@/lib/api/models").then((module) => module.fetchMonitors),
);
export const fetchPresets = createLazyFunction(() =>
  import("@/lib/api/models").then((module) => module.fetchPresets),
);
export const fetchSearchSpace = createLazyFunction(() =>
  import("@/lib/api/models").then((module) => module.fetchSearchSpace),
);

fetchConfigSchema satisfies ModelMetadataApi["fetchConfigSchema"];
fetchDatasets satisfies ModelMetadataApi["fetchDatasets"];
fetchMonitors satisfies ModelMetadataApi["fetchMonitors"];
fetchPresets satisfies ModelMetadataApi["fetchPresets"];
fetchSearchSpace satisfies ModelMetadataApi["fetchSearchSpace"];
