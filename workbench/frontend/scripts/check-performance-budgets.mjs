import { resolve } from "node:path";
import {
  assertBuildPerformanceBudgets,
  collectBuildPerformanceEvidence,
  formatPerformanceKilobytes,
} from "./performance-evidence.mjs";

const NEXT_DIRECTORY = resolve(".next");
const evidence = collectBuildPerformanceEvidence(NEXT_DIRECTORY);
assertBuildPerformanceBudgets(evidence);
const { first_load: firstLoad, route_specific: routeSpecific } = evidence.budgets;

console.log("Performance budgets passed");
console.log(
  `- First-load JavaScript: ${formatPerformanceKilobytes(firstLoad.gzip_bytes)} / ${formatPerformanceKilobytes(firstLoad.budget_bytes)}`,
);
console.log(
  `- Route-specific JavaScript: ${formatPerformanceKilobytes(routeSpecific.gzip_bytes)} / ${formatPerformanceKilobytes(routeSpecific.budget_bytes)}`,
);
for (const { label, bytes } of evidence.deferred) {
  console.log(`- Deferred ${label}: ${formatPerformanceKilobytes(bytes)} gzip`);
}
