import { resolve } from "node:path";
import {
  PERFORMANCE_EVIDENCE_POLICY,
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
  `- First-load JavaScript: ${formatPerformanceKilobytes(firstLoad.gzip_bytes)} / ${formatPerformanceKilobytes(PERFORMANCE_EVIDENCE_POLICY.requiredMeasured.firstLoadBytes)} required (${formatPerformanceKilobytes(firstLoad.budget_bytes)} policy budget)`,
);
console.log(
  `- Route-specific JavaScript: ${formatPerformanceKilobytes(routeSpecific.gzip_bytes)} / ${formatPerformanceKilobytes(PERFORMANCE_EVIDENCE_POLICY.requiredMeasured.routeSpecificBytes)} required (${formatPerformanceKilobytes(routeSpecific.budget_bytes)} policy budget)`,
);
for (const { label, bytes } of evidence.deferred) {
  console.log(`- Deferred ${label}: ${formatPerformanceKilobytes(bytes)} gzip`);
}
