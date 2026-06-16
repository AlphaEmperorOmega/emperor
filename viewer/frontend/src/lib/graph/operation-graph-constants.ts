// Dagre-free constants for the operation graph. Kept separate from
// `operation-layout.ts` so consumers that only need the node-id prefix (the
// barrel re-export, the operation view-state hook) do not pull in `dagre`.
export const OPERATION_GROUP_NODE_PREFIX = "operation-group:";
