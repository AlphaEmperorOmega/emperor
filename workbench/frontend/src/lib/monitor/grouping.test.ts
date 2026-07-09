import { describe, expect, it } from "vitest";

import { semanticGroupForSuffix } from "@/lib/monitor/grouping";

describe("monitor grouping", () => {
  it("groups parameter data and parameter gradients under separate accordions", () => {
    expect(semanticGroupForSuffix("weights/l2_norm")).toBe("Weights");
    expect(semanticGroupForSuffix("weights/grad_norm")).toBe("Weight gradients");
    expect(semanticGroupForSuffix("weights/update_ratio")).toBe("Weight gradients");
    expect(semanticGroupForSuffix("bias/mean")).toBe("Bias");
    expect(semanticGroupForSuffix("bias/grad_mean")).toBe("Bias gradients");
    expect(semanticGroupForSuffix("grad_norm")).toBe("Gradients");
    expect(semanticGroupForSuffix("optimizer/grad_norm")).toBe("Gradients");
  });
});
