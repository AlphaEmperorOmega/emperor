import { describe, expect, it } from "vitest";
import { formatNumber, formatRunDisplayName, formatRunTimestamp } from "@/lib/format";

describe("formatNumber", () => {
  it("formats normal finite values with up to four fraction digits", () => {
    expect(formatNumber(12)).toBe("12");
    expect(formatNumber(12.34567)).toBe("12.3457");
    expect(formatNumber(-12.34567)).toBe("-12.3457");
  });

  it("formats large, tiny, zero, and negative values with the shared exponential policy", () => {
    expect(formatNumber(1000)).toBe("1.00e+3");
    expect(formatNumber(0.0005)).toBe("5.00e-4");
    expect(formatNumber(0)).toBe("0.00e+0");
    expect(formatNumber(-0.0005)).toBe("-5.00e-4");
  });

  it("formats non-finite values as zero", () => {
    expect(formatNumber(Infinity)).toBe("0");
    expect(formatNumber(-Infinity)).toBe("0");
    expect(formatNumber(NaN)).toBe("0");
  });
});

describe("formatRunTimestamp", () => {
  it("passes timestamp strings through", () => {
    expect(formatRunTimestamp("2026-06-01 01:02:03")).toBe("2026-06-01 01:02:03");
  });

  it("falls back to unknown when the timestamp is missing", () => {
    expect(formatRunTimestamp()).toBe("unknown");
    expect(formatRunTimestamp(null)).toBe("unknown");
  });
});

describe("formatRunDisplayName", () => {
  it("uses the run name when available", () => {
    expect(
      formatRunDisplayName({
        name: "baseline",
        id: "run-1",
        startTime: "2026-06-01 01:02:03",
      }),
    ).toBe("baseline · 2026-06-01 01:02:03");
  });

  it("falls back to id when the name is missing", () => {
    expect(
      formatRunDisplayName({
        id: "run-1",
        startTime: "2026-06-01 01:02:03",
      }),
    ).toBe("run-1 · 2026-06-01 01:02:03");
  });

  it("uses the missing timestamp fallback", () => {
    expect(formatRunDisplayName({ name: "baseline" })).toBe("baseline · unknown");
  });
});
