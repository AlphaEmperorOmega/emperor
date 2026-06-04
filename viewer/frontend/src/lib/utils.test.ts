import { describe, expect, it } from "vitest";
import { cn, errorMessage } from "@/lib/utils";

// Characterization tests for the two shared utilities.

describe("cn", () => {
  it("joins truthy class values and drops falsey ones", () => {
    expect(cn("a", false, undefined, null, "b")).toBe("a b");
  });

  it("supports clsx object and array syntax", () => {
    expect(cn(["a", { b: true, c: false }], "d")).toBe("a b d");
  });

  it("lets tailwind-merge resolve conflicting utilities (last wins)", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
    expect(cn("text-sm", "text-lg")).toBe("text-lg");
  });
});

describe("errorMessage", () => {
  it("returns the message for Error instances", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
  });

  it("stringifies non-Error values", () => {
    expect(errorMessage("plain string")).toBe("plain string");
    expect(errorMessage(42)).toBe("42");
    expect(errorMessage(undefined)).toBe("undefined");
    expect(errorMessage({ detail: "x" })).toBe("[object Object]");
  });
});
