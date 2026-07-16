import { describe, expect, it, vi } from "vitest";
import { createLazyFunction, createLazyValue } from "@/lib/lazy-value";

describe("lazy values", () => {
  it("deduplicates in-flight loads and caches successful values", async () => {
    let resolve!: (value: object) => void;
    const value = {};
    const load = vi.fn(
      () =>
        new Promise<object>((promiseResolve) => {
          resolve = promiseResolve;
        }),
    );
    const lazy = createLazyValue(load);

    const first = lazy();
    const second = lazy();
    expect(first).toBe(second);
    expect(load).toHaveBeenCalledOnce();

    resolve(value);
    await expect(first).resolves.toBe(value);
    await expect(lazy()).resolves.toBe(value);
    expect(load).toHaveBeenCalledOnce();
  });

  it("clears rejected loads so the next caller can retry", async () => {
    const load = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new Error("chunk unavailable"))
      .mockResolvedValueOnce("ready");
    const lazy = createLazyValue(load);

    await expect(lazy()).rejects.toThrow("chunk unavailable");
    await expect(lazy()).resolves.toBe("ready");
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("loads and invokes lazy functions through the same cache", async () => {
    const implementation = vi.fn((value: number) => value * 2);
    const load = vi.fn(async () => implementation);
    const lazy = createLazyFunction(load);

    await expect(lazy(3)).resolves.toBe(6);
    await expect(lazy(4)).resolves.toBe(8);
    expect(load).toHaveBeenCalledOnce();
    expect(implementation).toHaveBeenCalledTimes(2);
  });
});
