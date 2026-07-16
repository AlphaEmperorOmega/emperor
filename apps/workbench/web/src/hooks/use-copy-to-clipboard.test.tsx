import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

afterEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: undefined,
  });
});

describe("useCopyToClipboard", () => {
  it("ignores completion for text that is no longer current", async () => {
    const firstWrite = deferred();
    const writeText = vi
      .fn<(text: string) => Promise<void>>()
      .mockImplementation((text) =>
        text === "first" ? firstWrite.promise : Promise.resolve(),
      );
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const rendered = renderHook(
      ({ text }: { text: string }) => useCopyToClipboard(text),
      { initialProps: { text: "first" } },
    );

    let pendingCopy!: Promise<void>;
    act(() => {
      pendingCopy = rendered.result.current.copy();
    });
    expect(writeText).toHaveBeenCalledWith("first");

    rendered.rerender({ text: "second" });
    await act(async () => {
      firstWrite.resolve();
      await pendingCopy;
    });

    expect(rendered.result.current.status).toBe("idle");

    await act(async () => {
      await rendered.result.current.copy();
    });
    expect(writeText).toHaveBeenLastCalledWith("second");
    expect(rendered.result.current.status).toBe("copied");
  });
});
