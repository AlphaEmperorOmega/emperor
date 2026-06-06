import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clearSessionAuthToken,
  getSessionAuthToken,
  setSessionAuthToken,
} from "@/lib/auth-token";

function stubWindow(value: unknown) {
  vi.stubGlobal("window", value as Window & typeof globalThis);
}

function storageThatThrows() {
  const throwStorageError = () => {
    throw new Error("storage unavailable");
  };

  return {
    length: 0,
    clear: vi.fn(throwStorageError),
    getItem: vi.fn(throwStorageError),
    key: vi.fn(throwStorageError),
    removeItem: vi.fn(throwStorageError),
    setItem: vi.fn(throwStorageError),
  } as unknown as Storage;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.sessionStorage.clear();
});

describe("session auth token helper", () => {
  it("stores, reads, and clears the token in sessionStorage", () => {
    expect(getSessionAuthToken()).toBeNull();

    setSessionAuthToken("secret-token");

    expect(getSessionAuthToken()).toBe("secret-token");

    clearSessionAuthToken();

    expect(getSessionAuthToken()).toBeNull();
  });

  it("returns null and no-ops when window is missing", () => {
    stubWindow(undefined);

    expect(getSessionAuthToken()).toBeNull();
    expect(() => setSessionAuthToken("secret-token")).not.toThrow();
    expect(() => clearSessionAuthToken()).not.toThrow();
  });

  it("returns null and no-ops when sessionStorage is unavailable", () => {
    stubWindow({});

    expect(getSessionAuthToken()).toBeNull();
    expect(() => setSessionAuthToken("secret-token")).not.toThrow();
    expect(() => clearSessionAuthToken()).not.toThrow();
  });

  it("swallows sessionStorage access and method failures", () => {
    const windowWithThrowingStorageGetter = {};
    Object.defineProperty(windowWithThrowingStorageGetter, "sessionStorage", {
      get: () => {
        throw new Error("storage blocked");
      },
    });

    stubWindow(windowWithThrowingStorageGetter);

    expect(getSessionAuthToken()).toBeNull();
    expect(() => setSessionAuthToken("secret-token")).not.toThrow();
    expect(() => clearSessionAuthToken()).not.toThrow();

    stubWindow({ sessionStorage: storageThatThrows() });

    expect(getSessionAuthToken()).toBeNull();
    expect(() => setSessionAuthToken("secret-token")).not.toThrow();
    expect(() => clearSessionAuthToken()).not.toThrow();
  });
});
