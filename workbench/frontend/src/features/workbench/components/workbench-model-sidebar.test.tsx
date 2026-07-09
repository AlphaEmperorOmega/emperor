import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchModels } from "@/lib/api";
import { WorkbenchModelSidebar } from "@/features/workbench/components/workbench-model-sidebar";

const mocks = vi.hoisted(() => ({
  useTargetQueryStatusState: vi.fn(),
}));

vi.mock("@/features/workbench/providers/workbench-providers", () => ({
  useTargetQueryStatusState: mocks.useTargetQueryStatusState,
}));

vi.mock("@/features/workbench/components/screen/target-preset-panel", () => ({
  TargetPresetPanel: () => null,
}));

vi.mock("@/features/workbench/components/config/config-summary-panel", () => ({
  ConfigSummaryPanel: () => null,
}));

type FakeResponseInit = {
  ok?: boolean;
  status?: number;
  statusText?: string;
  json?: () => Promise<unknown>;
};

function fakeResponse(init: FakeResponseInit) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? "OK",
    json: init.json ?? (() => Promise.resolve({})),
  } as unknown as Response;
}

type FetchFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

async function createUnauthorizedModelsError() {
  vi.stubGlobal(
    "fetch",
    vi.fn<FetchFn>(() =>
      Promise.resolve(
        fakeResponse({
          ok: false,
          status: 401,
          statusText: "Unauthorized",
          json: () => Promise.resolve({ detail: "Missing or invalid bearer credentials" }),
        }),
      ),
    ),
  );

  try {
    await fetchModels();
  } catch (error) {
    return error;
  }
  throw new Error("fetchModels unexpectedly succeeded");
}

function queryState(error?: unknown) {
  return {
    isError: error !== undefined,
    error,
  };
}

function renderSidebarWithModelError(error: unknown) {
  mocks.useTargetQueryStatusState.mockReturnValue({
    modelsQuery: queryState(error),
    presetsQuery: queryState(),
    datasetsQuery: queryState(),
    schemaQuery: queryState(),
  });

  render(<WorkbenchModelSidebar onOpenFullConfig={vi.fn()} />);
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("WorkbenchModelSidebar", () => {
  it("surfaces unauthorized model-list failures as authentication required", async () => {
    const error = await createUnauthorizedModelsError();

    renderSidebarWithModelError(error);

    expect(screen.getByText("Authentication required")).toBeInTheDocument();
    expect(screen.queryByText("Backend unavailable")).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "GET /models from http://127.0.0.1:9999 failed with 401: Missing or invalid bearer credentials",
      ),
    ).toBeInTheDocument();
  });
});
