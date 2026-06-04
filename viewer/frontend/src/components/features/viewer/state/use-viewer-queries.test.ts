import { createElement, type ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the API layer so we can observe which queries fire (the `enabled`
// gating) and with which arguments (the query-key inputs) per selection.
const mocks = vi.hoisted(() => ({
  fetchHealth: vi.fn(),
  fetchModels: vi.fn(),
  fetchPresets: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchMonitors: vi.fn(),
  fetchConfigSchema: vi.fn(),
}));
vi.mock("@/lib/api", () => mocks);

import { useViewerQueries } from "@/components/features/viewer/state/use-viewer-queries";

function renderQueries(model: string, preset: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(
    ({ m, p }: { m: string; p: string }) => useViewerQueries(m, p),
    {
      initialProps: { m: model, p: preset },
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
}

beforeEach(() => {
  mocks.fetchHealth.mockReset().mockResolvedValue({ status: "ok" });
  mocks.fetchModels.mockReset().mockResolvedValue({ models: ["linear"] });
  mocks.fetchPresets.mockReset().mockImplementation((model: string) =>
    Promise.resolve({ model, presets: [] }),
  );
  mocks.fetchDatasets.mockReset().mockImplementation((model: string) =>
    Promise.resolve({ model, datasets: [] }),
  );
  mocks.fetchMonitors.mockReset().mockImplementation((model: string) =>
    Promise.resolve({ model, monitors: [] }),
  );
  mocks.fetchConfigSchema.mockReset().mockImplementation((model: string) =>
    Promise.resolve({ model, fields: [] }),
  );
});

describe("useViewerQueries enabled gating", () => {
  it("fires only health + models when no model is selected", async () => {
    renderQueries("", "");

    await waitFor(() => expect(mocks.fetchModels).toHaveBeenCalled());
    expect(mocks.fetchHealth).toHaveBeenCalled();
    expect(mocks.fetchPresets).not.toHaveBeenCalled();
    expect(mocks.fetchDatasets).not.toHaveBeenCalled();
    expect(mocks.fetchMonitors).not.toHaveBeenCalled();
    expect(mocks.fetchConfigSchema).not.toHaveBeenCalled();
  });

  it("fires model-scoped queries once a model is selected, but not schema", async () => {
    renderQueries("linear", "");

    await waitFor(() => expect(mocks.fetchPresets).toHaveBeenCalledWith("linear"));
    expect(mocks.fetchDatasets).toHaveBeenCalledWith("linear");
    expect(mocks.fetchMonitors).toHaveBeenCalledWith("linear");
    expect(mocks.fetchConfigSchema).not.toHaveBeenCalled();
  });

  it("fires the schema query only when both model and preset are selected", async () => {
    renderQueries("linear", "base");

    await waitFor(() =>
      expect(mocks.fetchConfigSchema).toHaveBeenCalledWith("linear", "base"),
    );
  });
});
