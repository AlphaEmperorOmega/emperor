import { createElement, type ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the API layer so we can observe which queries fire (the `enabled`
// gating) and with which arguments (the query-key inputs) per selection.
const mocks = vi.hoisted(() => ({
  fetchHealth: vi.fn(),
  fetchCapabilities: vi.fn(),
  fetchModels: vi.fn(),
  fetchPresets: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchMonitors: vi.fn(),
  fetchConfigSchema: vi.fn(),
  fetchSearchSpace: vi.fn(),
}));
vi.mock("@/lib/api/model-catalog", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/model-catalog")>()),
  fetchModels: mocks.fetchModels,
}));
vi.mock("@/lib/api/model-metadata-client", () => ({
  fetchPresets: mocks.fetchPresets,
  fetchDatasets: mocks.fetchDatasets,
  fetchMonitors: mocks.fetchMonitors,
  fetchConfigSchema: mocks.fetchConfigSchema,
  fetchSearchSpace: mocks.fetchSearchSpace,
}));

import { useModelPackageMetadata } from "@/features/workbench/state/model-package/use-model-package-metadata";
import type { ModelIdentity } from "@/lib/api/model-catalog";

function renderQueries(
  model: string,
  preset: string,
  trainingPresets: string[] = [],
  options: {
    includeSearchMetadata?: boolean;
    protectedReadsEnabled?: boolean;
  } = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(
    ({
      m,
      p,
      presets,
    }: {
      m: string;
      p: string;
      presets: string[];
    }) =>
      useModelPackageMetadata(
        {
          modelPackage: { modelType: "linears", model: m },
          preset: p,
          searchPresets: presets,
        },
        options,
      ),
    {
      initialProps: { m: model, p: preset, presets: trainingPresets },
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
}

beforeEach(() => {
  mocks.fetchHealth.mockReset().mockResolvedValue({ status: "ok" });
  mocks.fetchCapabilities.mockReset().mockResolvedValue({
    authMode: "none",
    trainingEnabled: true,
    trainingCancellationCapability: "unsupported",
    logDeletionEnabled: true,
    configSnapshotsEnabled: true,
    historicalLogsEnabled: true,
    liveMonitorDataEnabled: true,
    historicalMonitorDataEnabled: true,
    uploadsEnabled: false,
    maxUploadSize: null,
  });
  mocks.fetchModels
    .mockReset()
    .mockResolvedValue({ models: [{ modelType: "linears", model: "linear" }] });
  mocks.fetchPresets.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, presets: [] }),
  );
  mocks.fetchDatasets.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({
      ...identity,
      defaultExperimentTask: "image-classification",
      datasetGroups: [
        {
          experimentTask: "image-classification",
          label: "Image Classification",
          datasets: [],
        },
      ],
    }),
  );
  mocks.fetchMonitors.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, monitors: [] }),
  );
  mocks.fetchConfigSchema.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, fields: [] }),
  );
  mocks.fetchSearchSpace.mockReset().mockImplementation(
    (identity: ModelIdentity, preset: string) =>
      Promise.resolve({ ...identity, preset, axes: [] }),
  );
});

describe("useModelPackageMetadata", () => {
  it("fires only the protected Model Package root query when no model is selected", async () => {
    renderQueries("", "");

    await waitFor(() => expect(mocks.fetchModels).toHaveBeenCalled());
    expect(mocks.fetchHealth).not.toHaveBeenCalled();
    expect(mocks.fetchCapabilities).not.toHaveBeenCalled();
    expect(mocks.fetchPresets).not.toHaveBeenCalled();
    expect(mocks.fetchDatasets).not.toHaveBeenCalled();
    expect(mocks.fetchMonitors).not.toHaveBeenCalled();
    expect(mocks.fetchConfigSchema).not.toHaveBeenCalled();
    expect(mocks.fetchSearchSpace).not.toHaveBeenCalled();
  });

  it("does not start protected metadata while the connection denies reads", () => {
    renderQueries("linear", "base", [], { protectedReadsEnabled: false });

    expect(mocks.fetchModels).not.toHaveBeenCalled();
    expect(mocks.fetchPresets).not.toHaveBeenCalled();
    expect(mocks.fetchDatasets).not.toHaveBeenCalled();
    expect(mocks.fetchMonitors).not.toHaveBeenCalled();
    expect(mocks.fetchConfigSchema).not.toHaveBeenCalled();
    expect(mocks.fetchSearchSpace).not.toHaveBeenCalled();
  });

  it("fires model-scoped queries once a model is selected, but not schema", async () => {
    renderQueries("linear", "");

    await waitFor(() =>
      expect(mocks.fetchPresets).toHaveBeenCalledWith(
        {
          modelType: "linears",
          model: "linear",
        },
        { signal: expect.any(AbortSignal) },
      ),
    );
    expect(mocks.fetchDatasets).toHaveBeenCalledWith(
      {
        modelType: "linears",
        model: "linear",
      },
      { signal: expect.any(AbortSignal) },
    );
    expect(mocks.fetchMonitors).toHaveBeenCalledWith(
      {
        modelType: "linears",
        model: "linear",
      },
      { signal: expect.any(AbortSignal) },
    );
    expect(mocks.fetchConfigSchema).not.toHaveBeenCalled();
    expect(mocks.fetchSearchSpace).not.toHaveBeenCalled();
  });

  it("fires preset-scoped queries only when both model and preset are selected", async () => {
    renderQueries("linear", "base", ["base", "post-norm"]);

    await waitFor(() =>
      expect(mocks.fetchConfigSchema).toHaveBeenCalledWith(
        { modelType: "linears", model: "linear" },
        "base",
        { signal: expect.any(AbortSignal) },
      ),
    );
    expect(mocks.fetchSearchSpace).toHaveBeenCalledWith(
      { modelType: "linears", model: "linear" },
      "base",
      ["base", "post-norm"],
      { signal: expect.any(AbortSignal) },
    );
  });

  it("can skip search-space when callers only need schema metadata", async () => {
    renderQueries("linear", "base", [], { includeSearchMetadata: false });

    await waitFor(() =>
      expect(mocks.fetchConfigSchema).toHaveBeenCalledWith(
        { modelType: "linears", model: "linear" },
        "base",
        { signal: expect.any(AbortSignal) },
      ),
    );
    expect(mocks.fetchSearchSpace).not.toHaveBeenCalled();
  });

  it("normalizes empty training preset selections to the selected preset", async () => {
    renderQueries("linear", "base");

    await waitFor(() =>
      expect(mocks.fetchSearchSpace).toHaveBeenCalledWith(
        { modelType: "linears", model: "linear" },
        "base",
        ["base"],
        { signal: expect.any(AbortSignal) },
      ),
    );
  });

  it("publishes semantic projections without exposing query machinery", async () => {
    const { result } = renderQueries("linear", "base");

    await waitFor(() => expect(result.current.runtimeDefaults.isReady).toBe(true));

    expect(result.current).toEqual({
      modelPackages: expect.objectContaining({
        records: [{ modelType: "linears", model: "linear" }],
        isReady: true,
      }),
      presets: expect.objectContaining({ records: [], isReady: true }),
      datasetMetadata: expect.objectContaining({
        defaultExperimentTask: "image-classification",
        isReady: true,
      }),
      monitorMetadata: expect.objectContaining({ records: [], isReady: true }),
      runtimeDefaults: expect.objectContaining({ fields: [], isReady: true }),
      searchMetadata: expect.objectContaining({ axes: [], isReady: true }),
    });
    for (const projection of Object.values(result.current)) {
      expect(projection).not.toHaveProperty("data");
      expect(projection).not.toHaveProperty("queryKey");
      expect(projection).not.toHaveProperty("refetch");
      expect(projection).not.toHaveProperty("status");
    }
  });
});
