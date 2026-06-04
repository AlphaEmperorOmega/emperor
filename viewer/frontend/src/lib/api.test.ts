import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelTrainingJob,
  createLogRunDeletePlan,
  createTrainingJob,
  deleteLogExperiment,
  deleteLogRuns,
  fetchConfigSchema,
  fetchHealth,
  fetchLogExperiments,
  fetchLogRunMonitorData,
  fetchLogRuns,
  fetchLogScalars,
  fetchLogTags,
  fetchModels,
  fetchMonitorData,
  fetchPresets,
  fetchSearchSpace,
  fetchTrainingJob,
  fetchTrainingRunPlan,
  inspectModel,
} from "@/lib/api";

// Characterization tests: assert the CURRENT behavior of the API client
// (URL/verb/body construction and requestJson error handling) so later
// refactors can be verified green-to-green. Not aspirational.

const BASE = "http://127.0.0.1:9999";

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

function stubFetch(response: Response) {
  const fetchMock = vi.fn<FetchFn>(() => Promise.resolve(response));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("requestJson success", () => {
  it("returns parsed data and sends content-type json", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );

    const result = await fetchHealth();

    expect(result).toEqual({ status: "ok" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/health`);
    expect((init as RequestInit).headers).toMatchObject({
      "content-type": "application/json",
    });
  });
});

describe("requestJson error handling", () => {
  it("throws with request context and JSON `detail` when the response is not ok", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () => Promise.resolve({ detail: "model import failed" }),
      }),
    );

    await expect(fetchModels()).rejects.toThrow(
      `GET /models from ${BASE} failed with 400: model import failed`,
    );
  });

  it("uses statusText when the error body is not JSON", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: () => Promise.reject(new Error("not json")),
      }),
    );

    await expect(fetchModels()).rejects.toThrow(
      `GET /models from ${BASE} failed with 500: Internal Server Error`,
    );
  });

  it("throws with request context when a successful body fails schema validation", async () => {
    stubFetch(
      fakeResponse({ json: () => Promise.resolve({ models: "not-an-array" }) }),
    );

    await expect(fetchModels()).rejects.toThrow(
      `Invalid API response for GET /models from ${BASE}: models:`,
    );
  });

  it("includes nested paths in log run schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            total: 2,
            limit: 1,
            offset: 0,
            hasMore: false,
            runs: [
              {
                id: "run-1",
                group: null,
                experiment: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                timestamp: "2026-06-01 01:02:03",
                version: "version_0",
                relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
            ],
          }),
      }),
    );

    await expect(fetchLogRuns()).rejects.toThrow(
      `Invalid API response for GET /logs/runs from ${BASE}: runs.0.model:`,
    );
  });

  it("includes nested paths in scalar schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            series: [
              {
                runId: "run-1",
                tag: "train/loss",
                points: [{ step: 1, wallTime: 1000, value: "bad" }],
              },
            ],
          }),
      }),
    );

    await expect(
      fetchLogScalars({ runIds: ["run-1"], tags: ["train/loss"] }),
    ).rejects.toThrow(
      `Invalid API response for POST /logs/scalars from ${BASE}: series.0.points.0.value:`,
    );
  });

  it("includes request context in delete experiment schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiment: "test_model",
            deletedRunIds: [123],
            deletedRunCount: 1,
            deletedRelativePath: "test_model",
          }),
      }),
    );

    await expect(deleteLogExperiment("test_model")).rejects.toThrow(
      `Invalid API response for DELETE /logs/experiments/test_model from ${BASE}: deletedRunIds.0:`,
    );
  });

  it("includes nested paths in log experiment schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiments: [
              {
                experiment: "test_model",
                runCount: "one",
                relativePath: "test_model",
              },
            ],
          }),
      }),
    );

    await expect(fetchLogExperiments()).rejects.toThrow(
      `Invalid API response for GET /logs/experiments from ${BASE}: experiments.0.runCount:`,
    );
  });
});

describe("URL and query construction", () => {
  it("encodes the preset query for fetchConfigSchema", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ model: "m", fields: [] }) }),
    );

    await fetchConfigSchema("linear", "preset/one");

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/models/linear/config-schema?preset=preset%2Fone`,
    );
  });

  it("encodes the preset query for fetchSearchSpace", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            model: "linear",
            preset: "baseline",
            axes: [
              {
                key: "hidden_dim",
                configKey: "HIDDEN_DIM",
                searchKey: "SEARCH_SPACE_HIDDEN_DIM",
                label: "hidden dim",
                section: "Layer Stack Options",
                type: "int",
                values: [64, 128],
                locked: false,
                lockedValue: null,
                lockedReason: "",
              },
            ],
          }),
      }),
    );

    const result = await fetchSearchSpace("linear", "baseline/preset");

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/models/linear/search-space?preset=baseline%2Fpreset`,
    );
    expect(result.axes[0]).toMatchObject({
      key: "hidden_dim",
      values: [64, 128],
    });
  });

  it("omits the query when no preset is given", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ model: "m", fields: [] }) }),
    );

    await fetchConfigSchema("linear");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/models/linear/config-schema`);
  });

  it("interpolates the model into preset/dataset/job paths", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () => Promise.resolve({ model: "linear", presets: [] }),
      }),
    );

    await fetchPresets("linear");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/models/linear/presets`);
  });

  it("builds the monitor-data query with nodePath, preset, and dataset", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            jobId: "j1",
            nodePath: "root.block",
            preset: "baseline",
            dataset: "mnist",
            logDir: null,
            scalarSeries: [],
            histograms: [],
            images: [],
          }),
      }),
    );

    await fetchMonitorData({
      jobId: "j1",
      nodePath: "root.block",
      preset: "baseline",
      dataset: "mnist",
    });

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/j1/monitor-data?nodePath=root.block&preset=baseline&dataset=mnist`,
    );
  });

  it("builds the historical log-run monitor-data query", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            jobId: "run/1",
            nodePath: "root.block",
            dataset: "mnist",
            logDir: "logs/run",
            scalarSeries: [],
            histograms: [],
            images: [],
          }),
      }),
    );

    const result = await fetchLogRunMonitorData({
      runId: "run/1",
      nodePath: "root.block",
    });

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/logs/runs/run%2F1/monitor-data?nodePath=root.block`,
    );
    expect(result).toMatchObject({
      jobId: "run/1",
      nodePath: "root.block",
      dataset: "mnist",
    });
  });

  it("fetches log runs with the expected schema", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            total: 2,
            limit: 1,
            offset: 0,
            hasMore: false,
            runs: [
              {
                id: "run-1",
                group: null,
                experiment: "linear",
                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                timestamp: "2026-06-01 01:02:03",
                version: "version_0",
                relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
            ],
          }),
      }),
    );

    const result = await fetchLogRuns();

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/runs`);
    expect(result).toMatchObject({
      total: 2,
      limit: 1,
      offset: 0,
      hasMore: false,
    });
    expect(result.runs[0]).toMatchObject({
      id: "run-1",
      experiment: "linear",
      dataset: "Mnist",
      hasResult: false,
    });
  });

  it("fetches all log run pages with limit and offset", async () => {
    const firstRun = {
      id: "run-1",
      group: null,
      experiment: "linear",
      model: "linear",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "aaa_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      version: "version_0",
      relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
      hasResult: false,
      eventFileCount: 1,
      checkpointCount: 0,
      hasHparams: true,
      metrics: {},
    };
    const secondRun = {
      ...firstRun,
      id: "run-2",
      runName: "bbb_20260601_020304",
      relativePath: "linear/BASELINE/Mnist/bbb_20260601_020304/version_0",
    };
    const fetchMock = vi
      .fn<FetchFn>()
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 0,
              hasMore: true,
              runs: [firstRun],
            }),
        }),
      )
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 1,
              hasMore: false,
              runs: [secondRun],
            }),
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchLogRuns();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      `${BASE}/logs/runs`,
      `${BASE}/logs/runs?limit=1&offset=1`,
    ]);
    expect(result.runs.map((run) => run.id)).toEqual(["run-1", "run-2"]);
    expect(result.hasMore).toBe(false);
  });

  it("fetches log experiments with run counts", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            total: 1,
            limit: 500,
            offset: 0,
            hasMore: false,
            experiments: [
              {
                experiment: "test_model",
                runCount: 2,
                relativePath: "test_model",
              },
            ],
          }),
      }),
    );

    const result = await fetchLogExperiments();

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/experiments`);
    expect(result.total).toBe(1);
    expect(result.hasMore).toBe(false);
    expect(result.experiments[0]).toEqual({
      experiment: "test_model",
      runCount: 2,
      relativePath: "test_model",
    });
  });

  it("fetches all log experiment pages with limit and offset", async () => {
    const fetchMock = vi
      .fn<FetchFn>()
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 0,
              hasMore: true,
              experiments: [
                {
                  experiment: "test_model",
                  runCount: 1,
                  relativePath: "test_model",
                },
              ],
            }),
        }),
      )
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 1,
              hasMore: false,
              experiments: [
                {
                  experiment: "test_model_2",
                  runCount: 1,
                  relativePath: "test_model_2",
                },
              ],
            }),
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchLogExperiments();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      `${BASE}/logs/experiments`,
      `${BASE}/logs/experiments?limit=1&offset=1`,
    ]);
    expect(result.experiments.map((entry) => entry.experiment)).toEqual([
      "test_model",
      "test_model_2",
    ]);
    expect(result.hasMore).toBe(false);
  });

  it("derives log run experiments from legacy API payloads", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            runs: [
              {
                id: "run-1",
                group: null,
                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                timestamp: "2026-06-01 01:02:03",
                version: "version_0",
                relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
              {
                id: "run-2",
                group: "viewer-training/job-1",
                model: "linear",
                preset: "BASELINE",
                dataset: "Cifar10",
                runName: "bbb_20260601_020304",
                timestamp: "2026-06-01 02:03:04",
                version: "version_0",
                relativePath:
                  "viewer-training/job-1/linear/BASELINE/Cifar10/bbb_20260601_020304/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
            ],
          }),
      }),
    );

    const result = await fetchLogRuns();

    expect(result.runs.map((run) => run.experiment)).toEqual([
      "linear",
      "viewer-training",
    ]);
  });

  it("does not fallback on 404 from the configured API base", async () => {
    const fetchMock = vi.fn<FetchFn>().mockResolvedValueOnce(
      fakeResponse({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: () => Promise.resolve({ detail: "Not Found" }),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchLogRuns()).rejects.toThrow(
      `GET /logs/runs from ${BASE} failed with 404: Not Found`,
    );

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([`${BASE}/logs/runs`]);
  });

  it("deletes a log experiment with DELETE", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiment: "test_model",
            deletedRunIds: ["run-1"],
            deletedRunCount: 1,
            deletedRelativePath: "test_model",
          }),
      }),
    );

    const result = await deleteLogExperiment("test_model");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/logs/experiments/test_model`);
    expect((init as RequestInit).method).toBe("DELETE");
    expect(result.deletedRunIds).toEqual(["run-1"]);
  });

  it("encodes delete log experiment names in the path", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiment: "test model/one",
            deletedRunIds: ["run-1"],
            deletedRunCount: 1,
            deletedRelativePath: "test model/one",
          }),
      }),
    );

    await deleteLogExperiment("test model/one");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/logs/experiments/test%20model%2Fone`);
    expect((init as RequestInit).method).toBe("DELETE");
  });
});

describe("POST requests", () => {
  it("posts the inspect body as JSON", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            model: "linear",
            preset: "base",
            parameterCount: 0,
            nodes: [],
            edges: [],
          }),
      }),
    );

    const input = { model: "linear", preset: "base", overrides: {}, dataset: "mnist" };
    await inspectModel(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/inspect`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
  });

  it("posts log tag and scalar requests as JSON", async () => {
    const tagFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            runs: [
              {
                runId: "run-1",
                scalarTags: ["train/loss"],
                histogramTags: [],
                imageTags: [],
              },
            ],
          }),
      }),
    );

    await fetchLogTags({ runIds: ["run-1"] });

    expect(tagFetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/tags`);
    expect((tagFetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((tagFetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify({ runIds: ["run-1"] }),
    );

    const scalarFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            series: [
              {
                runId: "run-1",
                tag: "train/loss",
                points: [{ step: 1, wallTime: 1000, value: 0.25 }],
              },
            ],
          }),
      }),
    );

    const scalars = await fetchLogScalars({
      runIds: ["run-1"],
      tags: ["train/loss"],
    });

    expect(scalarFetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/scalars`);
    expect((scalarFetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((scalarFetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify({ runIds: ["run-1"], tags: ["train/loss"] }),
    );
    expect(scalars.series[0].points[0].value).toBe(0.25);
  });

  it("posts filtered log-run delete planning requests as JSON", async () => {
    const filters = {
      experiments: ["test_model"],
      datasets: ["Mnist"],
      models: ["linear"],
      presets: ["BASELINE"],
      runIds: ["run-1"],
    };
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            candidateCount: 1,
            counts: {
              runs: 1,
              experiments: 1,
              datasets: 1,
              models: 1,
              presets: 1,
            },
            affected: {
              experiments: ["test_model"],
              datasets: ["Mnist"],
              models: ["linear"],
              presets: ["BASELINE"],
              runIds: ["run-1"],
            },
            candidates: [
              {
                id: "run-1",
                experiment: "test_model",
                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                version: "version_0",
                relativePath:
                  "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
              },
            ],
            blockedByActiveJobs: [],
            canDelete: true,
          }),
      }),
    );

    const result = await createLogRunDeletePlan(filters);

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/runs/delete-plan`);
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((fetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify(filters),
    );
    expect(result.candidateCount).toBe(1);
    expect(result.candidates[0].relativePath).toContain("version_0");
  });

  it("posts filtered log-run delete requests as JSON", async () => {
    const filters = {
      experiments: ["test_model"],
      datasets: ["Mnist"],
      models: ["linear"],
      presets: ["BASELINE"],
      runIds: ["run-1"],
    };
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            deletedRunIds: ["run-1"],
            deletedRunCount: 1,
            deletedRelativePaths: [
              "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
            ],
            candidateCount: 1,
            counts: {
              runs: 1,
              experiments: 1,
              datasets: 1,
              models: 1,
              presets: 1,
            },
            affected: {
              experiments: ["test_model"],
              datasets: ["Mnist"],
              models: ["linear"],
              presets: ["BASELINE"],
              runIds: ["run-1"],
            },
            candidates: [
              {
                id: "run-1",
                experiment: "test_model",
                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                version: "version_0",
                relativePath:
                  "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
              },
            ],
            blockedByActiveJobs: [],
            canDelete: true,
          }),
      }),
    );

    const result = await deleteLogRuns(filters);

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/runs/delete`);
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((fetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify(filters),
    );
    expect(result.deletedRunIds).toEqual(["run-1"]);
  });

  it("posts to the cancel endpoint without a body", async () => {
    const job = {
      id: "j1",
      status: "cancelled",
      model: "linear",
      preset: "base",
      datasets: [],
      overrides: {},
      monitors: [],
      logFolder: "test_model",
      createdAt: "t",
      updatedAt: "t",
      exitCode: null,
      pid: 1,
      currentDataset: null,
      epoch: null,
      step: null,
      metrics: {},
      logDir: null,
      events: [],
      logTail: [],
      resultLinks: [],
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    await cancelTrainingJob("j1");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/training/jobs/j1/cancel`);
    expect((init as RequestInit).method).toBe("POST");
  });

  it("preserves backend detail on delete experiment failure", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () => Promise.resolve({ detail: "Unknown log experiment: missing" }),
      }),
    );

    await expect(deleteLogExperiment("missing")).rejects.toThrow(
      `DELETE /logs/experiments/missing from ${BASE} failed with 400: Unknown log experiment: missing`,
    );
  });

  it("preserves backend detail on filtered run delete failure", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () =>
          Promise.resolve({
            detail: "A training job is still writing to this log folder.",
          }),
      }),
    );

    await expect(
      deleteLogRuns({
        experiments: ["test_model"],
        datasets: ["Mnist"],
        models: ["linear"],
        presets: ["BASELINE"],
        runIds: ["run-1"],
      }),
    ).rejects.toThrow(
      `POST /logs/runs/delete from ${BASE} failed with 400: A training job is still writing to this log folder.`,
    );
  });

  it("preserves backend detail on historical monitor-data failure", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () => Promise.resolve({ detail: "Unknown log run id: missing" }),
      }),
    );

    await expect(
      fetchLogRunMonitorData({ runId: "missing", nodePath: "root.block" }),
    ).rejects.toThrow(
      `GET /logs/runs/missing/monitor-data?nodePath=root.block from ${BASE} failed with 400: Unknown log run id: missing`,
    );
  });

  it("fetches a training job by id with GET", async () => {
    const job = {
      id: "j2",
      status: "running",
      model: "linear",
      preset: "base",
      datasets: [],
      overrides: {},
      monitors: [],
      logFolder: "test_model",
      createdAt: "t",
      updatedAt: "t",
      exitCode: null,
      pid: 2,
      currentDataset: null,
      epoch: null,
      step: null,
      metrics: {},
      logDir: null,
      events: [],
      logTail: [],
      resultLinks: [],
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    await fetchTrainingJob("j2");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/training/jobs/j2`);
  });
});

describe("createTrainingJob", () => {
  it("posts the job request body as JSON", async () => {
    const job = {
      id: "j3",
      status: "queued",
      model: "linear",
      preset: "base",
      datasets: ["mnist"],
      overrides: {},
      monitors: [],
      logFolder: "test_model",
      createdAt: "t",
      updatedAt: "t",
      exitCode: null,
      pid: 3,
      currentDataset: null,
      epoch: null,
      step: null,
      metrics: {},
      logDir: null,
      events: [],
      logTail: [],
      resultLinks: [],
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    const input = {
      model: "linear",
      preset: "base",
      presets: ["base", "gating"],
      datasets: ["mnist"],
      overrides: {},
      logFolder: "test_model",
      monitors: [],
    };
    await createTrainingJob(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/training/jobs`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
  });
});

describe("fetchTrainingRunPlan", () => {
  it("posts the run plan request body as JSON", async () => {
    const plan = {
      model: "linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: {},
      search: null,
      logFolder: "test_model",
      isRandomSearch: false,
      runs: [
        {
          id: "run-0001",
          index: 1,
          status: "Pending",
          preset: "baseline",
          dataset: "Mnist",
          changes: [],
          overrides: {},
          command: "source experiment.sh linear --preset baseline --datasets Mnist",
          totalEpochs: 30,
          currentEpoch: 0,
          metrics: {},
          logDir: null,
          error: null,
        },
      ],
      summary: {
        totalRuns: 1,
        completedRuns: 0,
        runningRuns: 0,
        pendingRuns: 1,
        failedRuns: 0,
        cancelledRuns: 0,
        skippedRuns: 0,
        totalEpochs: 30,
        completedEpochs: 0,
        remainingEpochs: 30,
      },
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(plan) }));
    const input = {
      model: "linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: {},
      logFolder: "test_model",
    };

    const result = await fetchTrainingRunPlan(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/training/run-plan`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
    expect(result.summary.remainingEpochs).toBe(30);
  });
});
