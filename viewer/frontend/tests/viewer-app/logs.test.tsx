import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  buildLargeLogFixture,
  buildSubsetDeleteFixture,
  capabilitiesResponse,
  expectLogsChecklistRowSizing,
  installFetchMock,
  logMetricGroupToggle,
  logRunsResponse,
  renderViewer,
  resetViewerAppTestState,
  scalarChartGridFor,
} from "./support";

describe("ViewerApp Logs Workspace", () => {
  beforeEach(resetViewerAppTestState);

  it("switches to logs workspace and renders historical scalar runs", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();
    expect(screen.getByLabelText("Experiments test_model")).toBeChecked();
    expect(screen.getByLabelText("Experiments test_model_2")).toBeChecked();
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    expect(logMetricGroupToggle("Train")).toHaveAttribute("aria-expanded", "true");
    expect(logMetricGroupToggle("Validation")).toHaveAttribute("aria-expanded", "true");
    expect(logMetricGroupToggle("Test")).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /test\/accuracy scalar chart/i }))
      .not.toBeInTheDocument();
    const accuracyLeaderboard = await screen.findByRole("table", {
      name: /test\/accuracy test leaderboard/i,
    });
    const accuracyRows = within(accuracyLeaderboard).getAllByRole("row").slice(1);
    expect(accuracyRows).toHaveLength(2);
    expect(accuracyRows[0]).toHaveTextContent("0.9");
    expect(within(accuracyRows[0]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
    expect(accuracyRows[1]).toHaveTextContent("0.62");
    expect(within(accuracyRows[1]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model · Mnist · linear · BASELINE · 2026-06-01 01:02:03/).length,
    ).toBeGreaterThan(0);
    const cifarLine = within(accuracyLeaderboard).getByRole("button", {
      name: /open run details for test_model_2 · Cifar10 · linear · BASELINE · 2026-06-01 02:03:04/i,
    });

    await user.click(cifarLine);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" }).closest("aside");
    expect(detailsPanel).not.toBeNull();
    expect(within(detailsPanel as HTMLElement).getByText("Experiment")).toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText("test_model_2")).toBeInTheDocument();
    expect(screen.getAllByText("No result.json").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /start training/i })).not.toBeInTheDocument();
  });

  it("collapses logs metric groups without changing selected tags or refetching scalars", async () => {
    const { logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    const trainToggle = logMetricGroupToggle("Train");
    const testToggle = logMetricGroupToggle("Test");
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(1);
    });

    await user.click(trainToggle);

    expect(trainToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("img", { name: /train\/loss scalar chart/i }))
      .not.toBeInTheDocument();
    expect(screen.getByLabelText("Scalar Tags train/loss")).toBeChecked();
    expect(logScalarRequests).toHaveLength(1);

    await user.click(trainToggle);

    expect(trainToggle).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(1);

    await user.click(testToggle);

    expect(testToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
      .not.toBeInTheDocument();
    expect(screen.getByLabelText("Scalar Tags test/accuracy")).toBeChecked();
    expect(logScalarRequests).toHaveLength(1);

    await user.click(testToggle);

    expect(testToggle).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }))
      .toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(1);
  });

  it("renders non-standard scalar tags in the Other metric group", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();

    const otherTag = screen.getByLabelText("Scalar Tags main_model.0.model/weights/mean");
    expect(otherTag).not.toBeChecked();
    await user.click(otherTag);

    const otherToggle = await screen.findByRole("button", {
      name: /^Other\s+1\s+metric$/i,
    });
    expect(otherToggle).toHaveAttribute("aria-expanded", "true");
    expect(
      await screen.findByRole("img", {
        name: /main_model\.0\.model\/weights\/mean scalar chart/i,
      }),
    ).toBeInTheDocument();
  });

  it("keeps collapsed logs metric groups collapsed after switching workspaces", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();

    await user.click(logMetricGroupToggle("Train"));
    expect(screen.queryByRole("img", { name: /train\/loss scalar chart/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const trainToggle = await screen.findByRole("button", {
      name: /^Train\s+1\s+metric$/i,
    });
    expect(trainToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("img", { name: /train\/loss scalar chart/i }))
      .not.toBeInTheDocument();
    expect(screen.getByLabelText("Scalar Tags train/loss")).toBeChecked();
  });

  it("sorts test loss leaderboards ascending", async () => {
    installFetchMock({
      logTagsByRun: {
        "log-mnist": ["test/loss"],
        "log-cifar": ["test/loss"],
      },
      logScalarSeries: [
        {
          runId: "log-mnist",
          tag: "test/loss",
          points: [{ step: 3, wallTime: 1780000003, value: 0.42 }],
        },
        {
          runId: "log-cifar",
          tag: "test/loss",
          points: [{ step: 3, wallTime: 1780000003, value: 0.27 }],
        },
      ],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const lossLeaderboard = await screen.findByRole("table", {
      name: /test\/loss test leaderboard/i,
    });
    const lossRows = within(lossLeaderboard).getAllByRole("row").slice(1);
    expect(lossRows).toHaveLength(2);
    expect(lossRows[0]).toHaveTextContent("0.27");
    expect(within(lossRows[0]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();
    expect(lossRows[1]).toHaveTextContent("0.42");
    expect(within(lossRows[1]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
  });

  it("switches historical scalar chart layouts without refetching scalars", async () => {
    const { logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const chart = await screen.findByRole("img", { name: /train\/loss scalar chart/i });
    const chartGrid = scalarChartGridFor(chart);
    const layoutControl = screen.getByRole("tablist", { name: /scalar chart layout/i });
    const fullTab = within(layoutControl).getByRole("tab", { name: /^full$/i });
    const twoColumnTab = within(layoutControl).getByRole("tab", { name: /^2 col$/i });
    const threeColumnTab = within(layoutControl).getByRole("tab", { name: /^3 col$/i });

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(1);
    });
    expect(fullTab).toHaveAttribute("aria-selected", "true");
    expect(twoColumnTab).toHaveAttribute("aria-selected", "false");
    expect(threeColumnTab).toHaveAttribute("aria-selected", "false");
    expect(chartGrid).toHaveClass("grid", "gap-4");
    expect(chartGrid).not.toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");

    await user.click(twoColumnTab);

    expect(fullTab).toHaveAttribute("aria-selected", "false");
    expect(twoColumnTab).toHaveAttribute("aria-selected", "true");
    expect(chartGrid).toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(1);

    await user.click(threeColumnTab);

    expect(twoColumnTab).toHaveAttribute("aria-selected", "false");
    expect(threeColumnTab).toHaveAttribute("aria-selected", "true");
    expect(chartGrid).toHaveClass("xl:grid-cols-2", "2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: /refresh scalar charts/i }));

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(2);
    });
  });

  it("logs workspace experiment and tag checkboxes hide chart content", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(
      await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model_2 · Cifar10 · linear · BASELINE · 2026-06-01 02:03:04/).length,
    ).toBeGreaterThan(0);

    await user.click(screen.getByLabelText("Experiments test_model"));

    await waitFor(() => {
      expect(screen.queryByText(/test_model · Mnist · linear · BASELINE · 2026-06-01 01:02:03/))
        .not.toBeInTheDocument();
    });
    const datasetSection = screen.getByLabelText("Datasets Cifar10").closest("section");
    expect(datasetSection).not.toBeNull();
    expect(within(datasetSection as HTMLElement).getByText("1 / 1")).toBeInTheDocument();
    expect(screen.queryByLabelText("Datasets Mnist")).not.toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model_2 · Cifar10 · linear · BASELINE · 2026-06-01 02:03:04/).length,
    ).toBeGreaterThan(0);

    await user.click(screen.getByLabelText("Scalar Tags validation/accuracy"));

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();

    await user.click(screen.getByLabelText("Scalar Tags test/accuracy"));

    await waitFor(() => {
      expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
        .not.toBeInTheDocument();
    });

    const experimentSection = screen.getByLabelText("Experiments test_model_2").closest("section");
    expect(experimentSection).not.toBeNull();
    await user.click(within(experimentSection as HTMLElement).getByRole("button", { name: /^none$/i }));

    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
  });

  it("deletes a log experiment after exact-name confirmation", async () => {
    const { fetchMock, deleteExperimentRequests, logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByLabelText("Experiments test_model")).toBeChecked();

    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );

    const dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    expect(within(dialog).getByText(/permanently deletes 1 run/i)).toBeInTheDocument();
    const deleteButton = within(dialog).getByRole("button", {
      name: /^delete experiment$/i,
    });
    expect(deleteButton).toBeDisabled();

    await user.type(within(dialog).getByLabelText(/type experiment name/i), "test");
    expect(deleteButton).toBeDisabled();
    await user.clear(within(dialog).getByLabelText(/type experiment name/i));
    await user.type(within(dialog).getByLabelText(/type experiment name/i), "test_model");
    expect(deleteButton).toBeEnabled();

    await user.click(deleteButton);

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete experiment$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteExperimentRequests).toEqual(["test_model"]);
    const deleteCall = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/logs/experiments/test_model"),
    );
    expect(deleteCall?.[1]?.method).toBe("DELETE");
    expect(screen.queryByLabelText("Experiments test_model")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Experiments test_model_2")).toBeChecked();

    await waitFor(() => {
      expect(logScalarRequests.at(-1)?.runIds).toEqual(["log-cifar"]);
    });
  });

  it("does not render the sidebar-level delete visible runs action", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();

    expect(
      screen.queryByRole("button", { name: /^delete visible runs$/i }),
    ).not.toBeInTheDocument();
  });

  it("hides destructive log deletion actions when capabilities disable them", async () => {
    const { deleteExperimentRequests, deleteRunPlanRequests, deleteRunRequests } =
      installFetchMock({
        capabilitiesResponse: {
          ...capabilitiesResponse,
          authMode: "bearer",
          logDeletionEnabled: false,
        },
      });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByLabelText("Experiments test_model")).toBeChecked();

    expect(
      screen.queryByRole("button", { name: /^delete experiment test_model$/i }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByLabelText("Experiments test_model_2"));

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^delete dataset/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^delete preset/i }))
      .not.toBeInTheDocument();
    expect(screen.getByText(/log deletion is disabled/i)).toBeInTheDocument();
    expect(deleteExperimentRequests).toHaveLength(0);
    expect(deleteRunPlanRequests).toHaveLength(0);
    expect(deleteRunRequests).toHaveLength(0);
  });

  it("shows dataset row delete actions only for a single selected experiment", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByLabelText("Datasets Mnist")).toBeChecked();
    expect(screen.queryByRole("button", { name: /^delete dataset/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByLabelText("Experiments test_model_2"));

    expect(
      await screen.findByRole("button", {
        name: /^delete dataset Mnist from experiment test_model$/i,
      }),
    ).toBeInTheDocument();

    const experimentSection = screen.getByLabelText("Experiments test_model").closest("section");
    expect(experimentSection).not.toBeNull();
    await user.click(
      within(experimentSection as HTMLElement).getByRole("button", { name: /^none$/i }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^delete dataset/i }))
        .not.toBeInTheDocument();
    });
  });

  it("deletes dataset row runs without typed confirmation using the target run set", async () => {
    const { deleteRunPlanRequests, deleteRunRequests } = installFetchMock(
      buildSubsetDeleteFixture(),
    );
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByLabelText("Experiments test_model_2"));

    await user.click(screen.getByRole("button", { name: /presets\s+2\s*\/\s*2/i }));
    await user.click(screen.getByLabelText("Presets ALT"));
    expect(screen.getByLabelText("Presets ALT")).not.toBeChecked();

    await user.click(
      await screen.findByRole("button", {
        name: /^delete dataset Mnist from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete dataset$/i });
    expect(await within(dialog).findByText(/2 matched runs/i)).toBeInTheDocument();
    expect(within(dialog).getAllByText("test_model").length).toBeGreaterThan(0);
    expect(within(dialog).getAllByText("Mnist").length).toBeGreaterThan(0);
    expect(within(dialog).getByText("ALT")).toBeInTheDocument();
    expect(within(dialog).getByText("BASELINE")).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "test_model/linear/BASELINE/Mnist/mnist_baseline_20260601_010203/version_0",
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("version_*")).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/type/i)).not.toBeInTheDocument();

    const deleteButton = within(dialog).getByRole("button", {
      name: /^delete dataset$/i,
    });
    expect(deleteButton).toBeEnabled();
    await user.click(deleteButton);

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete dataset$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteRunPlanRequests).toEqual([
      {
        experiments: ["test_model"],
        datasets: ["Mnist"],
        models: ["linear"],
        presets: ["ALT", "BASELINE"],
        runIds: ["log-mnist-alt", "log-mnist-baseline"],
      },
    ]);
    expect(deleteRunRequests).toEqual(deleteRunPlanRequests);
    await waitFor(() => {
      expect(screen.queryByLabelText("Datasets Mnist")).not.toBeInTheDocument();
    });
  });

  it("deletes preset row runs using the target experiment and preset", async () => {
    const { deleteRunPlanRequests, deleteRunRequests } = installFetchMock(
      buildSubsetDeleteFixture(),
    );
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByLabelText("Experiments test_model_2"));
    await user.click(screen.getByLabelText("Datasets Cifar10"));
    expect(screen.getByLabelText("Datasets Cifar10")).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: /presets\s+2\s*\/\s*2/i }));
    await user.click(
      await screen.findByRole("button", {
        name: /^delete preset BASELINE from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete preset$/i });
    expect(await within(dialog).findByText(/2 matched runs/i)).toBeInTheDocument();
    expect(within(dialog).getAllByText("test_model").length).toBeGreaterThan(0);
    expect(within(dialog).getAllByText("BASELINE").length).toBeGreaterThan(0);
    expect(within(dialog).getByText("Cifar10")).toBeInTheDocument();
    expect(within(dialog).getByText("Mnist")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^delete preset$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete preset$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteRunPlanRequests).toEqual([
      {
        experiments: ["test_model"],
        datasets: ["Cifar10", "Mnist"],
        models: ["linear"],
        presets: ["BASELINE"],
        runIds: ["log-cifar-baseline", "log-mnist-baseline"],
      },
    ]);
    expect(deleteRunRequests).toEqual(deleteRunPlanRequests);
  });

  it("blocks dataset row deletion when an active training job uses an affected folder", async () => {
    const { deleteRunRequests } = installFetchMock({
      deleteLogRunsBlockers: [
        { id: "job-1", logFolder: "test_model", status: "running" },
      ],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByLabelText("Experiments test_model_2"));
    await user.click(
      await screen.findByRole("button", {
        name: /^delete dataset Mnist from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete dataset$/i });
    expect(
      await within(dialog).findByText(
        /A training job is still writing to this log folder/i,
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/job-1 · logs\/test_model/i)).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", { name: /^delete dataset$/i }),
    ).toBeDisabled();
    expect(deleteRunRequests).toHaveLength(0);
  });

  it("keeps logs sidebar checklist rows full-height with many options", async () => {
    const fixture = buildLargeLogFixture();
    const { deleteExperimentRequests, logScalarRequests } = installFetchMock(fixture);
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const firstExperiment = await screen.findByLabelText("Experiments experiment_01");
    const experimentsSection = firstExperiment.closest("section");
    if (!(experimentsSection instanceof HTMLElement)) {
      throw new Error("Expected experiments checklist to render inside a section");
    }
    expect(firstExperiment).toBeChecked();
    expectLogsChecklistRowSizing(firstExperiment);
    expectLogsChecklistRowSizing(screen.getByLabelText("Experiments experiment_42"));
    expect(screen.queryByLabelText("Experiments experiment_64")).not.toBeInTheDocument();
    expect(within(experimentsSection).getByText(/Showing 50 of 64\. Search to narrow\./i))
      .toBeInTheDocument();

    const experimentSearch = screen.getByLabelText(/^search experiments$/i);
    await user.type(experimentSearch, "64");
    const lastExperiment = screen.getByLabelText("Experiments experiment_64");
    expect(lastExperiment).toBeChecked();
    expectLogsChecklistRowSizing(lastExperiment);
    await user.clear(experimentSearch);

    const tagOption = await screen.findByLabelText("Scalar Tags custom/tag-42");
    expectLogsChecklistRowSizing(tagOption);
    await user.click(tagOption);
    expect(tagOption).toBeChecked();

    await waitFor(() => {
      expect(logScalarRequests.at(-1)).toEqual({
        runIds: fixture.logRunsResponse.runs.map((run) => run.id),
        tags: ["custom/tag-42"],
      });
    });

    await user.type(screen.getByLabelText(/^search scalar tags$/i), "tag-42");

    const filteredTagOption = screen.getByLabelText("Scalar Tags custom/tag-42");
    expect(filteredTagOption).toBeChecked();
    expectLogsChecklistRowSizing(filteredTagOption);
    expect(screen.queryByLabelText("Scalar Tags custom/tag-01")).not.toBeInTheDocument();

    await user.click(filteredTagOption);
    expect(filteredTagOption).not.toBeChecked();

    await user.click(
      screen.getByRole("button", { name: /^delete experiment experiment_01$/i }),
    );
    const dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.type(
      within(dialog).getByLabelText(/type experiment name/i),
      "experiment_01",
    );
    await user.click(
      within(dialog).getByRole("button", { name: /^delete experiment$/i }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete experiment$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteExperimentRequests).toEqual(["experiment_01"]);
    expect(screen.queryByLabelText("Experiments experiment_01")).not.toBeInTheDocument();

    const secondExperiment = screen.getByLabelText("Experiments experiment_02");
    expect(secondExperiment).toBeChecked();
    expectLogsChecklistRowSizing(secondExperiment);
    await user.click(secondExperiment);
    expect(secondExperiment).not.toBeChecked();
    expect(screen.getByLabelText("Experiments experiment_42")).toBeChecked();
  });

  it("does not delete a log experiment when the dialog is cancelled", async () => {
    const { deleteExperimentRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );
    let dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete experiment$/i }))
        .not.toBeInTheDocument();
    });

    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );
    dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.click(
      within(dialog).getByRole("button", { name: /^close delete experiment$/i }),
    );

    expect(deleteExperimentRequests).toHaveLength(0);
    expect(screen.getByLabelText("Experiments test_model")).toBeChecked();
  });

  it("keeps the delete dialog open and shows backend errors", async () => {
    const { deleteExperimentRequests } = installFetchMock({
      deleteLogExperimentError: "Refusing to delete symlink log experiment: test_model",
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );

    const dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.type(within(dialog).getByLabelText(/type experiment name/i), "test_model");
    await user.click(
      within(dialog).getByRole("button", { name: /^delete experiment$/i }),
    );

    expect(
      await within(dialog).findByText(/refusing to delete symlink log experiment/i),
    ).toBeInTheDocument();
    expect(deleteExperimentRequests).toEqual(["test_model"]);
    expect(screen.getByLabelText("Experiments test_model")).toBeChecked();
  });

  it("omits stale scalar tags from chart requests after experiment filtering", async () => {
    const { logScalarRequests } = installFetchMock({
      logTagsByRun: {
        "log-mnist": ["train/loss", "validation/accuracy"],
        "log-cifar": ["train/loss"],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    await waitFor(() => {
      expect(logScalarRequests.at(-1)).toEqual({
        runIds: ["log-mnist", "log-cifar"],
        tags: ["train/loss", "validation/accuracy"],
      });
    });

    await user.click(screen.getByLabelText("Experiments test_model"));

    await waitFor(() => {
      expect(logScalarRequests.at(-1)).toEqual({
        runIds: ["log-cifar"],
        tags: ["train/loss"],
      });
    });
    expect(screen.getByText(/1 runs · 1 selected tags/i)).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByLabelText("Experiments test_model"));

    await waitFor(() => {
      expect(screen.getByText(/2 runs · 2 selected tags/i)).toBeInTheDocument();
    });
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
  });

  it("shows the scalar empty state without scalar fetches when event runs have no tags", async () => {
    const { logScalarRequests } = installFetchMock({
      logRunsResponse: {
        runs: [
          {
            ...logRunsResponse.runs[0],
            id: "log-empty",
            eventFileCount: 1,
            hasResult: false,
            metrics: {},
          },
        ],
      },
      logTagsByRun: { "log-empty": [] },
      logScalarSeries: [],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("No TensorBoard scalars")).toBeInTheDocument();
    expect(
      screen.getByText("The selected runs do not contain scalar event data."),
    ).toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(0);
  });

});
