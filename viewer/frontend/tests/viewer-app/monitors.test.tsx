import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  buildHistoricalMonitorFixture,
  defaultLogRunMonitorPayload,
  deferred,
  installFetchMock,
  longSelectedNodeId,
  longSelectedNodeInspectResponse,
  monitorScopeInspectResponse,
  parameterShapeInspectResponse,
  renderViewer,
  repeatedLayersInspectResponse,
  resetViewerAppTestState,
  selectSearchableDropdownOption,
  selectNewTrainingLogFolder,
  selectTrainingMonitorOption,
  semanticMonitorPayload,
} from "./support";

describe("ViewerApp Monitor Charts And Errors", () => {
  beforeEach(resetViewerAppTestState);

  async function selectExperimentRun(
    user: ReturnType<typeof userEvent.setup>,
    optionName: string | RegExp,
  ) {
    void optionName;
    const experimentsTab = await screen.findByRole("radio", { name: "Experiments" });
    await waitFor(() => expect(experimentsTab).not.toBeDisabled());
    await user.click(experimentsTab);

    const experimentControl = await screen.findByRole("combobox", {
      name: "Experiment",
    });
    await user.click(experimentControl);
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^experiment options$/i }),
      ).getByRole("option", { name: /^monitor_exp/ }),
    );

    const datasetControl = await screen.findByRole("combobox", {
      name: "Dataset",
    });
    await user.click(datasetControl);
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^dataset options$/i }),
      ).getByRole("option", { name: /^Mnist/ }),
    );

    const presetControl = await screen.findByRole("combobox", {
      name: "Preset",
    });
    await user.click(presetControl);
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^preset options$/i }),
      ).getByRole("option", { name: /^BASELINE/ }),
    );
  }

  async function openSearchableDropdown(
    user: ReturnType<typeof userEvent.setup>,
    control: HTMLElement,
  ) {
    await user.click(control);
    const root = control.parentElement;

    if (!(root instanceof HTMLElement)) {
      throw new Error("Expected searchable dropdown root");
    }

    return {
      root,
      listbox: await within(root).findByRole("listbox"),
    };
  }

  it("opens selected-node monitor charts for the active training job", async () => {
    const { monitorDataRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_charts");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    expect(dialog).toBeInTheDocument();
    await waitFor(() => {
      expect(monitorDataRequests[0]).toEqual({
        jobId: "job-1",
        nodePath: "main_model.0.model",
        preset: "baseline",
        dataset: "Mnist",
      });
    });
    expect(within(dialog).queryByLabelText(/^compare$/i)).not.toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Activations\s+1 chart/i }))
      .toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("output/mean")).toBeInTheDocument();
    const visualSummaries = within(dialog).getByRole("button", {
      name: /Visual summaries\s+2 charts/i,
    });
    expect(visualSummaries).toHaveAttribute("aria-expanded", "false");
    await user.click(visualSummaries);
    expect(
      screen.getByAltText(
        "Monitor image for main_model.0.model/heatmap/usage_fraction at step 2",
      ),
    ).toBeInTheDocument();
  });

  it("groups selected-node monitor charts into semantic accordions", async () => {
    const { monitorDataRequests } = installFetchMock({
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "semantic_monitor_charts");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    await waitFor(() => {
      expect(monitorDataRequests[0]).toEqual({
        jobId: "job-1",
        nodePath: "main_model.0.model",
        preset: "baseline",
        dataset: "Mnist",
      });
    });

    const activations = within(dialog).getByRole("button", {
      name: /Activations\s+1 chart/i,
    });
    const gradients = within(dialog).getByRole("button", {
      name: /Gradients\s+1 chart/i,
    });
    const bias = within(dialog).getByRole("button", { name: /Bias\s+1 chart/i });
    const weights = within(dialog).getByRole("button", {
      name: /Weights\s+1 chart/i,
    });
    const attention = within(dialog).getByRole("button", {
      name: /Attention\s+1 chart/i,
    });
    const recurrent = within(dialog).getByRole("button", {
      name: /Recurrent\s+1 chart/i,
    });
    const controllers = within(dialog).getByRole("button", {
      name: /Controllers\s+1 chart/i,
    });
    const parametric = within(dialog).getByRole("button", {
      name: /Parametric\s+1 chart/i,
    });
    const routing = within(dialog).getByRole("button", {
      name: /Routing\s+1 chart/i,
    });
    const visualSummaries = within(dialog).getByRole("button", {
      name: /Visual summaries\s+2 charts/i,
    });

    expect(activations).toHaveAttribute("aria-expanded", "true");
    expect(gradients).toHaveAttribute("aria-expanded", "false");
    expect(bias).toHaveAttribute("aria-expanded", "false");
    expect(weights).toHaveAttribute("aria-expanded", "false");
    expect(attention).toHaveAttribute("aria-expanded", "false");
    expect(recurrent).toHaveAttribute("aria-expanded", "false");
    expect(controllers).toHaveAttribute("aria-expanded", "false");
    expect(parametric).toHaveAttribute("aria-expanded", "false");
    expect(routing).toHaveAttribute("aria-expanded", "false");
    expect(visualSummaries).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByRole("button", { name: /Other/i })).not.toBeInTheDocument();
    expect(within(dialog).getByText("input/mean")).toBeInTheDocument();
    expect(within(dialog).queryByText("bias/grad_mean")).not.toBeInTheDocument();

    await user.click(gradients);
    const gradientSection = gradients.closest("section");
    expect(gradientSection).not.toBeNull();
    expect(within(gradientSection as HTMLElement).getByText("bias/grad_mean"))
      .toBeInTheDocument();

    await user.click(bias);
    const biasSection = bias.closest("section");
    expect(biasSection).not.toBeNull();
    expect(within(biasSection as HTMLElement).getByText("bias/mean")).toBeInTheDocument();
    expect(within(biasSection as HTMLElement).queryByText("bias/grad_mean"))
      .not.toBeInTheDocument();

    await user.click(visualSummaries);
    const visualSection = visualSummaries.closest("section");
    expect(visualSection).not.toBeNull();
    expect(within(visualSection as HTMLElement).getByText("histogram/usage_fraction"))
      .toBeInTheDocument();
    expect(
      within(visualSection as HTMLElement).getByAltText(
        "Monitor image for main_model.0.model/heatmap/usage_fraction at step 2",
      ),
    ).toBeInTheDocument();
  });

  it("compares selected-node monitor charts with a numeric sibling layer", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: repeatedLayersInspectResponse,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_compare");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));
    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    expect(within(dialog).getByLabelText(/^scope$/i)).toHaveTextContent("Same stack");
    await selectSearchableDropdownOption(
      user,
      within(dialog).getByLabelText(/^compare$/i),
      "main_model.1.model",
      "main_model.1.model",
    );

    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "main_model.1.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
    expect(within(dialog).getByText("Primary layer")).toBeInTheDocument();
    expect(within(dialog).getByText("Comparison layer")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Activations\s+1 pair/i }))
      .toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getAllByText("input/mean").length).toBeGreaterThanOrEqual(2);

    const gradients = within(dialog).getByRole("button", { name: /Gradients\s+1 pair/i });
    expect(gradients).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByText("bias/grad_mean")).not.toBeInTheDocument();
    await user.click(gradients);
    const gradientSection = gradients.closest("section");
    expect(gradientSection).not.toBeNull();
    expect(within(gradientSection as HTMLElement).getByText("main_model.0.model/bias/grad_mean"))
      .toBeInTheDocument();
    expect(within(gradientSection as HTMLElement).getByText("main_model.1.model/bias/grad_mean"))
      .toBeInTheDocument();
  });

  it("switches selected-node comparison scope to input and output linear layers", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: monitorScopeInspectResponse,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_scope_select");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));
    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    const scopeSelect = within(dialog).getByLabelText(/^scope$/i);
    const compareSelect = within(dialog).getByLabelText(/^compare$/i);

    expect(scopeSelect).toHaveTextContent("Same stack");
    await selectSearchableDropdownOption(
      user,
      scopeSelect,
      "All linear layers",
      "all linear",
    );

    expect(scopeSelect).toHaveTextContent("All linear layers");
    let compareDropdown = await openSearchableDropdown(user, compareSelect);
    expect(within(compareDropdown.listbox).getByRole("option", { name: "input_model.model" }))
      .toBeInTheDocument();
    expect(within(compareDropdown.listbox).getByRole("option", { name: "output_model.model" }))
      .toBeInTheDocument();
    await user.type(
      within(compareDropdown.root).getByRole("searchbox", {
        name: /^search compare$/i,
      }),
      "input_model.model",
    );
    await user.click(
      within(compareDropdown.listbox).getByRole("option", {
        name: "input_model.model",
      }),
    );
    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "input_model.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });

    await selectSearchableDropdownOption(user, scopeSelect, "Same stack", "same");
    await waitFor(() => expect(compareSelect).toHaveTextContent("No comparison"));

    await selectSearchableDropdownOption(
      user,
      scopeSelect,
      "All linear layers",
      "all linear",
    );
    compareDropdown = await openSearchableDropdown(user, compareSelect);
    await user.type(
      within(compareDropdown.root).getByRole("searchbox", {
        name: /^search compare$/i,
      }),
      "output_model.model",
    );
    await user.click(
      within(compareDropdown.listbox).getByRole("option", {
        name: "output_model.model",
      }),
    );
    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "output_model.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
  });

  it("opens graph-card monitor charts for the wrapped linear target", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: repeatedLayersInspectResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "graph_card_monitor");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await user.click(
      await screen.findByRole("button", {
        name: /^open monitor charts for main_model\.0$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    expect(dialog).toBeInTheDocument();
    await selectSearchableDropdownOption(
      user,
      within(dialog).getByLabelText(/^compare$/i),
      "main_model.1.model",
      "main_model.1.model",
    );
    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "main_model.1.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
  });

  it("opens input/output graph-card monitor modals with all-layers scope selected", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: monitorScopeInspectResponse,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "input_output_monitor_scope");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await user.click(
      await screen.findByRole("button", {
        name: /^open monitor charts for input_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    const scopeSelect = within(dialog).getByLabelText(/^scope$/i);
    const compareSelect = within(dialog).getByLabelText(/^compare$/i);

    expect(scopeSelect).toHaveTextContent("All linear layers");
    const compareDropdown = await openSearchableDropdown(user, compareSelect);
    expect(within(compareDropdown.listbox).getByRole("option", { name: "output_model.model" }))
      .toBeInTheDocument();
    await user.type(
      within(compareDropdown.root).getByRole("searchbox", {
        name: /^search compare$/i,
      }),
      "output_model.model",
    );
    await user.click(
      within(compareDropdown.listbox).getByRole("option", {
        name: "output_model.model",
      }),
    );

    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "input_model.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "output_model.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
  });

  it("hides graph-card monitor buttons when the active job lacks the linear monitor", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Sampler usage/i);
    await selectNewTrainingLogFolder(user, "sampler_only_monitor");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /training (running|completed) mnist/i }),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByRole("button", {
        name: /^open monitor charts for main_model\.0$/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("opens selected-node monitor charts for the latest five filtered historical runs", async () => {
    const fixture = buildHistoricalMonitorFixture();
    const { logRunMonitorDataRequests } = installFetchMock({
      ...fixture,
    });
    renderViewer();
    const user = userEvent.setup();

    // Pick a run so its experiment/dataset drives the historical monitor group.
    await selectExperimentRun(
      user,
      "monitor_exp · BASELINE · Mnist · 2026-06-01 06:00:00",
    );
    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await user.click(
      await screen.findByRole("button", { name: /^select main_model\.0\.model$/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    await waitFor(() => {
      expect(logRunMonitorDataRequests).toEqual([
        { runId: "historical-06", nodePath: "main_model.0.model" },
        { runId: "historical-05", nodePath: "main_model.0.model" },
        { runId: "historical-04", nodePath: "main_model.0.model" },
        { runId: "historical-03", nodePath: "main_model.0.model" },
        { runId: "historical-02", nodePath: "main_model.0.model" },
      ]);
    });
    expect(within(dialog).getByText("historical group")).toBeInTheDocument();
    expect(within(dialog).getByText("monitor_exp")).toBeInTheDocument();
    expect(within(dialog).getByText("5 runs")).toBeInTheDocument();
    expect(
      within(dialog).getByRole("img", {
        name: /output\/mean multi-run scalar chart/i,
      }),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/monitor_run_06_20260601_060000/))
      .toBeInTheDocument();
    expect(within(dialog).getByText(/monitor_run_02_20260601_020000/))
      .toBeInTheDocument();
    expect(within(dialog).queryByText(/monitor_run_01_20260601_010000/))
      .not.toBeInTheDocument();
  });

  it("renders graph parameter activity after selecting an experiment dataset and preset", async () => {
    const fixture = buildHistoricalMonitorFixture(1);
    const { fetchMock } = installFetchMock({
      ...fixture,
      inspectResponse: parameterShapeInspectResponse,
      logParameterStatusResponse: ({ runIds }) => ({
        runs: runIds.map((runId) => ({
          sourceId: runId,
          preset: "BASELINE",
          dataset: "Mnist",
          logDir: `logs/${runId}`,
          nodes: [
            {
              nodePath: "main_model.0.model",
              weights: {
                status: "updated",
                metric: "main_model.0.model/weights/relative_delta_norm",
                lastStep: 12,
                observedPoints: 2,
              },
              bias: {
                status: "unchanged",
                metric: "main_model.0.model/bias/delta_norm",
                lastStep: 12,
                observedPoints: 1,
              },
            },
          ],
        })),
      }),
    });
    renderViewer();
    const user = userEvent.setup();

    await selectExperimentRun(
      user,
      "monitor_exp · BASELINE · Mnist · 2026-06-01 01:00:00",
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith("/logs/parameter-status"),
        ),
      ).toBe(true);
    });
    expect(
      await screen.findByLabelText("Weights parameter activity: updated"),
    ).toBeInTheDocument();
    expect(
      await screen.findByLabelText("Bias parameter activity: unchanged"),
    ).toBeInTheDocument();
  });

  it("shows a monitor path mismatch when historical status cannot attach to the graph", async () => {
    const fixture = buildHistoricalMonitorFixture(1);
    installFetchMock({
      ...fixture,
      inspectResponse: parameterShapeInspectResponse,
      logParameterStatusResponse: ({ runIds }) => ({
        runs: runIds.map((runId) => ({
          sourceId: runId,
          preset: "BASELINE",
          dataset: "Mnist",
          logDir: `logs/${runId}`,
          nodes: [
            {
              nodePath: "other_model.0.model",
              weights: {
                status: "updated",
                metric: "other_model.0.model/weights/relative_delta_norm",
                lastStep: 12,
                observedPoints: 2,
              },
              bias: {
                status: "updated",
                metric: "other_model.0.model/bias/delta_norm",
                lastStep: 12,
                observedPoints: 1,
              },
            },
          ],
        })),
      }),
    });
    renderViewer();
    const user = userEvent.setup();

    await selectExperimentRun(
      user,
      "monitor_exp · BASELINE · Mnist · 2026-06-01 01:00:00",
    );

    expect(await screen.findByText("path mismatch")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Weights parameter activity: updated"),
    ).not.toBeInTheDocument();
  });

  it("shows pending and loading parameter activity while switching experiments", async () => {
    const fixture = buildHistoricalMonitorFixture(2);
    const statusB = deferred<{
      runs: Array<{
        sourceId: string;
        preset: string;
        dataset: string;
        logDir: string;
        nodes: unknown[];
      }>;
    }>();
    const [runA, runB] = fixture.logRunsResponse.runs;
    const runs = [
      {
        ...runA,
        id: "historical-a",
        group: "monitor_exp_a",
        experiment: "monitor_exp_a",
        runName: "monitor_a_20260601_010000",
        timestamp: "2026-06-01 01:00:00",
        relativePath:
          "monitor_exp_a/linear/BASELINE/Mnist/monitor_a_20260601_010000/version_0",
      },
      {
        ...runB,
        id: "historical-b",
        group: "monitor_exp_b",
        experiment: "monitor_exp_b",
        runName: "monitor_b_20260601_020000",
        timestamp: "2026-06-01 02:00:00",
        relativePath:
          "monitor_exp_b/linear/BASELINE/Mnist/monitor_b_20260601_020000/version_0",
      },
    ];
    const statusPayload = (
      runId: string,
      weightsStatus: "updated" | "unchanged",
      biasStatus: "updated" | "unchanged",
    ) => ({
      runs: [
        {
          sourceId: runId,
          preset: "BASELINE",
          dataset: "Mnist",
          logDir: `logs/${runId}`,
          nodes: [
            {
              nodePath: "main_model.0.model",
              weights: {
                status: weightsStatus,
                metric: "main_model.0.model/weights/relative_delta_norm",
                lastStep: 12,
                observedPoints: 2,
              },
              bias: {
                status: biasStatus,
                metric: "main_model.0.model/bias/delta_norm",
                lastStep: 12,
                observedPoints: 1,
              },
            },
          ],
        },
      ],
    });
    installFetchMock({
      ...fixture,
      logRunsResponse: { runs },
      logTagsByRun: {
        "historical-a": fixture.logTagsByRun["historical-01"],
        "historical-b": fixture.logTagsByRun["historical-02"],
      },
      inspectResponse: parameterShapeInspectResponse,
      logParameterStatusResponse: ({ runIds }) => {
        if (runIds.length === 1 && runIds[0] === "historical-b") {
          return statusB.promise;
        }
        return statusPayload(runIds[0] ?? "historical-a", "updated", "unchanged");
      },
    });
    renderViewer();
    const user = userEvent.setup();
    const selectExperiment = async (name: RegExp) => {
      const experimentsTab = await screen.findByRole("radio", {
        name: "Experiments",
      });
      await waitFor(() => expect(experimentsTab).not.toBeDisabled());
      await user.click(experimentsTab);

      const experimentControl = await screen.findByRole("combobox", {
        name: "Experiment",
      });
      await user.click(experimentControl);
      await user.click(
        within(
          await screen.findByRole("listbox", { name: /^experiment options$/i }),
        ).getByRole("option", { name }),
      );
    };
    const selectDatasetAndPreset = async () => {
      const datasetControl = await screen.findByRole("combobox", {
        name: "Dataset",
      });
      await user.click(datasetControl);
      await user.click(
        within(
          await screen.findByRole("listbox", { name: /^dataset options$/i }),
        ).getByRole("option", { name: /^Mnist/ }),
      );

      const presetControl = await screen.findByRole("combobox", {
        name: "Preset",
      });
      await user.click(presetControl);
      await user.click(
        within(
          await screen.findByRole("listbox", { name: /^preset options$/i }),
        ).getByRole("option", { name: /^BASELINE/ }),
      );
    };

    await selectExperiment(/^monitor_exp_a/);
    await selectDatasetAndPreset();

    expect(
      await screen.findByLabelText("Weights parameter activity: updated"),
    ).toBeInTheDocument();
    expect(
      await screen.findByLabelText("Bias parameter activity: unchanged"),
    ).toBeInTheDocument();

    await selectExperiment(/^monitor_exp_b/);

    expect(await screen.findByText("pending")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Weights parameter activity: updated"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Bias parameter activity: unchanged"),
    ).not.toBeInTheDocument();

    await selectDatasetAndPreset();

    expect(
      await screen.findByLabelText("Weights parameter activity: loading"),
    ).toBeInTheDocument();
    expect(
      await screen.findByLabelText("Bias parameter activity: loading"),
    ).toBeInTheDocument();
    expect(screen.getByText("activity")).toBeInTheDocument();
    expect(screen.getByText("loading")).toBeInTheDocument();

    statusB.resolve(statusPayload("historical-b", "unchanged", "updated"));

    expect(
      await screen.findByLabelText("Weights parameter activity: unchanged"),
    ).toBeInTheDocument();
    expect(
      await screen.findByLabelText("Bias parameter activity: updated"),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Weights parameter activity: updated"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Bias parameter activity: unchanged"),
    ).not.toBeInTheDocument();
  });

  it("keeps performance-only historical runs from enabling graph monitor charts", async () => {
    const fixture = buildHistoricalMonitorFixture(1);
    installFetchMock({
      ...fixture,
      logTagsByRun: {
        "historical-01": {
          scalarTags: ["epoch", "train/loss", "validation/accuracy"],
          histogramTags: [],
          imageTags: ["validation/examples/predictions"],
          textTags: [],
        },
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const experimentsTab = await screen.findByRole("radio", { name: "Experiments" });
    await waitFor(() => expect(experimentsTab).not.toBeDisabled());
    await user.click(experimentsTab);
    const experimentControl = await screen.findByRole("combobox", {
      name: "Experiment",
    });
    await waitFor(() => expect(experimentControl).toBeDisabled());
    expect(screen.getByRole("combobox", { name: "Dataset" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Preset" })).toBeDisabled();

    expect(
      screen.queryByRole("button", { name: /^monitor charts$/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText(/Weights parameter activity:/i),
    ).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Bias parameter activity:/i)).not.toBeInTheDocument();
  });

  it("renders available historical monitor charts while later runs are still loading", async () => {
    const fixture = buildHistoricalMonitorFixture(5);
    const delayedRuns = new Map(
      ["historical-03", "historical-02", "historical-01"].map((runId) => [
        runId,
        deferred<ReturnType<typeof defaultLogRunMonitorPayload>>(),
      ]),
    );
    installFetchMock({
      ...fixture,
      logRunMonitorDataResponse: ({ jobId, nodePath }) => {
        const delayed = delayedRuns.get(jobId);
        if (delayed) {
          return delayed.promise;
        }
        return defaultLogRunMonitorPayload(nodePath);
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await selectExperimentRun(
      user,
      "monitor_exp · BASELINE · Mnist · 2026-06-01 05:00:00",
    );
    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await user.click(
      await screen.findByRole("button", { name: /^select main_model\.0\.model$/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    expect(
      await within(dialog).findByRole("img", {
        name: /output\/mean multi-run scalar chart/i,
      }),
    ).toBeInTheDocument();
    expect(within(dialog).queryByText("Loading monitor data")).not.toBeInTheDocument();
    expect(
      within(dialog).getByText(/Primary: loaded 2 of 5 historical runs/i),
    ).toBeInTheDocument();

    for (const delayed of delayedRuns.values()) {
      delayed.resolve(defaultLogRunMonitorPayload("main_model.0.model"));
    }

    await waitFor(() => {
      expect(
        within(dialog).queryByText(/Primary: loaded \d+ of 5 historical runs/i),
      ).not.toBeInTheDocument();
    });
  });

  it("compares historical monitor charts through log-run monitor-data requests", async () => {
    const fixture = buildHistoricalMonitorFixture(2);
    const { logRunMonitorDataRequests } = installFetchMock({
      inspectResponse: repeatedLayersInspectResponse,
      ...fixture,
    });
    renderViewer();
    const user = userEvent.setup();

    // Pick a run so its experiment/dataset drives the historical monitor group.
    await selectExperimentRun(
      user,
      "monitor_exp · BASELINE · Mnist · 2026-06-01 02:00:00",
    );
    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));
    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    await selectSearchableDropdownOption(
      user,
      within(dialog).getByLabelText(/^compare$/i),
      "main_model.1.model",
      "main_model.1.model",
    );

    await waitFor(() => {
      expect(logRunMonitorDataRequests).toEqual(
        expect.arrayContaining([
          { runId: "historical-02", nodePath: "main_model.0.model" },
          { runId: "historical-01", nodePath: "main_model.0.model" },
          { runId: "historical-02", nodePath: "main_model.1.model" },
          { runId: "historical-01", nodePath: "main_model.1.model" },
        ]),
      );
    });
    expect(within(dialog).getByText("Primary layer")).toBeInTheDocument();
    expect(within(dialog).getByText("Comparison layer")).toBeInTheDocument();
  });

  it("prefers an active linear training job over filtered historical monitor runs", async () => {
    const fixture = buildHistoricalMonitorFixture(2);
    const { monitorDataRequests, logRunMonitorDataRequests } = installFetchMock({
      ...fixture,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "active_precedence");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
    expect(logRunMonitorDataRequests).toHaveLength(0);
  });

  it("keeps long selected node ids from squeezing the path column", async () => {
    installFetchMock({ inspectResponse: longSelectedNodeInspectResponse });
    renderViewer();

    const idRow = (await screen.findAllByTitle(longSelectedNodeId)).find((element) =>
      element.classList.contains("grid"),
    );
    expect(idRow).toBeDefined();
    const resolvedIdRow = idRow as HTMLElement;
    expect(resolvedIdRow).toHaveClass("grid");
    expect(within(resolvedIdRow).getByText("ID")).toBeInTheDocument();
    expect(within(resolvedIdRow).getByText(longSelectedNodeId)).toHaveClass("truncate");

    const pathText = screen
      .getAllByText(longSelectedNodeId)
      .find((element) => element.classList.contains("break-words"));
    expect(pathText).toBeDefined();
  });

  it("renders a non-crashing API error panel", async () => {
    installFetchMock({ inspectError: true });
    renderViewer();

    expect(await screen.findByText("Preview failed")).toBeInTheDocument();
    expect(screen.getByText(/invalid override value/i)).toBeInTheDocument();
  });
});
