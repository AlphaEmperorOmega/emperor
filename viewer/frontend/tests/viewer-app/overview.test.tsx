import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  IMPLEMENTED_FEATURES,
  installFetchMock,
  logRunsResponse,
  openDatasetSelector,
  openFullConfig,
  renderViewer,
  resetViewerAppTestState,
  schemaResponse,
  selectTargetOption,
  typeConfigFieldValue,
  waitForTargetValue,
} from "./support";

describe("ViewerApp Overview", () => {
  beforeEach(resetViewerAppTestState);

  it("renders model and preset selectors from API data", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const model = await waitForTargetValue("model", "linear");
    const preset = await waitForTargetValue("preset", "baseline");

    expect(model).toHaveTextContent("linear");
    expect(preset).toHaveTextContent("baseline");
    expect(model).toHaveAttribute("aria-expanded", "false");
    expect(preset).toHaveAttribute("aria-expanded", "false");

    await user.click(model);

    const modelOptions = await screen.findByRole("listbox", {
      name: /model options/i,
    });
    expect(model).toHaveAttribute("aria-expanded", "true");
    expect(within(modelOptions).getByRole("option", { name: "linear" }))
      .toBeInTheDocument();
    expect(within(modelOptions).getByRole("option", { name: "bert_linear" }))
      .toBeInTheDocument();

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("listbox", { name: /model options/i }))
        .not.toBeInTheDocument();
    });
    expect(model).toHaveAttribute("aria-expanded", "false");
    expect(model).toHaveFocus();

    await user.click(preset);

    const presetOptions = await screen.findByRole("listbox", {
      name: /preset options/i,
    });
    expect(preset).toHaveAttribute("aria-expanded", "true");
    expect(within(presetOptions).getByRole("option", { name: "baseline" }))
      .toBeInTheDocument();
    expect(
      within(presetOptions).getByRole("option", {
        name: "recurrent-gating-halting",
      }),
    ).toBeInTheDocument();
  });

  it("shows config status pills while keeping graph count pills out of the header", async () => {
    installFetchMock();
    renderViewer();

    const header = document.querySelector("header");
    if (!header) {
      throw new Error("Expected app header to render");
    }

    expect(within(header).getByText(/^API$/)).toBeInTheDocument();
    expect(await within(header).findByText("online")).toBeInTheDocument();
    const overrideLabel = within(header).getByText(/^overrides$/);
    const presetLabel = within(header).getByText(/^presets$/);
    expect(overrideLabel.nextElementSibling).toHaveTextContent("0");
    expect(presetLabel.nextElementSibling).toHaveTextContent("0");
    expect(within(header).queryByText(/^nodes$/i)).not.toBeInTheDocument();
    expect(within(header).queryByText(/^edges$/i)).not.toBeInTheDocument();
  });

  it("shows preset-owned field count in the top header", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: schemaResponse.fields.map((field) =>
          field.key === "gate_flag"
            ? {
                ...field,
                locked: true,
                lockedValue: true,
                lockedReason:
                  "Locked by the GATING preset because this preset enables stack gating.",
              }
            : field,
        ),
      },
    });
    renderViewer();

    const header = document.querySelector("header");
    if (!header) {
      throw new Error("Expected app header to render");
    }
    const presetLabel = within(header).getByText(/^presets$/);
    const presetPill = presetLabel.parentElement;
    if (!(presetPill instanceof HTMLElement)) {
      throw new Error("Expected presets status pill to render");
    }

    await waitFor(() => {
      expect(presetLabel.nextElementSibling).toHaveTextContent("1");
      expect(presetPill).toHaveClass("text-amber");
    });
  });

  it("supports keyboard selection and Escape on target dropdowns", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const preset = await waitForTargetValue("preset", "baseline");

    await user.click(preset);
    expect(await screen.findByRole("listbox", { name: /preset options/i }))
      .toBeInTheDocument();
    expect(preset).toHaveAttribute("aria-expanded", "true");

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("listbox", { name: /preset options/i }))
        .not.toBeInTheDocument();
    });
    expect(preset).toHaveAttribute("aria-expanded", "false");
    expect(preset).toHaveFocus();

    await user.keyboard("{ArrowDown}{Enter}");

    await waitFor(() => {
      expect(preset).toHaveTextContent("recurrent-gating-halting");
    });
    expect(preset).toHaveAttribute("aria-expanded", "false");
  });

  it("requests the initial preview for the first selected preset", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(inspectBodies[0]).toEqual({
      model: "linear",
      preset: "baseline",
      dataset: "Cifar10",
      overrides: {},
    });
  });

  it("shows the selected preset description in a compact popup", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    const trigger = screen.getByRole("button", { name: /show preset description/i });
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toBeEnabled();
    expect(screen.queryByText("Baseline")).not.toBeInTheDocument();

    await user.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /preset description/i });
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByRole("heading", { name: "baseline" })).toBeInTheDocument();
    expect(within(dialog).getByText("Baseline")).toBeInTheDocument();

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /preset description/i }))
        .not.toBeInTheDocument();
    });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveFocus();

    await user.click(trigger);
    expect(await screen.findByRole("dialog", { name: /preset description/i }))
      .toBeInTheDocument();
    await selectTargetOption(user, "preset", "recurrent-gating-halting");

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /preset description/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /show preset description/i }))
      .toHaveAttribute("aria-expanded", "false");
  });

  it("opens and closes the implemented features dialog without API side effects", async () => {
    const { inspectBodies, trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const initialInspectRequestCount = inspectBodies.length;
    const featuresButton = screen.getByRole("button", { name: /^features$/i });

    await user.click(featuresButton);

    const dialog = await screen.findByRole("dialog", { name: /implemented features/i });
    expect(within(dialog).getByText(`${IMPLEMENTED_FEATURES.length} features`))
      .toBeInTheDocument();
    expect(within(dialog).getByText("Model inspection")).toBeInTheDocument();
    expect(
      within(dialog).getByText("Graph canvas, modes, scopes, and layout"),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText("Training job creation, polling, and cancellation"),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("TensorBoard monitor data")).toBeInTheDocument();

    await user.click(
      within(dialog).getByRole("button", { name: /close implemented features/i }),
    );

    expect(screen.queryByRole("dialog", { name: /implemented features/i }))
      .not.toBeInTheDocument();
    expect(inspectBodies).toHaveLength(initialInspectRequestCount);
    expect(trainingBodies).toHaveLength(0);
  });

  it("uses the first selected dataset for preview requests", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const datasetDialog = await openDatasetSelector(user);
    await user.click(within(datasetDialog).getByLabelText(/dataset Cifar10/i));
    await user.click(within(datasetDialog).getByLabelText(/dataset Mnist/i));
    await user.click(screen.getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toMatchObject({ dataset: "Cifar10" });
    });
  });

  it("opens dataset selection from the compact sidebar trigger", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const trigger = await screen.findByRole("button", {
      name: /datasets\s+1\s*\/\s*2/i,
    });
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByLabelText(/dataset Mnist/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/dataset Cifar10/i)).not.toBeInTheDocument();

    await user.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /dataset selector/i });
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/dataset Mnist/i)).toBeChecked();
    expect(within(dialog).getByLabelText(/dataset Cifar10/i)).not.toBeChecked();

    await user.click(within(dialog).getByLabelText(/dataset Cifar10/i));

    expect(screen.getByRole("button", { name: /datasets\s+2\s*\/\s*2/i }))
      .toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^first$/i }));

    expect(screen.getByRole("button", { name: /datasets\s+1\s*\/\s*2/i }))
      .toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^all$/i }));

    expect(screen.getByRole("button", { name: /datasets\s+2\s*\/\s*2/i }))
      .toBeInTheDocument();

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /dataset selector/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /datasets\s+2\s*\/\s*2/i }))
      .toHaveFocus();
  });

  it("renders the experiments preset filter and selects a run on click", async () => {
    const { inspectBodies } = installFetchMock({
      logRunsResponse: {
        runs: [
          logRunsResponse.runs[1],
          logRunsResponse.runs[0],
          {
            ...logRunsResponse.runs[0],
            id: "bert-run",
            group: "bert_experiment",
            experiment: "bert_experiment",
            model: "bert_linear",
            relativePath:
              "bert_experiment/bert_linear/BASELINE/Mnist/ccc_20260601_030405/version_0",
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByRole("heading", { name: "Experiments" }))
      .toBeInTheDocument();
    // The preset filter is independent of the model's build preset; it defaults to
    // "All presets" and lists the presets present in the model's runs.
    const presetFilter = (
      await screen.findByRole("option", { name: "BASELINE (2)" })
    ).closest("select") as HTMLSelectElement;
    expect(presetFilter).toHaveValue("");
    expect(
      within(presetFilter).getByRole("option", { name: "All presets" }),
    ).toBeInTheDocument();

    // Both layer-data runs are listed; nothing is auto-selected.
    const newestRun = await screen.findByRole("button", {
      name: /select experiment run test_model_2 BASELINE Cifar10 2026-06-01 02:03:04/i,
    });
    const olderRun = screen.getByRole("button", {
      name: /select experiment run test_model BASELINE Mnist 2026-06-01 01:02:03/i,
    });
    expect(newestRun).toHaveAttribute("aria-pressed", "false");
    expect(olderRun).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByText("bert_experiment")).not.toBeInTheDocument();

    // Selecting a run drives the preview; selecting it again clears the selection.
    await user.click(newestRun);
    await waitFor(() => expect(newestRun).toHaveAttribute("aria-pressed", "true"));
    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Cifar10",
        overrides: {},
      });
    });

    await user.click(newestRun);
    await waitFor(() => expect(newestRun).toHaveAttribute("aria-pressed", "false"));
  });

  it("filtering by preset narrows the visible run list", async () => {
    installFetchMock({
      logRunsResponse: {
        runs: [
          {
            ...logRunsResponse.runs[0],
            id: "fast-mnist",
            experiment: "exp_alpha",
            group: "exp_alpha",
            preset: "RECURRENT_GATING_HALTING",
            dataset: "Mnist",
            timestamp: "2026-06-01 01:00:00",
            runName: "fast_20260601_010000",
            relativePath:
              "exp_alpha/linear/RECURRENT_GATING_HALTING/Mnist/fast_20260601_010000/version_0",
          },
          {
            ...logRunsResponse.runs[1],
            id: "base-cifar",
            experiment: "exp_beta",
            group: "exp_beta",
            preset: "BASELINE",
            dataset: "Cifar10",
            timestamp: "2026-06-01 03:00:00",
            runName: "base_20260601_030000",
            relativePath:
              "exp_beta/linear/BASELINE/Cifar10/base_20260601_030000/version_0",
          },
        ],
      },
      logTagsByRun: {
        "fast-mnist": ["main_model.0.model/weights/mean"],
        "base-cifar": ["main_model.0.model/weights/mean"],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const presetFilter = (
      await screen.findByRole("option", { name: "All presets" })
    ).closest("select") as HTMLSelectElement;

    const fastRunName =
      /select experiment run exp_alpha RECURRENT_GATING_HALTING Mnist 2026-06-01 01:00:00/i;
    const baseRunName =
      /select experiment run exp_beta BASELINE Cifar10 2026-06-01 03:00:00/i;

    // All presets: both runs visible.
    expect(await screen.findByRole("button", { name: fastRunName })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: baseRunName })).toBeInTheDocument();

    // Narrow to BASELINE.
    await user.selectOptions(presetFilter, "BASELINE");
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: fastRunName })).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: baseRunName })).toBeInTheDocument();

    // Narrow to the other preset.
    await user.selectOptions(presetFilter, "RECURRENT_GATING_HALTING");
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: baseRunName })).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: fastRunName })).toBeInTheDocument();

    // Back to all presets.
    await user.selectOptions(presetFilter, "");
    expect(await screen.findByRole("button", { name: baseRunName })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: fastRunName })).toBeInTheDocument();
  });

  it("selecting a historical run syncs preset and dataset, clears overrides, and refreshes preview", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    expect(screen.getByText(/1 overrides?/i)).toBeInTheDocument();

    const mnistRun = await screen.findByRole("button", {
      name: /select experiment run test_model BASELINE Mnist 2026-06-01 01:02:03/i,
    });
    await user.click(mnistRun);

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      });
    });
    expect(mnistRun).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("0 overrides")).toBeInTheDocument();
  });

});
