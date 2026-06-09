import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  capabilitiesResponse,
  commandField,
  configFieldRowFor,
  deferred,
  expandedTrainingDetails,
  expandedTrainingDetailsWithConfig,
  expandTrainingPanel,
  fullConfigSearchPopup,
  fullConfigSearchResultRow,
  inspectResponse,
  installFetchMock,
  openFullConfig,
  openTrainingFullConfig,
  openTrainingMultiSelect,
  renderViewer,
  resetViewerAppTestState,
  schemaResponse,
  searchSpaceResponse,
  selectExistingTrainingLogFolder,
  selectNewTrainingLogFolder,
  selectTargetOption,
  selectTrainingMonitorOption,
  selectTrainingTargetOption,
  setTrainingHiddenDimOverride,
  setTrainingMultiSelectOption,
  trainingFullConfigButton,
  typeConfigFieldValue,
  waitForTargetValue,
} from "./support";

describe("ViewerApp Training And Preview", () => {
  beforeEach(resetViewerAppTestState);

  it("renders preset-locked fields disabled with their reason", async () => {
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
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const closeButton = within(dialog).getByRole("button", {
      name: /close full config/i,
    });
    const headerActions = closeButton.parentElement;
    if (!(headerActions instanceof HTMLElement)) {
      throw new Error("Expected full config close button to render in the header actions");
    }
    const headerPresetBadge = within(headerActions).getByText("1 preset");
    expect(headerPresetBadge).toHaveClass("text-amber");
    expect(closeButton.previousElementSibling).toBe(headerPresetBadge);

    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides, 1 preset/i,
    });
    const layerSection = layerAccordion.closest("section");
    const gateSection = gateAccordion.closest("section");
    const layerJump = within(sectionNav).getByRole("button", {
      name: /jump to layer stack options/i,
    });
    const gateJump = within(sectionNav).getByRole("button", {
      name: /jump to gate stack options/i,
    });
    const layerNavRow = layerJump.parentElement?.parentElement;
    const gateNavRow = gateJump.parentElement?.parentElement;
    if (
      !(layerSection instanceof HTMLElement) ||
      !(gateSection instanceof HTMLElement) ||
      !(layerNavRow instanceof HTMLElement) ||
      !(gateNavRow instanceof HTMLElement)
    ) {
      throw new Error("Expected full config sections and sidebar rows to render");
    }

    expect(gateNavRow).toHaveClass(
      "border-amber/30",
      "bg-amber/[0.055]",
      "hover:bg-amber/[0.09]",
    );
    expect(within(gateNavRow).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerNavRow).not.toHaveClass("border-amber/30", "bg-amber/[0.055]");
    expect(within(layerNavRow).queryByText("1 preset")).not.toBeInTheDocument();
    expect(gateAccordion).toHaveClass("bg-amber/[0.08]", "hover:bg-amber/[0.12]");
    expect(gateSection).toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(within(gateAccordion).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerAccordion).not.toHaveClass("bg-amber/[0.08]");
    expect(layerSection).not.toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(within(layerAccordion).queryByText("1 preset")).not.toBeInTheDocument();

    const gateSwitch = within(dialog).getByRole("switch", { name: /gate flag/i });
    const gateRow = configFieldRowFor(gateSwitch);
    const presetBadge = within(gateRow).getByText("preset");

    expect(gateSwitch).toBeDisabled();
    expect(gateRow).toHaveClass("border-amber/55", "bg-amber/[0.055]");
    expect(gateRow).not.toHaveClass("border-violet/40");
    expect(presetBadge).toHaveClass("text-amber");
    expect(within(gateRow).queryByText("override")).not.toBeInTheDocument();
    expect(within(dialog).getByText(/locked by the GATING preset/i)).toBeInTheDocument();

    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });
    await user.type(search, "gate");
    const searchPopup = fullConfigSearchPopup(dialog);
    const gateSearchRow = fullConfigSearchResultRow(searchPopup, /gate flag/i);
    const searchPresetBadge = within(gateSearchRow).getByText("preset");
    const searchGateSwitch = within(gateSearchRow).getByRole("switch", {
      name: /current value/i,
    });

    expect(gateSearchRow).toHaveTextContent(/current\s*true/i);
    expect(gateSearchRow).toHaveTextContent(/default\s*false/i);
    expect(searchPresetBadge).toHaveClass("text-amber");
    expect(searchGateSwitch).toBeDisabled();
    expect(
      within(gateSearchRow).getByRole("button", { name: /reset search result override/i }),
    ).toBeDisabled();
  });

  it("expanded training panel shows the flattened setup flow in order", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const footerFieldBoxClasses = [
      "grid",
      "content-start",
      "gap-1.5",
      "rounded-[10px]",
      "border",
      "border-line",
      "bg-white/[0.018]",
      "px-2.5",
      "py-2",
    ];
    const footerFieldGridClasses = [
      "grid",
      "gap-3",
      "sm:grid-cols-2",
      "xl:grid-cols-3",
    ];
    const footerFieldHeaderClasses = [
      "flex",
      "min-h-[38px]",
      "flex-wrap",
      "items-center",
      "justify-between",
      "gap-2",
    ];
    const footerIconClasses = ["h-[15px]", "w-[15px]", "text-violet"];
    function expectBoxedField(element: HTMLElement) {
      expect(element).toHaveClass(...footerFieldBoxClasses);
    }
    function expectFieldHeader(element: HTMLElement) {
      const header = element.parentElement;
      if (!(header instanceof HTMLElement)) {
        throw new Error("Expected field heading to render inside a header row");
      }
      expect(header).toHaveClass(...footerFieldHeaderClasses);
      return header;
    }
    function closestWithClasses(element: HTMLElement, classNames: string[]) {
      let current: HTMLElement | null = element;
      while (current && current !== details) {
        const candidate = current;
        if (
          classNames.every((className) => candidate.classList.contains(className))
        ) {
          return candidate;
        }
        current = current.parentElement;
      }
      return null;
    }
    function closestFooterFieldBox(element: HTMLElement) {
      const fieldBox = closestWithClasses(element, footerFieldBoxClasses);
      if (!fieldBox) {
        throw new Error("Expected control to render inside a footer field box");
      }
      return fieldBox;
    }
    function expectHeadingIcon(label: string) {
      const heading = within(details).getByText(label);
      const icon = heading.querySelector("svg");
      if (!(icon instanceof SVGElement)) {
        throw new Error(`Expected ${label} heading to render an icon`);
      }
      expect(icon).toHaveClass(...footerIconClasses);
      return heading;
    }

    const modelHeading = expectHeadingIcon("Model");
    const presetsHeading = expectHeadingIcon("Presets");
    const modelSelector = within(details).getByRole("combobox", {
      name: /^training model$/i,
    });
    const presetSelector = within(details).getByRole("combobox", {
      name: /^presets\s+1\s*\/\s*2 selected$/i,
    });
    const datasetsHeading = expectHeadingIcon("Datasets");
    const monitorsHeading = expectHeadingIcon("Monitors");
    const gridSearchHeading = expectHeadingIcon("Grid Search");
    const searchModeControl = within(details).getByRole("tablist", {
      name: /training search mode/i,
    });
    const logFolderSelect = within(details).getByLabelText("Log experiment folder");
    const logFolderField = closestFooterFieldBox(logFolderSelect);
    const logFolderModeControl = within(details).getByRole("tablist", {
      name: /log folder mode/i,
    });
    const modelField = closestFooterFieldBox(modelSelector);
    const presetField = closestFooterFieldBox(presetSelector);
    const datasetSelector = within(details).getByRole("combobox", {
      name: /^training datasets\s+1\s*\/\s*2 selected$/i,
    });
    const datasetBox = closestFooterFieldBox(datasetSelector);
    const monitorSelector = within(details).getByRole("combobox", {
      name: /^training monitors\s+0\s*\/\s*2 selected$/i,
    });
    const monitorBox = closestFooterFieldBox(monitorSelector);
    const fullConfigButton = trainingFullConfigButton(details);
    const configAction = closestFooterFieldBox(fullConfigButton);
    const configHeading = within(configAction).getByText(/^Overrides$/);
    const configHeadingIcon = configHeading.querySelector("svg");
    if (!(configHeadingIcon instanceof SVGElement)) {
      throw new Error("Expected Config action heading to render an icon");
    }
    expect(configHeadingIcon).toHaveClass(...footerIconClasses);
    const resetButton = within(configAction).getByRole("button", { name: /^reset$/i });
    const searchBox = closestFooterFieldBox(searchModeControl);
    const fieldGrid = closestWithClasses(logFolderField, footerFieldGridClasses);
    if (!fieldGrid) {
      throw new Error("Expected setup fields to render inside the footer field grid");
    }
    const fieldGridItems = Array.from(fieldGrid.children);

    function expectBefore(before: HTMLElement, after: HTMLElement) {
      expect(
        before.compareDocumentPosition(after) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }

    expectBefore(modelHeading, modelSelector);
    expectBefore(modelSelector, presetSelector);
    expectBefore(presetsHeading, presetSelector);
    expectBefore(presetSelector, datasetsHeading);
    expectBefore(datasetsHeading, monitorsHeading);
    expectBefore(monitorsHeading, monitorSelector);
    expectBefore(monitorSelector, configHeading);
    expectBefore(configHeading, fullConfigButton);
    expectBefore(fullConfigButton, gridSearchHeading);
    expect(fieldGrid).toHaveClass(...footerFieldGridClasses);
    expect(fieldGridItems).toHaveLength(6);
    expect(fieldGridItems[0]).toContainElement(logFolderField);
    expect(fieldGridItems[1]).toContainElement(modelSelector);
    expect(fieldGridItems[2]).toContainElement(presetSelector);
    expect(fieldGridItems[3]).toContainElement(datasetSelector);
    expect(fieldGridItems[4]).toContainElement(monitorSelector);
    expect(fieldGridItems[5]).toBe(configAction);
    expect(configAction).toContainElement(fullConfigButton);
    expect(fullConfigButton).toHaveAttribute("aria-label", "Open Full Config");
    expect(fullConfigButton).toHaveTextContent(/^Config$/);
    expectBoxedField(logFolderField);
    expectBoxedField(modelField);
    expectBoxedField(presetField);
    expectBoxedField(datasetBox);
    expectBoxedField(monitorBox);
    expectBoxedField(configAction);
    expectBoxedField(searchBox);
    expectFieldHeader(modelHeading);
    expectFieldHeader(presetsHeading);
    expectFieldHeader(datasetsHeading);
    expectFieldHeader(monitorsHeading);
    expectFieldHeader(configHeading);
    expectFieldHeader(gridSearchHeading);
    expect(closestWithClasses(fullConfigButton, footerFieldBoxClasses)).toBe(configAction);
    expect(searchBox).toContainElement(gridSearchHeading);
    expect(searchBox).toContainElement(searchModeControl);
    expect(fieldGrid).not.toContainElement(searchBox);
    expect(closestFooterFieldBox(logFolderModeControl)).toBe(logFolderField);
    expect(logFolderField).toContainElement(logFolderModeControl);
    expect(logFolderField).toContainElement(
      within(logFolderField).getByRole("combobox", {
        name: "Log experiment folder",
      }),
    );
    const activeLogFolderLabel = within(logFolderField).getByText("Existing folder", {
      selector: "span",
    });
    const activeLogFolderIcon = activeLogFolderLabel.querySelector("svg");
    if (!(activeLogFolderIcon instanceof SVGElement)) {
      throw new Error("Expected active log folder field label to render an icon");
    }
    expect(activeLogFolderIcon).toHaveClass(...footerIconClasses);
    expectFieldHeader(activeLogFolderLabel);
    expect(within(details).queryByText("Experiment Setup")).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("combobox", { name: /^training preset$/i }),
    ).not.toBeInTheDocument();
    expect(modelSelector).toHaveTextContent("linear");
    expect(presetSelector).toHaveTextContent("baseline");
    expect(
      within(details).getAllByRole("combobox", { name: /^presets\b/i }),
    ).toHaveLength(1);
    expect(datasetSelector).toHaveTextContent("Mnist");
    expect(within(details).getByRole("button", { name: /^Primary only$/i }))
      .toBeInTheDocument();
    expect(monitorBox).toContainElement(monitorSelector);
    expect(within(monitorBox).getByText("0 / 2")).toBeInTheDocument();
    expect(within(monitorBox).queryByRole("button", { name: /^(all|none)$/i }))
      .not.toBeInTheDocument();
    expect(within(details).queryByLabelText(/monitor Linear layers/i))
      .not.toBeInTheDocument();
    expect(within(details).queryByText(/^Metrics$/)).not.toBeInTheDocument();
    expect(within(details).queryByText(/^Runs$/)).not.toBeInTheDocument();

    const { listbox: datasetList } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    );
    expect(within(datasetList).getByRole("option", { name: /Mnist/i }))
      .toHaveAttribute("aria-selected", "true");
    expect(within(datasetList).getByRole("option", { name: /Cifar 10/i }))
      .toHaveAttribute("aria-selected", "false");
    await user.keyboard("{Escape}");

    const allDatasetsButton = within(datasetBox).getByRole("button", { name: /^All$/i });
    const firstDatasetButton = within(datasetBox).getByRole("button", {
      name: /^First$/i,
    });
    expect(datasetBox).toContainElement(allDatasetsButton);
    expect(datasetBox).toContainElement(firstDatasetButton);
    expect(allDatasetsButton.parentElement).toBe(firstDatasetButton.parentElement);
    expect(allDatasetsButton.parentElement).toHaveClass("grid", "grid-cols-2", "gap-2");
    expect(allDatasetsButton).toHaveClass("h-9", "text-[13px]");
    expect(firstDatasetButton).toHaveClass(
      "h-9",
      "border",
      "border-line",
      "bg-white/[0.025]",
      "text-[13px]",
    );
    await user.click(allDatasetsButton);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+2\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    await user.click(firstDatasetButton);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    expect(within(details).getAllByText(/^Overrides$/)).toHaveLength(1);
    expect(within(details).getAllByText("0 overrides").length).toBeGreaterThan(0);
    expect(within(details).getAllByText("4 fields").length).toBeGreaterThan(0);
    expect(configAction).toContainElement(configHeading);
    expect(within(configAction).getByText("4 fields")).toBeInTheDocument();
    expect(within(configAction).getByText("0 overrides")).toBeInTheDocument();
    expect(resetButton).toBeDisabled();
    expect(fullConfigButton).toBeEnabled();
    expect(closestWithClasses(resetButton, [
      "edge",
      "grid",
      "gap-2",
      "rounded-card",
      "p-3",
    ])).toBeNull();
    expect(within(configAction).queryByText("Config fields")).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("combobox", { name: /search config fields/i }),
    ).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("navigation", { name: /training override sections/i }),
    ).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("button", {
        name: /layer stack options section/i,
      }),
    ).not.toBeInTheDocument();
    expect(within(details).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(details).getByRole("tab", { name: /new folder/i })).toBeInTheDocument();
    await user.click(within(details).getByRole("tab", { name: /new folder/i }));
    const newLogFolderModeControl = within(details).getByRole("tablist", {
      name: /log folder mode/i,
    });
    const newLogFolderField = closestFooterFieldBox(
      within(details).getByLabelText("New log folder"),
    );
    expectBoxedField(newLogFolderField);
    expect(closestFooterFieldBox(newLogFolderModeControl)).toBe(newLogFolderField);
    expect(newLogFolderField).toContainElement(newLogFolderModeControl);
    expect(newLogFolderField).toContainElement(
      within(newLogFolderField).getByRole("textbox", {
        name: "New log folder",
      }),
    );
    const newLogFolderLabel = within(newLogFolderField).getByText("New folder", {
      selector: "span",
    });
    const newLogFolderIcon = newLogFolderLabel.querySelector("svg");
    if (!(newLogFolderIcon instanceof SVGElement)) {
      throw new Error("Expected new log folder field label to render an icon");
    }
    expect(newLogFolderIcon).toHaveClass(...footerIconClasses);
    expectFieldHeader(newLogFolderLabel);
    const { listbox: monitorList } = await openTrainingMultiSelect(
      user,
      details,
      "Training monitors",
    );
    expect(within(monitorList).getByRole("option", { name: /Linear layers/i }))
      .toBeInTheDocument();
    expect(within(monitorList).getByRole("option", { name: /Sampler usage/i }))
      .toBeInTheDocument();
  });

  it("training setup opens the shared full config dialog and reflects popup edits", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    let dialog = await openTrainingFullConfig(user, details);

    expect(dialog).toBeInTheDocument();
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));

    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);
    expect(within(details).getByRole("button", { name: /^reset$/i })).toBeEnabled();

    await user.click(within(details).getByRole("button", { name: /^reset$/i }));

    await waitFor(() => {
      expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
      expect(within(details).getByRole("button", { name: /^reset$/i })).toBeDisabled();
    });

    dialog = await openTrainingFullConfig(user, details);

    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
  });

  it("training setup selectors update shared target state and clear overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);

    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    expect(
      within(listbox).queryByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    ).not.toBeInTheDocument();
    await user.click(
      within(details).getByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    );
    await user.keyboard("{Escape}");

    expect(await waitForTargetValue("preset", "recurrent-gating-halting"))
      .toHaveTextContent("recurrent-gating-halting");
    expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsWithConfig(user);
    let dialog = await openTrainingFullConfig(user, details);
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    });
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));

    await setTrainingHiddenDimOverride(user, details, "192");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);

    await selectTrainingTargetOption(user, "model", "bert_linear");

    expect(await waitForTargetValue("model", "bert_linear")).toHaveTextContent("bert_linear");
    expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsWithConfig(user);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("bert-baseline");
    });
    dialog = await openTrainingFullConfig(user, details);
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    });
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+1\s*\/\s*1 selected$/i,
      }),
    ).toHaveTextContent("Toy Text");
  });

  it("training panel posts selected model, preset, datasets, and overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );
    await selectNewTrainingLogFolder(user, "my_experiment");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        model: "linear",
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Mnist", "Cifar10"],
        overrides: { hidden_dim: "128" },
        logFolder: "my_experiment",
        monitors: [],
      });
      expect(trainingBodies[0]).toHaveProperty("runPlan.summary.totalRuns", 2);
      expect(trainingBodies[0]).toHaveProperty(
        "runPlan.summary.remainingEpochs",
        60,
      );
    });
  });

  it("training panel posts selected presets and multiplies planned runs", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox: presetListbox } = await openTrainingMultiSelect(
      user,
      details,
      "Presets",
    );
    const baselinePreset = within(presetListbox).getByRole("option", {
      name: /baseline/i,
    });
    expect(baselinePreset).toHaveAttribute("aria-selected", "true");
    expect(baselinePreset).toHaveAttribute("aria-disabled", "true");
    await user.click(
      within(presetListbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );

    expect(within(details).getByText("2 presets")).toBeInTheDocument();
    expect(within(details).getByText("4 planned runs")).toBeInTheDocument();

    await selectNewTrainingLogFolder(user, "multi_preset");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        model: "linear",
        preset: "baseline",
        presets: ["baseline", "recurrent-gating-halting"],
        datasets: expect.arrayContaining(["Mnist", "Cifar10"]),
        logFolder: "multi_preset",
      });
    });
  });

  it("shows planned training runs and row commands before training starts", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await expandedTrainingDetailsWithConfig(user);
    const progressButton = await screen.findByRole("button", {
      name: /0\s*\/\s*1 runs.*30 epochs left/i,
    });
    await user.click(progressButton);

    const progressDialog = await screen.findByRole("dialog", {
      name: /training progress/i,
    });
    const progressOverlay = progressDialog.parentElement;
    const progressHeader = progressDialog.querySelector("header");
    const progressBody = progressDialog.querySelector(".full-config-dialog-body");

    if (
      !(progressOverlay instanceof HTMLElement) ||
      !(progressHeader instanceof HTMLElement) ||
      !(progressBody instanceof HTMLElement)
    ) {
      throw new Error("Expected training progress dialog chrome to render");
    }

    expect(progressOverlay).toHaveClass(
      "fixed",
      "inset-0",
      "items-center",
      "justify-center",
      "bg-black/70",
      "p-3",
      "backdrop-blur-sm",
      "sm:p-6",
    );
    expect(progressOverlay.parentElement).toBe(document.body);
    expect(progressDialog).toHaveClass(
      "edge",
      "full-config-dialog-shell",
      "w-full",
      "max-w-[92rem]",
      "rounded-card",
      "max-h-[calc(100vh-1.5rem)]",
      "sm:max-h-[calc(100vh-3rem)]",
    );
    expect(progressDialog).not.toHaveClass(
      "h-full",
      "max-w-none",
      "rounded-none",
    );
    expect(progressDialog).not.toHaveClass("max-w-6xl");
    expect(progressHeader).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-header",
    );
    expect(progressBody).toHaveClass("full-config-dialog-body");
    expect(progressDialog.querySelector("footer")).not.toBeInTheDocument();
    expect(within(progressDialog).getByText("Pending")).toBeInTheDocument();
    expect(within(progressDialog).getAllByText("baseline").length).toBeGreaterThan(0);
    expect(within(progressDialog).getByText("Mnist")).toBeInTheDocument();
    expect(within(progressDialog).getByText("0 / 30")).toBeInTheDocument();

    await user.click(
      within(progressDialog).getByRole("button", { name: /command for run 1/i }),
    );
    const commandDialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --datasets Mnist",
    );
  });

  it("shows Resample in the progress popup for random search before start", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await user.click(within(details).getByRole("tab", { name: /^random$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    await user.click(
      await screen.findByRole("button", {
        name: /0\s*\/\s*2 runs.*60 epochs left/i,
      }),
    );

    const progressDialog = await screen.findByRole("dialog", {
      name: /training progress/i,
    });
    expect(
      within(progressDialog).getByRole("button", { name: /^resample$/i }),
    ).toBeInTheDocument();
  });

  it("keeps completed job progress visible after the draft config changes", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );
    await selectNewTrainingLogFolder(user, "completed_plan");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(
        screen.getByRole("button", {
          name: /2\s*\/\s*2 runs.*0 epochs left/i,
        }),
      ).toBeEnabled();
    });

    await user.click(within(details).getByRole("button", { name: /^reset$/i }));
    await waitFor(() => {
      expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    });
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
      false,
    );
    await user.keyboard("{Escape}");
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });

    const progressButton = await screen.findByRole("button", {
      name: /2\s*\/\s*2 runs.*0 epochs left/i,
    });
    await user.click(progressButton);

    const progressDialog = await screen.findByRole("dialog", {
      name: /training progress/i,
    });
    expect(within(progressDialog).getAllByText("Completed")).toHaveLength(2);
    expect(within(progressDialog).getAllByText("hidden_dim=128")).toHaveLength(2);
    expect(within(progressDialog).getByText("Cifar10")).toBeInTheDocument();
    expect(within(progressDialog).getByText("Mnist")).toBeInTheDocument();
    expect(within(progressDialog).getByText("2 runs")).toBeInTheDocument();
    expect(within(progressDialog).getByText("0 epochs left")).toBeInTheDocument();
    expect(
      within(progressDialog).queryByRole("button", { name: /^resample$/i }),
    ).not.toBeInTheDocument();
  });

  it("starts the next training run from the changed draft plan after completion", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await selectNewTrainingLogFolder(user, "first_plan");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
      expect(
        screen.getByRole("button", {
          name: /1\s*\/\s*1 runs.*0 epochs left/i,
        }),
      ).toBeEnabled();
    });

    await setTrainingHiddenDimOverride(user, details, "192");
    await selectNewTrainingLogFolder(user, "second_plan");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /start training/i }))
        .toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(2);
      expect(trainingBodies[1]).toMatchObject({
        logFolder: "second_plan",
        overrides: { hidden_dim: "192" },
      });
      expect(trainingBodies[1]).toHaveProperty(
        "runPlan.runs.0.overrides.hidden_dim",
        "192",
      );
      expect(trainingBodies[1]).toHaveProperty(
        "runPlan.runs.0.command",
        "source experiment.sh linear --preset baseline --datasets Mnist --logdir second_plan --config --hidden-dim 192",
      );
    });
  });

  it("making a selected preset primary updates the primary target and resets setup state", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    expect(within(details).getByText(/1 fixed override replaced by search axes/i))
      .toBeInTheDocument();

    const initialInspectRequestCount = inspectBodies.length;
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    expect(within(details).getByText("2 presets")).toBeInTheDocument();
    expect(
      within(listbox).queryByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    ).not.toBeInTheDocument();

    await user.click(
      await within(details).findByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    );
    await user.keyboard("{Escape}");

    expect(await waitForTargetValue("preset", "recurrent-gating-halting"))
      .toHaveTextContent("recurrent-gating-halting");
    expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(within(details).getByRole("tab", { name: /^off$/i }))
        .toHaveAttribute("aria-selected", "true");
    });
    expect(inspectBodies).toHaveLength(initialInspectRequestCount);

    await user.click(screen.getByRole("button", { name: /update preview/i }));
    await waitFor(() =>
      expect(inspectBodies).toHaveLength(initialInspectRequestCount + 1),
    );
    expect(inspectBodies.at(-1)).toEqual({
      model: "linear",
      preset: "recurrent-gating-halting",
      dataset: "Mnist",
      overrides: {},
    });
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("recurrent-gating-halting");
  });

  it("removing the current primary preset promotes another selected preset", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );

    const currentPrimary = within(listbox).getByRole("option", { name: /baseline/i });
    expect(currentPrimary).toHaveAttribute("aria-selected", "true");
    expect(currentPrimary).not.toHaveAttribute("aria-disabled", "true");
    await user.click(currentPrimary);

    expect(await waitForTargetValue("preset", "recurrent-gating-halting"))
      .toHaveTextContent("recurrent-gating-halting");
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toHaveTextContent("recurrent-gating-halting");
    });
    expect(within(listbox).getByRole("option", { name: /baseline/i }))
      .toHaveAttribute("aria-selected", "false");
  });

  it("keeps the last selected preset selected in the preset multiselect", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    const onlySelectedOption = within(listbox).getByRole("option", { name: /baseline/i });

    expect(onlySelectedOption).toHaveAttribute("aria-selected", "true");
    expect(onlySelectedOption).toHaveAttribute("aria-disabled", "true");
    await user.click(onlySelectedOption);

    expect(onlySelectedOption).toHaveAttribute("aria-selected", "true");
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+1\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("baseline");
  });

  it("keeps the last dataset selected in the dataset multiselect", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    );
    const selectedDataset = within(listbox).getByRole("option", { name: /Mnist/i });

    expect(selectedDataset).toHaveAttribute("aria-selected", "true");
    expect(selectedDataset).toHaveAttribute("aria-disabled", "true");
    await user.click(selectedDataset);

    expect(selectedDataset).toHaveAttribute("aria-selected", "true");
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+1\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("Mnist");
  });

  it("filters training multiselect options and selects matching results", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    let { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.type(
      within(details).getByRole("searchbox", { name: /^search presets$/i }),
      "recurrent",
    );

    expect(within(listbox).queryByRole("option", { name: /baseline/i }))
      .not.toBeInTheDocument();
    await user.click(
      within(listbox).getByRole("option", { name: /recurrent-gating-halting/i }),
    );
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("recurrent-gating-halting");

    await user.keyboard("{Escape}");
    ({ listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    ));
    await user.type(
      within(details).getByRole("searchbox", { name: /^search training datasets$/i }),
      "cifar",
    );

    expect(within(listbox).queryByRole("option", { name: /Mnist/i }))
      .not.toBeInTheDocument();
    await user.click(within(listbox).getByRole("option", { name: /Cifar 10/i }));
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("Cifar 10");
  });

  it("multiplies grid planned runs by selected presets and datasets", async () => {
    installFetchMock({
      searchSpaceResponse: {
        ...searchSpaceResponse,
        axes: [
          {
            ...searchSpaceResponse.axes[0],
            values: [64, 128, 256],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingMultiSelectOption(
      user,
      details,
      "Presets",
      /recurrent-gating-halting/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );

    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));

    expect(within(details).getByText("3 combinations")).toBeInTheDocument();
    expect(within(details).getAllByText("12 planned runs").length).toBeGreaterThan(0);
  });

  it("resetting overrides from training setup clears posted overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);

    await user.click(within(details).getByRole("button", { name: /^reset$/i }));

    await waitFor(() => {
      expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
      expect(within(details).getByRole("button", { name: /^reset$/i })).toBeDisabled();
    });
    const dialog = await openTrainingFullConfig(user, details);
    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    await selectNewTrainingLogFolder(user, "reset_overrides");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "reset_overrides",
        overrides: {},
      });
    });
  });

  it("starts grid search with selected axis values and omits conflicting overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await selectNewTrainingLogFolder(user, "grid_search");

    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();

    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    await user.click(within(details).getByLabelText(/^search value hidden_dim 128$/i));

    expect(within(details).getByText(/1 fixed override replaced by search axes/i))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        model: "linear",
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Mnist"],
        overrides: {},
        logFolder: "grid_search",
        monitors: [],
        search: {
          mode: "grid",
          values: { hidden_dim: [64] },
        },
      });
      expect(trainingBodies[0]).toHaveProperty("runPlan.summary.totalRuns", 1);
      expect(trainingBodies[0]).toHaveProperty(
        "runPlan.runs.0.overrides.hidden_dim",
        64,
      );
    });
  });

  it("posts random search with the configured sample count", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await selectNewTrainingLogFolder(user, "random_search");
    await user.click(within(details).getByRole("tab", { name: /^random$/i }));
    expect(within(details).getByLabelText(/^random search samples$/i)).toHaveValue(10);

    await user.click(within(details).getByLabelText(/^search axis stack_activation$/i));
    const samplesInput = within(details).getByLabelText(/^random search samples$/i);
    await user.clear(samplesInput);
    await user.type(samplesInput, "7");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "random_search",
        search: {
          mode: "random",
          values: { stack_activation: ["RELU", "GELU"] },
          randomSamples: 7,
        },
      });
    });
  });

  it("selects all search axes and values from the grid setup", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await selectNewTrainingLogFolder(user, "all_axes_search");
    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByRole("button", { name: /^all axes$/i }));

    expect(within(details).getByText("3 axes")).toBeInTheDocument();
    expect(within(details).getByText("8 combinations")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "all_axes_search",
        search: {
          mode: "grid",
          values: {
            hidden_dim: [64, 128],
            stack_num_layers: [2, 4],
            stack_activation: ["RELU", "GELU"],
          },
        },
      });
    });
  });

  it("confirms large grid searches before posting", async () => {
    const largeSearchSpace = {
      ...searchSpaceResponse,
      axes: [
        {
          ...searchSpaceResponse.axes[0],
          values: Array.from({ length: 11 }, (_, index) => index + 1),
        },
        {
          ...searchSpaceResponse.axes[1],
          values: Array.from({ length: 10 }, (_, index) => index + 1),
        },
      ],
    };
    const { trainingBodies } = installFetchMock({
      searchSpaceResponse: largeSearchSpace,
    });
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await selectNewTrainingLogFolder(user, "large_grid_search");
    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByRole("button", { name: /^all axes$/i }));

    expect(within(details).getAllByText("110 planned runs").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /start training/i }));

    const dialog = await screen.findByRole("dialog", { name: /confirm grid search/i });
    expect(trainingBodies).toHaveLength(0);
    expect(within(dialog).getByText(/110 training runs/i)).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /confirm grid search/i }))
        .not.toBeInTheDocument();
    });
    expect(trainingBodies).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: /start training/i }));
    await user.click(
      within(await screen.findByRole("dialog", { name: /confirm grid search/i }))
        .getByRole("button", { name: /start training/i }),
    );

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "large_grid_search",
        search: {
          mode: "grid",
        },
      });
    });
  });

  it("training panel posts selected monitor names", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_run");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "monitor_run",
        monitors: ["linear"],
      });
    });
    expect(screen.getAllByText(/1 monitors/i).length).toBeGreaterThan(0);
  });

  it("requires a valid log folder before starting training", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const startButton = await screen.findByRole("button", { name: /start training/i });
    expect(startButton).toBeDisabled();

    await expandTrainingPanel(user);
    await user.click(screen.getByRole("tab", { name: /new folder/i }));
    const input = screen.getByLabelText(/^new log folder$/i);

    for (const value of [
      "my experiment",
      "my-experiment",
      "my.folder",
      "my/folder",
      "_my_folder",
      "my__folder",
    ]) {
      await user.clear(input);
      await user.type(input, value);
      expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();
      expect(screen.getByRole("alert")).toHaveTextContent(/single underscores/i);
    }

    await user.clear(input);
    await user.type(input, "my_experiment");
    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "my_experiment" });
    });
  });

  it("disables training planning and submission when capabilities disable training", async () => {
    const { fetchMock, trainingBodies } = installFetchMock({
      capabilitiesResponse: {
        ...capabilitiesResponse,
        authMode: "bearer",
        trainingEnabled: false,
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const startButton = await screen.findByRole("button", { name: /start training/i });
    expect(startButton).toBeDisabled();

    const details = await expandedTrainingDetails(user);
    expect(within(details).getByText(/training is disabled/i)).toBeInTheDocument();

    await user.click(within(details).getByRole("tab", { name: /new folder/i }));
    await user.type(within(details).getByLabelText(/^new log folder$/i), "hosted_disabled");

    expect(startButton).toBeDisabled();
    expect(trainingBodies).toHaveLength(0);
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/training/run-plan")))
      .toBe(false);
  });

  it("keeps Start Training disabled when the selected model has no datasets", async () => {
    const { trainingBodies } = installFetchMock({
      datasetsResponse: { model: "linear", datasets: [] },
    });
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetails(user);
    expect(within(details).getByText("No datasets for this model")).toBeInTheDocument();

    await user.click(within(details).getByRole("tab", { name: /new folder/i }));
    await user.type(within(details).getByLabelText(/^new log folder$/i), "no_dataset_run");

    expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();
    expect(trainingBodies).toHaveLength(0);
  });

  it("posts the selected existing log folder", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectExistingTrainingLogFolder(user, "test_model_2");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "test_model_2" });
    });
  });

  it("checks the started experiment when switching to logs", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectNewTrainingLogFolder(user, "fresh_run");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "fresh_run" });
    });

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByLabelText("Experiments fresh_run")).toBeChecked();
  });

  it("Update Preview sends a new inspect request for the same selection", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const initialRequestCount = inspectBodies.length;

    await user.click(screen.getByRole("button", { name: /update preview/i }));
    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 1));

    await user.click(screen.getByRole("button", { name: /update preview/i }));
    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 2));
  });

  it("changing presets clears overrides without auto-updating the preview", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    const initialRequestCount = inspectBodies.length;

    await selectTargetOption(user, "preset", "recurrent-gating-halting");

    expect(screen.getByText("0 overrides")).toBeInTheDocument();
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
    expect(inspectBodies).toHaveLength(initialRequestCount);

    await user.click(screen.getByRole("button", { name: /update preview/i }));

    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 1));
    expect(inspectBodies.at(-1)).toEqual({
      model: "linear",
      preset: "recurrent-gating-halting",
      dataset: "Mnist",
      overrides: {},
    });
  });

  it("resetting overrides refreshes the preview when a target is selected", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const requestCountBeforeReset = inspectBodies.length;

    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));

    await waitFor(() => expect(inspectBodies).toHaveLength(requestCountBeforeReset + 1));
    expect(inspectBodies.at(-1)).toEqual({
      model: "linear",
      preset: "baseline",
      dataset: "Mnist",
      overrides: {},
    });
  });

  it("clears the displayed graph while a preview refresh is pending", async () => {
    const nextPreview = deferred<unknown>();
    installFetchMock({
      logRunsResponse: { runs: [] },
      inspectResponseFactory: (requestIndex) =>
        requestIndex === 1 ? nextPreview.promise : inspectResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(screen.queryByText("main_model.0")).not.toBeInTheDocument();
    });
    expect(screen.getByText("building")).toBeInTheDocument();

    nextPreview.resolve(inspectResponse);

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
  });

});
