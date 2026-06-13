import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  commandField,
  configFieldGridFor,
  configFieldRowFor,
  expectFullConfigSectionGrid,
  expectResponsiveConfigFieldGrid,
  fullConfigSearchPopup,
  fullConfigSearchResultRow,
  fullConfigSectionFor,
  fullConfigSectionGridFor,
  fullConfigSectionNavRowFor,
  installFetchMock,
  openFullConfig,
  openTrainingCommand,
  renderViewer,
  resetViewerAppTestState,
  schemaResponse,
  scrollIntoViewMock,
  selectTargetOption,
  typeConfigFieldValue,
  waitForOpenFullConfigButton,
  waitForTargetValue,
} from "./support";

describe("ViewerApp Full Config", () => {
  beforeEach(resetViewerAppTestState);

  it("keeps full config controls out of the left sidebar", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /open full config/i }))
      .not.toBeInTheDocument();
    expect(await waitForOpenFullConfigButton(user)).toBeInTheDocument();
    expect(screen.queryByText("Sections")).not.toBeInTheDocument();
    expect(screen.queryByText("Fields")).not.toBeInTheDocument();
    expect(screen.queryByText("Changed")).not.toBeInTheDocument();
    expect(screen.queryByText("Layer Stack Options")).not.toBeInTheDocument();
    expect(screen.queryByText("Gate Stack Options")).not.toBeInTheDocument();
    expect(screen.queryByText("Modified Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
  });

  it("keeps the global snapshot library out of the model sidebar", async () => {
    installFetchMock({
      configSnapshotsResponse: { model: "linear", snapshots: [] },
      configSnapshotLibraryResponse: {
        snapshots: [
          {
            id: "bert-snapshot",
            model: "bert_linear",
            preset: "bert-baseline",
            name: "Bert tuned",
            overrides: { hidden_dim: "128" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderViewer();

    await waitForTargetValue("preset", "baseline");
    expect(screen.getByText(/^Target$/)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Snapshots" }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("Bert tuned")).not.toBeInTheDocument();
  });

  it("opens the full config popup with section accordions expanded by default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const dialogHeader = dialog.querySelector("header");
    const dialogBody = dialog.querySelector(".full-config-dialog-body");
    const dialogFooter = dialog.querySelector("footer");
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides/i,
    });
    const layerNavToggle = within(sectionNav).getByRole("button", {
      name: /close layer stack options/i,
    });
    const gateNavToggle = within(sectionNav).getByRole("button", {
      name: /close gate stack options/i,
    });
    const layerSection = layerAccordion.closest("section");
    const gateSection = gateAccordion.closest("section");
    const sectionGrid = fullConfigSectionGridFor(layerAccordion);
    const closeButton = within(dialog).getByRole("button", {
      name: /close full config/i,
    });

    if (
      !(dialogHeader instanceof HTMLElement) ||
      !(dialogBody instanceof HTMLElement) ||
      !(dialogFooter instanceof HTMLElement)
    ) {
      throw new Error("Expected full config dialog chrome to render");
    }

    expect(dialog).toHaveClass("edge", "full-config-dialog-shell");
    expect(dialogHeader).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-header",
      "border-line-soft",
    );
    expect(dialogHeader).not.toHaveClass("bg-panel/85");
    expect(dialogHeader).not.toHaveClass("border-line");
    expect(dialogBody).toHaveClass("full-config-dialog-body");
    expect(dialogBody).not.toHaveClass("bg-bg-2/80");
    expect(dialogFooter).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-footer",
      "border-line-soft",
    );
    expect(dialogFooter).not.toHaveClass("bg-panel/85");
    expect(dialogFooter).not.toHaveClass("border-line");
    expect(closeButton).toHaveClass("border-line-soft", "bg-white/[0.025]");
    expect(closeButton).not.toHaveClass("bg-white/[0.035]");
    expect(sectionGrid).toBe(fullConfigSectionGridFor(gateAccordion));
    expectFullConfigSectionGrid(sectionGrid);
    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(layerAccordion).toHaveAttribute("aria-controls");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-controls");
    expect(layerAccordion).toHaveClass("overflow-hidden", "bg-white/[0.055]");
    expect(gateAccordion).toHaveClass("overflow-hidden", "bg-white/[0.055]");
    expect(layerSection).toHaveClass(
      "overflow-hidden",
      "rounded-[12px]",
      "border",
      "border-line",
      "bg-panel/80",
      "shadow-[0_16px_40px_-30px_rgba(0,0,0,0.95)]",
    );
    expect(gateSection).toHaveClass(
      "overflow-hidden",
      "rounded-[12px]",
      "border",
      "border-line",
      "bg-panel/80",
      "shadow-[0_16px_40px_-30px_rgba(0,0,0,0.95)]",
    );
    expect(layerNavToggle).toHaveAttribute("aria-expanded", "true");
    expect(layerNavToggle).toHaveAttribute("aria-controls");
    expect(gateNavToggle).toHaveAttribute("aria-expanded", "true");
    expect(gateNavToggle).toHaveAttribute("aria-controls");
    expect(layerAccordion).not.toHaveTextContent(/3 fields|0 overrides/i);
    expect(within(layerAccordion).getByLabelText("3 fields")).not.toHaveAttribute("tabindex");
    expect(within(layerAccordion).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
    expect(within(gateAccordion).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(within(gateAccordion).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
    expect(within(dialog).getByLabelText("4 fields")).toHaveTextContent("4");
    expect(within(dialog).getByLabelText("4 fields")).not.toHaveTextContent("4 fields");
    const layerJump = within(sectionNav).getByRole("button", {
      name: /jump to layer stack options/i,
    });
    const gateJump = within(sectionNav).getByRole("button", {
      name: /jump to gate stack options/i,
    });
    const layerNavRow = layerJump.parentElement?.parentElement;
    expect(layerNavRow).toHaveClass("group/section-row");
    expect(layerNavRow).toHaveClass("focus-within:ring-2", "hover:bg-violet/10");
    expect(layerJump).not.toHaveClass("peer/title", "group/title");
    expect(layerJump).not.toHaveClass("pr-[7.75rem]");
    expect(gateJump).not.toHaveClass("pr-[7.75rem]");
    expect(within(sectionNav).getByLabelText("3 fields")).toHaveTextContent("3");
    expect(within(sectionNav).getByLabelText("3 fields")).not.toHaveAttribute("tabindex");
    within(sectionNav)
      .getAllByLabelText("0 overrides")
      .forEach((metric) => expect(metric).not.toHaveAttribute("tabindex"));
    expect(layerJump).not.toHaveTextContent(/3 fields|0 overrides/i);
    expect(within(sectionNav).getByLabelText("1 field")).toHaveTextContent("1");
    expect(within(sectionNav).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(gateJump).not.toHaveTextContent(/1 field|0 overrides/i);
    const hiddenDimControl = within(dialog).getByLabelText(/hidden dim/i);
    const gateSwitch = within(dialog).getByRole("switch", { name: /gate flag/i });

    expect(hiddenDimControl).toBeInTheDocument();
    expectResponsiveConfigFieldGrid(configFieldGridFor(hiddenDimControl));
    expect(hiddenDimControl).toHaveClass("h-10", "px-3", "py-2");
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(gateSwitch).toBeInTheDocument();
    expect(gateSwitch.parentElement).toHaveClass("h-10", "px-3");
  });

  it("shows an accessible full config field search", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    expect(search).toHaveAttribute("aria-expanded", "false");
    expect(search).toHaveAttribute("aria-controls");
    expect(search).toHaveAttribute("aria-haspopup", "dialog");
    expect(search).not.toHaveAttribute("aria-activedescendant");

    await user.type(search, "hidden");

    expect(search).toHaveAttribute("aria-expanded", "true");
    expect(search).not.toHaveAttribute("aria-activedescendant");
    const searchPopup = fullConfigSearchPopup(dialog);
    const hiddenDimRow = fullConfigSearchResultRow(searchPopup, /hidden dim/i);

    expect(hiddenDimRow).toHaveTextContent(/default\s*256/i);
    expect(hiddenDimRow).not.toHaveTextContent(/current\s*256/i);
    expect(within(hiddenDimRow).getByRole("button", { name: /hidden dim/i }))
      .toBeInTheDocument();
    expect(
      within(hiddenDimRow).getByRole("spinbutton", { name: /current value/i }),
    ).toHaveValue(256);
  });

  it("filters full config cards and sidebar sections while typing", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");

    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", {
        name: /layer stack options section, 1 field, 0 overrides/i,
      }),
    ).toBeInTheDocument();
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });
    const sectionGrid = fullConfigSectionGridFor(layerAccordion);
    expectFullConfigSectionGrid(sectionGrid);
    expect(sectionGrid.children).toHaveLength(1);
    expect(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to gate stack options/i }),
    ).not.toBeInTheDocument();
    expect(within(sectionNav).getByLabelText("1 field")).toHaveTextContent("1");
  });

  it("finds full config fields by flag and key", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "--gate-flag");

    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "gate_flag");

    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", {
        name: /gate stack options section, 1 field, 0 overrides/i,
      }),
    ).toBeInTheDocument();
  });

  it("does not match full config fields by current or default value text", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "256");
    const searchPopup = fullConfigSearchPopup(dialog);

    expect(within(searchPopup).getByText("No matching fields")).toBeInTheDocument();
    expect(
      within(searchPopup).queryByRole("group", { name: /hidden dim/i }),
    ).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(sectionNav).getByText("No matching sections")).toBeInTheDocument();
  });

  it("selects a dropdown field and filters to exactly that field", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "activation");
    const searchPopup = fullConfigSearchPopup(dialog);
    const stackActivationRow = fullConfigSearchResultRow(searchPopup, /stack activation/i);
    await user.click(
      within(stackActivationRow).getByRole("button", { name: /stack activation/i }),
    );

    expect(search).toHaveValue("stack activation");
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();
    expect(
      within(dialog).queryByRole("dialog", { name: /matching config fields/i }),
    ).not.toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("edits a numeric field from full config search results", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const searchPopup = fullConfigSearchPopup(dialog);
    const hiddenDimRow = fullConfigSearchResultRow(searchPopup, /hidden dim/i);
    const hiddenDimSearchInput = within(hiddenDimRow).getByRole("spinbutton", {
      name: /current value/i,
    });

    expect(hiddenDimRow).toHaveTextContent(/default\s*256/i);
    expect(hiddenDimRow).not.toHaveTextContent(/current\s*256/i);

    await user.clear(hiddenDimSearchInput);
    await user.type(hiddenDimSearchInput, "128");

    expect(fullConfigSearchPopup(dialog)).toBeInTheDocument();
    expect(hiddenDimRow).toHaveTextContent(/current\s*128/i);
    expect(hiddenDimRow).toHaveTextContent(/default\s*256/i);
    expect(within(hiddenDimRow).getByText("override")).toHaveClass("text-violet");
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(128);

    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: { hidden_dim: "128" },
      });
    });
  });

  it("edits enum and bool fields from full config search results", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "activation");
    let searchPopup = fullConfigSearchPopup(dialog);
    const stackActivationRow = fullConfigSearchResultRow(searchPopup, /stack activation/i);
    const stackActivationSelect = within(stackActivationRow).getByRole("combobox", {
      name: /current value/i,
    });

    await user.selectOptions(stackActivationSelect, "RELU");

    expect(stackActivationRow).toHaveTextContent(/current\s*RELU/i);
    expect(within(dialog).getByLabelText(/stack activation/i)).toHaveValue("RELU");

    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));
    await user.type(search, "gate");
    searchPopup = fullConfigSearchPopup(dialog);
    const gateFlagRow = fullConfigSearchResultRow(searchPopup, /gate flag/i);
    const gateFlagSwitch = within(gateFlagRow).getByRole("switch", {
      name: /current value/i,
    });

    await user.click(gateFlagSwitch);

    expect(gateFlagRow).toHaveTextContent(/current\s*true/i);
    expect(within(dialog).getByRole("switch", { name: /gate flag/i }))
      .toHaveAttribute("aria-checked", "true");
    expect(within(dialog).getAllByLabelText("2 overrides").length).toBeGreaterThan(0);
  });

  it("clears full config search and restores all sections and fields", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));

    expect(search).toHaveValue("");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", { name: /jump to gate stack options/i }),
    ).toBeInTheDocument();
  });

  it("shows empty states when full config search has no matches", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "missing option");

    const noResults = within(dialog).getByText('No config fields match "missing option".');
    expect(noResults).toBeInTheDocument();
    expect(noResults).not.toHaveClass("md:col-span-2");
    expect(noResults).not.toHaveClass("2xl:col-span-3");
    const sectionGrid = noResults.parentElement;
    expect(sectionGrid).toBeInstanceOf(HTMLElement);
    expectFullConfigSectionGrid(sectionGrid as HTMLElement);
    expect(within(sectionNav).getByText("No matching sections")).toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to layer stack options/i }),
    ).not.toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to gate stack options/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps sidebar jumps and toggles working while full config search is filtered", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });

    await user.click(within(sectionNav).getByRole("button", { name: /^close all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();

    await user.click(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("keeps filtered accordion metric pills non-tabbable and number-only", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });

    expect(layerAccordion).not.toHaveTextContent(/1 field|0 overrides/i);
    expect(within(layerAccordion).getByLabelText("1 field")).toHaveTextContent("1");
    expect(within(layerAccordion).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(within(layerAccordion).getByLabelText("0 overrides")).toHaveTextContent("0");
    expect(within(layerAccordion).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
  });

  it("shows popup config metric tooltips on hover and focus", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const fieldMetric = within(dialog).getByLabelText("4 fields");
    const overrideMetric = within(dialog).getAllByLabelText("0 overrides")[0];

    expect(fieldMetric).toHaveAttribute("tabindex", "0");
    expect(overrideMetric).toHaveAttribute("tabindex", "0");

    await user.click(fieldMetric);
    expect(within(dialog).getByRole("tooltip")).toHaveTextContent("Fields");

    await user.click(within(dialog).getByLabelText(/hidden dim/i));
    expect(within(dialog).queryByRole("tooltip")).not.toBeInTheDocument();

    await user.hover(overrideMetric);
    expect(within(dialog).getByRole("tooltip")).toHaveTextContent("Overrides");

    await user.unhover(overrideMetric);
    expect(within(dialog).queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("global section toggle closes and opens every popup config accordion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides/i,
    });

    const closeAllButton = within(sectionNav).getByRole("button", { name: /^close all$/i });
    expect(closeAllButton).toBeInTheDocument();
    expect(closeAllButton).toHaveClass("whitespace-nowrap", "min-w-[5.75rem]");

    await user.click(closeAllButton);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(sectionNav).getByRole("button", { name: /^open all$/i })).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(sectionNav).getByRole("button", { name: /^close all$/i })).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("global section toggle opens every popup config accordion from a partial state", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides/i,
    });

    await user.click(layerAccordion);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(sectionNav).getByRole("button", { name: /^open all$/i })).toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(sectionNav).getByRole("button", { name: /^close all$/i })).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("collapsing a popup config section hides its field controls", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });

    await user.click(layerAccordion);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(layerAccordion).toHaveClass("bg-white/[0.025]");
    expect(layerAccordion.closest("section")).toHaveClass(
      "rounded-[12px]",
      "border-line-soft",
      "bg-panel/70",
    );
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("sidebar section clicks reopen collapsed sections and scroll to them", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await user.click(layerAccordion);
    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");

    await user.click(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("sidebar section triggers toggle their matching accordion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await user.click(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(
      within(sectionNav).getByRole("button", { name: /open layer stack options/i }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(scrollIntoViewMock).not.toHaveBeenCalled();

    await user.click(
      within(sectionNav).getByRole("button", { name: /open layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).not.toHaveBeenCalled();
  });

  it("editing a popup field updates overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    const hiddenDimInput = await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const hiddenDimRow = configFieldRowFor(hiddenDimInput);
    const overrideBadge = within(hiddenDimRow).getByText("override");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 1 override/i,
    });
    const layerSection = fullConfigSectionFor(layerAccordion);
    const layerNavRow = fullConfigSectionNavRowFor(
      sectionNav,
      /jump to layer stack options/i,
    );

    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
    expect(within(dialog).queryByText("1 override")).not.toBeInTheDocument();
    expect(layerAccordion).toHaveClass("bg-violet/[0.08]", "hover:bg-violet/[0.12]");
    expect(within(layerAccordion).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerAccordion).queryByText("1 preset")).not.toBeInTheDocument();
    expect(layerSection).toHaveClass("border-violet/35", "bg-violet/[0.06]");
    expect(layerSection).not.toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(layerNavRow).toHaveClass(
      "border-violet/30",
      "bg-violet/[0.055]",
      "hover:bg-violet/15",
    );
    expect(layerNavRow).not.toHaveClass("border-amber/30", "bg-amber/[0.055]");
    expect(within(layerNavRow).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerNavRow).queryByText("1 preset")).not.toBeInTheDocument();
    expect(overrideBadge).toBeInTheDocument();
    expect(overrideBadge).toHaveClass("text-violet");
    expect(hiddenDimRow).toHaveClass("border-violet/40");
    expect(hiddenDimRow).not.toHaveClass("border-amber/55", "bg-amber/[0.055]");
    expect(within(hiddenDimRow).queryByText("preset")).not.toBeInTheDocument();
    expect(within(dialog).queryByText(/\d+ preset/i)).not.toBeInTheDocument();
    expect(within(dialog).getAllByText("hidden dim")).toHaveLength(1);
  });

  it("highlights a section gradient when it has an override and a preset-owned field", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: schemaResponse.fields.map((field) =>
          field.key === "stack_activation"
            ? {
                ...field,
                locked: true,
                lockedValue: "GELU",
                lockedReason:
                  "Locked by the ACTIVATION preset because this preset fixes stack activation.",
              }
            : field,
        ),
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");

    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 1 override, 1 preset/i,
    });
    const layerSection = fullConfigSectionFor(layerAccordion);
    const layerNavRow = fullConfigSectionNavRowFor(
      sectionNav,
      /jump to layer stack options/i,
    );

    expect(layerAccordion).toHaveClass(
      "bg-[linear-gradient(90deg,rgba(255,209,102,0.12),rgba(167,139,250,0.13))]",
      "hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.16),rgba(167,139,250,0.17))]",
    );
    expect(within(layerAccordion).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerAccordion).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerSection).toHaveClass(
      "border-amber/35",
      "bg-[linear-gradient(135deg,rgba(255,209,102,0.075),rgba(167,139,250,0.105))]",
      "ring-violet/25",
    );
    expect(layerSection).not.toHaveClass("bg-amber/[0.045]", "bg-violet/[0.06]");
    expect(layerNavRow).toHaveClass(
      "border-amber/35",
      "bg-[linear-gradient(90deg,rgba(255,209,102,0.075),rgba(167,139,250,0.095))]",
      "hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.11),rgba(167,139,250,0.13))]",
      "ring-violet/20",
    );
    expect(within(layerNavRow).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerNavRow).getByText("1 preset")).toHaveClass("text-amber");
  });

  it("renders stack layer count as an editable numeric default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const stackLayersInput = within(dialog).getByLabelText(/stack num layers/i);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(stackLayersInput).toHaveAttribute("type", "number");
    expect(stackLayersInput).toHaveValue(5);

    await user.clear(stackLayersInput);
    await user.type(stackLayersInput, "7");

    expect(stackLayersInput).toHaveValue(7);
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
  });

  it("popup Update Preview posts selected model, preset, and overrides", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: { hidden_dim: "128" },
      });
    });
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
  });

  it("opens a training command popup without closing full config", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandDialog).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline",
    );
  });

  it("uses the current selected preset and omits --config when there are no overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await selectTargetOption(user, "preset", "recurrent-gating-halting");
    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset recurrent-gating-halting",
    );
    expect((commandField(commandDialog) as HTMLTextAreaElement).value).not.toContain("--config");
  });

  it("includes live overrides in display order before Update Preview", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.selectOptions(within(dialog).getByLabelText(/stack activation/i), "RELU");
    await user.click(within(dialog).getByRole("switch", { name: /gate flag/i }));

    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --hidden-dim 128 --stack-activation RELU --gate-flag true",
    );
  });

  it("updates the training command after resetting overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    let commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --hidden-dim 128",
    );

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));
    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));
    commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline",
    );
  });

  it("shell-quotes override values and serializes nullable empty overrides as None", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "dropout_schedule",
            configKey: "DROPOUT_SCHEDULE",
            flag: "--dropout-schedule",
            label: "dropout schedule",
            section: "Layer Stack Options",
            type: "enum",
            default: null,
            nullable: true,
            choices: ["cosine decay"],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const scheduleInput = within(dialog).getByLabelText(/dropout schedule/i);
    await user.selectOptions(scheduleInput, "cosine decay");
    let commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --dropout-schedule 'cosine decay'",
    );

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));
    await user.selectOptions(scheduleInput, "");
    commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --dropout-schedule None",
    );
  });

  it("copies the exact training command to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    installFetchMock();
    renderViewer();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const commandDialog = await openTrainingCommand(user, dialog);
    const expectedCommand = "source experiment.sh linear --preset baseline --config --hidden-dim 128";

    await user.click(within(commandDialog).getByRole("button", { name: /copy command/i }));

    expect(writeText).toHaveBeenCalledWith(expectedCommand);
    expect(within(commandDialog).getByRole("status")).toHaveTextContent("Command copied");
  });

  it("closing the training command popup leaves the full config dialog open", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));

    expect(screen.queryByRole("dialog", { name: /training command/i })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
  });

  it("popup Reset Overrides clears override state", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const hiddenInput = within(dialog).getByLabelText(/hidden dim/i);

    await user.clear(hiddenInput);
    await user.type(hiddenInput, "128");
    expect(hiddenInput).toHaveValue(128);

    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));

    expect(hiddenInput).toHaveValue(256);
    expect(within(dialog).getAllByLabelText("0 overrides").length).toBeGreaterThan(0);
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
  });

  it("close button removes the full config dialog", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });

    await user.click(within(dialog).getByRole("button", { name: /close full config/i }));

    expect(screen.queryByRole("dialog", { name: /full configuration/i })).not.toBeInTheDocument();
  });

});
