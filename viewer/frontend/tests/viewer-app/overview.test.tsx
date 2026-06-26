import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  IMPLEMENTED_FEATURES,
  capabilitiesResponse,
  commandField,
  expandedTrainingDetails,
  installFetchMock,
  inspectResponse,
  jsonResponse,
  logRunsResponse,
  openFullConfig,
  openTrainingMultiSelect,
  renderViewer,
  resetViewerAppTestState,
  schemaResponse,
  selectTargetOption,
  setTrainingMultiSelectOption,
  typeConfigFieldValue,
  waitForTargetValue,
} from "./support";
import {
  readPersistedTargetSelection,
} from "@/features/viewer/state/target/target-selection-storage";
import {
  setViewerApiBaseUrl,
  VIEWER_API_BASE_URL_STORAGE_KEY,
} from "@/lib/api";

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

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
    expect(
      screen.queryByRole("button", { name: /update preview/i }),
    ).not.toBeInTheDocument();
    expect(model).toHaveAttribute("aria-expanded", "false");
    expect(preset).toHaveAttribute("aria-expanded", "false");

    await user.click(model);

    const modelOptions = await screen.findByRole("listbox", {
      name: /model options/i,
    });
    expect(model).toHaveAttribute("aria-expanded", "true");
	    expect(within(modelOptions).getByRole("option", { name: "linear" }))
	      .toBeInTheDocument();
	    expect(within(modelOptions).queryByRole("option", { name: "bert_linear" }))
	      .not.toBeInTheDocument();

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

  it("renders workspace navigation in the header without duplicating it in the sidebar", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("model", "linear");
    await waitForTargetValue("preset", "baseline");

    const header = document.querySelector("header");
    if (!header) {
      throw new Error("Expected app header to render");
    }
    const workspaceContent = document.getElementById("viewer-workspace-content");
    const sidebar = workspaceContent?.firstElementChild;
    if (!(sidebar instanceof HTMLElement)) {
      throw new Error("Expected workspace sidebar to render");
    }

    expect(within(header).queryByText("Emperor Model Viewer")).not.toBeInTheDocument();
    expect(within(header).queryByText(/linear\s*\/\s*baseline/i))
      .not.toBeInTheDocument();

    const headerNav = within(header).getByRole("navigation", { name: "Workspace" });
    expect(
      within(headerNav).getAllByRole("button").map((button) => button.textContent?.trim()),
    ).toEqual(["Model", "Training", "Logs", "Compare"]);
    expect(within(sidebar).queryByRole("navigation", { name: "Workspace" }))
      .not.toBeInTheDocument();

    await user.click(within(headerNav).getByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();
    expect(within(headerNav).getByRole("button", { name: /^logs$/i }))
      .toHaveAttribute("aria-current", "page");
    expect(within(sidebar).queryByRole("navigation", { name: "Workspace" }))
      .not.toBeInTheDocument();
  });

  it("opens API connection settings with frontend and backend environment guidance", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    installFetchMock();
    renderViewer();

    const apiButton = screen.getByRole("button", {
      name: /api connection settings/i,
    });
    expect(apiButton).toBeInTheDocument();

    await user.click(apiButton);

    const dialog = await screen.findByRole("dialog", {
      name: /api connection/i,
    });
    const frontendOrigin = window.location.origin;
    const apiBaseUrl = "http://127.0.0.1:9999";
    const corsEnvValue = `VIEWER_API_CORS_ORIGINS='${JSON.stringify([
      frontendOrigin,
    ])}'`;
    const frontendEnvValue = `NEXT_PUBLIC_VIEWER_API_URL=${apiBaseUrl}`;

    expect(within(dialog).getByText(frontendOrigin)).toBeInTheDocument();
    expect(within(dialog).getByText(apiBaseUrl)).toBeInTheDocument();
    expect(
      within(dialog).getByLabelText("Backend CORS environment variable"),
    ).toHaveValue(corsEnvValue);
    expect(
      within(dialog).getByLabelText("Frontend API URL environment variable"),
    ).toHaveValue(frontendEnvValue);
    expect(within(dialog).getByLabelText("API base URL")).toHaveValue(apiBaseUrl);
    expect(dialog).toHaveTextContent(/backend deployment/i);
    expect(dialog).toHaveTextContent(/browser frontend code cannot add/i);
    expect(within(dialog).getByText(/bearer auth/i)).toBeInTheDocument();
    expect(dialog).toHaveTextContent(/unsafe local mutation/i);
    expect(dialog).toHaveTextContent(/training, log deletion, and config snapshots/i);

    await user.click(
      within(dialog).getByRole("button", {
        name: /copy backend cors environment variable/i,
      }),
    );
    await user.click(
      within(dialog).getByRole("button", {
        name: /copy frontend api url environment variable/i,
      }),
    );

    expect(writeText).toHaveBeenNthCalledWith(1, corsEnvValue);
    expect(writeText).toHaveBeenNthCalledWith(2, frontendEnvValue);
  });

  it("renders flat header action buttons in order in the top nav", async () => {
    installFetchMock();
    renderViewer();

    const header = document.querySelector("header");
    if (!header) {
      throw new Error("Expected app header to render");
    }

    const actionButtons = [
      within(header).getByRole("button", {
        name: /api connection settings/i,
      }),
      within(header).getByRole("button", { name: /import logs/i }),
      within(header).getByRole("button", { name: /^features$/i }),
      within(header).getByRole("button", { name: /^reset overrides$/i }),
    ];
    const headerButtons = within(header).getAllByRole("button");

    expect(
      headerButtons.filter((button) => actionButtons.includes(button)),
    ).toEqual(actionButtons);
    for (const button of actionButtons) {
      expect(button).toHaveClass(
        "h-9",
        "gap-1.5",
        "rounded-control-sm",
        "border-0",
        "bg-transparent",
        "text-ink-dim",
        "hover:bg-control-hover",
        "hover:text-ink",
        "focus-visible:ring-focus",
      );
      for (const legacyClassName of [
        "rounded-control",
        "border-line",
        "bg-control",
        "hover:border-line-hover",
        "hover:bg-control-active",
      ]) {
        expect(button).not.toHaveClass(legacyClassName);
      }
    }
    expect(actionButtons[3]).toHaveClass(
      "disabled:text-ink-faint",
      "disabled:hover:bg-transparent",
      "disabled:hover:text-ink-faint",
    );
  });

  it("imports a selected zip through the configured API base URL", async () => {
    const user = userEvent.setup();
    setViewerApiBaseUrl("https://api.example.test/viewer");
    const { fetchMock, logImportRequests } = installFetchMock({
      capabilitiesResponse: {
        ...capabilitiesResponse,
        uploadsEnabled: true,
        maxUploadSize: null,
      },
      logImportResponse: {
        extractedFileCount: 2,
        skippedFileCount: 1,
        destinationRoot: "/workspace/logs",
      },
    });
    renderViewer();

    await user.click(screen.getByRole("button", { name: /import logs/i }));

    const dialog = await screen.findByRole("dialog", { name: /import logs/i });
    const fileInput = within(dialog).getByLabelText(/log archive zip file/i);
    await waitFor(() => expect(fileInput).toBeEnabled());

    const file = new File(["zip"], "logs.zip", { type: "application/zip" });
    await user.upload(fileInput, file);

    expect(within(dialog).getByText("logs.zip")).toBeInTheDocument();
    expect(within(dialog).getByText("3 B")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^import logs$/i }));

    await waitFor(() => expect(logImportRequests).toHaveLength(1));
    expect(
      fetchMock.mock.calls.some(
        ([url]) => String(url) === "https://api.example.test/viewer/logs/import",
      ),
    ).toBe(true);
    const request = logImportRequests[0];
    expect(request.body).toBeInstanceOf(FormData);
    const headers = new Headers(request.headers);
    expect(headers.has("content-type")).toBe(false);
    expect(within(dialog).getByRole("status")).toHaveTextContent(
      /2 files extracted; 1 existing file skipped/i,
    );
    await waitFor(() => {
      expect((fileInput as HTMLInputElement).files).toHaveLength(0);
    });
    expect(within(dialog).getByText("No file selected")).toBeInTheDocument();
  });

  it("shows loading and success states while importing logs", async () => {
    const user = userEvent.setup();
    const importResponse = deferred<{
      extractedFileCount: number;
      skippedFileCount: number;
      destinationRoot: string;
    }>();
    const importResult = {
      extractedFileCount: 1,
      skippedFileCount: 0,
      destinationRoot: "/workspace/logs",
    };
    installFetchMock({
      capabilitiesResponse: {
        ...capabilitiesResponse,
        uploadsEnabled: true,
        maxUploadSize: null,
      },
      logImportResponseFactory: () => importResponse.promise,
    });
    renderViewer();

    await user.click(screen.getByRole("button", { name: /import logs/i }));
    const dialog = await screen.findByRole("dialog", { name: /import logs/i });
    const fileInput = within(dialog).getByLabelText(/log archive zip file/i);
    await waitFor(() => expect(fileInput).toBeEnabled());
    await user.upload(
      fileInput,
      new File(["zip"], "logs.zip", { type: "application/zip" }),
    );
    await user.click(within(dialog).getByRole("button", { name: /^import logs$/i }));

    expect(within(dialog).getByRole("button", { name: /importing/i }))
      .toBeDisabled();
    expect(fileInput).toBeDisabled();

    importResponse.resolve(importResult);

    expect(await within(dialog).findByRole("status")).toHaveTextContent(
      /1 file extracted; 0 existing files skipped/i,
    );
  });

  it("shows an import error from the backend", async () => {
    const user = userEvent.setup();
    installFetchMock({
      capabilitiesResponse: {
        ...capabilitiesResponse,
        uploadsEnabled: true,
        maxUploadSize: null,
      },
      logImportError: "Invalid zip archive.",
    });
    renderViewer();

    await user.click(screen.getByRole("button", { name: /import logs/i }));
    const dialog = await screen.findByRole("dialog", { name: /import logs/i });
    const fileInput = within(dialog).getByLabelText(/log archive zip file/i);
    await waitFor(() => expect(fileInput).toBeEnabled());
    await user.upload(
      fileInput,
      new File(["not a zip"], "logs.zip", { type: "application/zip" }),
    );
    await user.click(within(dialog).getByRole("button", { name: /^import logs$/i }));

    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      /invalid zip archive/i,
    );
  });

  it("shows disabled state when the backend does not expose upload capability", async () => {
    const user = userEvent.setup();
    installFetchMock({
      capabilitiesResponse: {
        ...capabilitiesResponse,
        uploadsEnabled: false,
        maxUploadSize: null,
      },
    });
    renderViewer();

    await user.click(screen.getByRole("button", { name: /import logs/i }));
    const dialog = await screen.findByRole("dialog", { name: /import logs/i });

    expect(dialog).toHaveTextContent(/log imports are disabled by this backend/i);
    expect(within(dialog).getByLabelText(/log archive zip file/i)).toBeDisabled();
    expect(within(dialog).getByRole("button", { name: /^import logs$/i }))
      .toBeDisabled();
  });

  it("uses and resets a runtime API URL from the existing connection dialog", async () => {
    const user = userEvent.setup();
    const { fetchMock } = installFetchMock();
    renderViewer();

    await user.click(screen.getByRole("button", { name: /api connection settings/i }));

    const dialog = await screen.findByRole("dialog", {
      name: /api connection/i,
    });
    const input = within(dialog).getByLabelText("API base URL");

    await user.clear(input);
    await user.type(input, " https://api.example.test/viewer/// ");
    await user.click(within(dialog).getByRole("button", { name: /^use$/i }));

    await waitFor(() => {
      expect(input).toHaveValue("https://api.example.test/viewer");
    });
    expect(window.localStorage.getItem(VIEWER_API_BASE_URL_STORAGE_KEY)).toBe(
      "https://api.example.test/viewer",
    );
    expect(within(dialog).getByText("https://api.example.test/viewer"))
      .toBeInTheDocument();
    expect(
      within(dialog).getByLabelText("Frontend API URL environment variable"),
    ).toHaveValue("NEXT_PUBLIC_VIEWER_API_URL=https://api.example.test/viewer");
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url) === "https://api.example.test/viewer/health",
        ),
      ).toBe(true);
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url) === "https://api.example.test/viewer/inspect",
        ),
      ).toBe(true);
    });

    const requestCountBeforeReset = fetchMock.mock.calls.length;
    await user.click(within(dialog).getByRole("button", { name: /^reset$/i }));

    await waitFor(() => {
      expect(input).toHaveValue("http://127.0.0.1:9999");
    });
    expect(window.localStorage.getItem(VIEWER_API_BASE_URL_STORAGE_KEY)).toBeNull();
    expect(
      within(dialog).getByLabelText("Frontend API URL environment variable"),
    ).toHaveValue("NEXT_PUBLIC_VIEWER_API_URL=http://127.0.0.1:9999");
    await waitFor(() => {
      expect(fetchMock.mock.calls.length).toBeGreaterThan(requestCountBeforeReset);
      expect(
        fetchMock.mock.calls
          .slice(requestCountBeforeReset)
          .some(([url]) => String(url) === "http://127.0.0.1:9999/health"),
      ).toBe(true);
      expect(
        fetchMock.mock.calls
          .slice(requestCountBeforeReset)
          .some(([url]) => String(url) === "http://127.0.0.1:9999/inspect"),
      ).toBe(true);
    });
  });

  it("shows an inline validation error for invalid runtime API URLs", async () => {
    const user = userEvent.setup();
    const { fetchMock } = installFetchMock();
    renderViewer();

    await user.click(screen.getByRole("button", { name: /api connection settings/i }));

    const dialog = await screen.findByRole("dialog", {
      name: /api connection/i,
    });
    const input = within(dialog).getByLabelText("API base URL");
    const requestCount = fetchMock.mock.calls.length;

    await user.clear(input);
    await user.type(input, "https://api.example.test?debug=true");
    await user.click(within(dialog).getByRole("button", { name: /^use$/i }));

    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      /without a query string or fragment/i,
    );
    expect(window.localStorage.getItem(VIEWER_API_BASE_URL_STORAGE_KEY)).toBeNull();
    expect(fetchMock.mock.calls).toHaveLength(requestCount);
  });

  it("refetches active API status queries after switching backends", async () => {
    const user = userEvent.setup();
    const { fetchMock } = installFetchMock();
    const defaultFetch = fetchMock.getMockImplementation();
    fetchMock.mockImplementation((input, init) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        if (url === "https://api-online.example.test/health") {
          return jsonResponse({ status: "ok" });
        }
        return jsonResponse({ detail: "offline" }, 503);
      }
      if (!defaultFetch) {
        throw new Error("Expected default fetch mock implementation");
      }
      return defaultFetch(input, init);
    });
    renderViewer();

    const header = document.querySelector("header");
    if (!header) {
      throw new Error("Expected app header to render");
    }
    expect(await within(header).findByText("offline")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /api connection settings/i }));
    const dialog = await screen.findByRole("dialog", {
      name: /api connection/i,
    });
    const input = within(dialog).getByLabelText("API base URL");
    await user.clear(input);
    await user.type(input, "https://api-online.example.test");
    await user.click(within(dialog).getByRole("button", { name: /^use$/i }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url) === "https://api-online.example.test/health",
        ),
      ).toBe(true);
    });
    expect(await within(header).findByText("online")).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        ([url]) => String(url) === "https://api-online.example.test/models",
      ),
    ).toBe(true);
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
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      dataset: "Mnist",
      overrides: {},
    });
  });

  it("defaults the target panel to Presets and renders all target mode tabs", async () => {
    installFetchMock();
    renderViewer();

    await waitForTargetValue("preset", "baseline");

    const presetsTab = screen.getByRole("radio", { name: "Presets" });
    const snapshotsTab = screen.getByRole("radio", { name: "Snapshots" });
    const experimentsTab = screen.getByRole("radio", { name: "Experiments" });

    expect(presetsTab).toHaveAttribute("aria-checked", "true");
    expect(snapshotsTab).toHaveAttribute("aria-checked", "false");
    expect(snapshotsTab).toBeDisabled();
    expect(experimentsTab).toHaveAttribute("aria-checked", "false");
    expect(presetsTab.querySelector("svg")).toBeNull();
    expect(snapshotsTab.querySelector("svg")).toBeNull();
    expect(experimentsTab.querySelector("svg")).toBeNull();
    expect(screen.getByRole("combobox", { name: /^preset$/i }))
      .toHaveTextContent("baseline");
    expect(screen.queryByRole("combobox", { name: /^snapshot$/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: /^experiment$/i }))
      .not.toBeInTheDocument();
  });

  it("keeps the Experiments tab reachable when lazy tag reads find no eligible layer-monitor runs", async () => {
    installFetchMock({
      logTagsByRun: {
        "log-mnist": ["train/loss"],
        "log-cifar": ["validation/accuracy"],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");

    const experimentsTab = screen.getByRole("radio", { name: "Experiments" });
    expect(experimentsTab).not.toBeDisabled();

    await user.click(experimentsTab);

    const experimentControl = await screen.findByRole("combobox", {
      name: /^experiment$/i,
    });
    await waitFor(() => {
      expect(experimentControl).toBeDisabled();
    });
    expect(screen.getByRole("combobox", { name: /^dataset$/i })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: /^preset$/i })).toBeDisabled();
  });

  it("shows only current-model snapshots in the target snapshot selector", async () => {
    installFetchMock({
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "linear-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { stack_hidden_dim: "128", num_epochs: "5" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
          {
            id: "bert-wide",
            modelType: "transformer_encoder",
            model: "bert_linear",
            preset: "bert-baseline",
            name: "Bert snapshot",
            overrides: { stack_hidden_dim: "64" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await user.click(screen.getByRole("radio", { name: "Snapshots" }));
    const snapshotControl = await screen.findByRole("combobox", {
      name: /^snapshot$/i,
    });
    expect(snapshotControl).toHaveTextContent("Wide snapshot");

    await user.click(snapshotControl);
    const snapshotOptions = await screen.findByRole("listbox", {
      name: /^snapshot options$/i,
    });
    expect(
      within(snapshotOptions).getByRole("option", {
        name: "Wide snapshot",
      }),
    ).toBeInTheDocument();
    expect(
      within(snapshotOptions).queryByRole("option", {
        name: "Bert snapshot",
      }),
    ).not.toBeInTheDocument();
  });

  it("selects a snapshot target with its preset and saved overrides", async () => {
    let inspectBodies: unknown[] = [];
    ({ inspectBodies } = installFetchMock({
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "linear-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { stack_hidden_dim: "128", num_epochs: "5" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
          {
            id: "linear-recurrent",
            modelType: "linears",
            model: "linear",
            preset: "recurrent-gating-halting",
            name: "Recurrent snapshot",
            overrides: { stack_hidden_dim: "64" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
      inspectResponseFactory: (requestIndex) => {
        const request = inspectBodies[requestIndex] as { preset?: string } | undefined;
        return { ...inspectResponse, preset: request?.preset ?? "baseline" };
      },
    }));
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await user.click(screen.getByRole("radio", { name: "Snapshots" }));
    const snapshotControl = await screen.findByRole("combobox", {
      name: /^snapshot$/i,
    });
    await user.click(snapshotControl);
    const snapshotRoot = snapshotControl.parentElement;
    if (!(snapshotRoot instanceof HTMLElement)) {
      throw new Error("Expected snapshot dropdown root");
    }
    await user.type(
      within(snapshotRoot).getByRole("searchbox", { name: /^search snapshot$/i }),
      "recurrent",
    );
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^snapshot options$/i }),
      ).getByRole("option", {
        name: "Recurrent snapshot",
      }),
    );

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "recurrent-gating-halting",
        dataset: "Mnist",
        overrides: { stack_hidden_dim: "64" },
      });
    });
    await waitFor(() => {
      expect(screen.getAllByText("linear / recurrent-gating-halting").length)
        .toBeGreaterThan(0);
    });
    expect(screen.getByRole("combobox", { name: /^snapshot$/i }))
      .toHaveTextContent("Recurrent snapshot");
  });

  it("restores the selected model and snapshot target after a refresh", async () => {
    const { inspectBodies } = installFetchMock({
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "linear-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { stack_hidden_dim: "128" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
          {
            id: "bert-wide",
            modelType: "transformer_encoder",
            model: "bert_linear",
            preset: "bert-baseline",
            name: "Bert snapshot",
            overrides: { stack_hidden_dim: "64" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    const firstRender = renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("model", "linear");
    await selectTargetOption(user, "model type", "Transformer encoder");
    await waitForTargetValue("model", "bert_linear");
    await waitForTargetValue("preset", "bert-baseline");
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: "Snapshots" })).toBeEnabled(),
    );
    await user.click(screen.getByRole("radio", { name: "Snapshots" }));
    expect(await screen.findByRole("combobox", { name: /^snapshot$/i }))
      .toHaveTextContent("Bert snapshot");
	    await waitFor(() => {
	      expect(readPersistedTargetSelection()).toMatchObject({
	        selectedModelType: "transformer_encoder",
	        selectedModel: "bert_linear",
        selectedPreset: "bert-baseline",
        selectedTargetMode: "snapshot",
        selectedSnapshotId: "bert-wide",
      });
    });

    firstRender.unmount();
    inspectBodies.length = 0;
    renderViewer();

    expect(await waitForTargetValue("model", "bert_linear"))
      .toHaveTextContent("bert_linear");
    expect(screen.getByRole("radio", { name: "Snapshots" }))
      .toHaveAttribute("aria-checked", "true");
    expect(await screen.findByRole("combobox", { name: /^snapshot$/i }))
      .toHaveTextContent("Bert snapshot");
    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "transformer_encoder",
        model: "bert_linear",
        preset: "bert-baseline",
        dataset: "ToyText",
        overrides: { stack_hidden_dim: "64" },
      });
    });
  });

  it("switches from a snapshot target back to Presets with empty overrides", async () => {
    const { inspectBodies } = installFetchMock({
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "linear-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { stack_hidden_dim: "128", num_epochs: "5" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await user.click(screen.getByRole("radio", { name: "Snapshots" }));
    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: { stack_hidden_dim: "128", num_epochs: "5" },
      });
    });

    await user.click(screen.getByRole("radio", { name: "Presets" }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      });
    });
    expect(screen.getByRole("combobox", { name: /^preset$/i }))
      .toHaveTextContent("baseline");
    expect(screen.queryByRole("combobox", { name: /^snapshot$/i }))
      .not.toBeInTheDocument();
  });

  it("refreshes snapshot options when the selected model changes", async () => {
    installFetchMock({
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "linear-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { stack_hidden_dim: "128" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
          {
            id: "bert-wide",
            modelType: "transformer_encoder",
            model: "bert_linear",
            preset: "bert-baseline",
            name: "Bert snapshot",
            overrides: { stack_hidden_dim: "64" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await selectTargetOption(user, "model type", "Transformer encoder");
    await waitForTargetValue("model", "bert_linear");
    await waitForTargetValue("preset", "bert-baseline");

    await waitFor(() =>
      expect(screen.getByRole("radio", { name: "Snapshots" })).toBeEnabled(),
    );
    await user.click(screen.getByRole("radio", { name: "Snapshots" }));
    const snapshotControl = await screen.findByRole("combobox", {
      name: /^snapshot$/i,
    });
    expect(snapshotControl).toHaveTextContent("Bert snapshot");

    await user.click(snapshotControl);
    const snapshotOptions = await screen.findByRole("listbox", {
      name: /^snapshot options$/i,
    });
    expect(
      within(snapshotOptions).getByRole("option", {
        name: "Bert snapshot",
      }),
    ).toBeInTheDocument();
    expect(
      within(snapshotOptions).queryByRole("option", {
        name: "Wide snapshot",
      }),
    ).not.toBeInTheDocument();
  });

  it("opens a target preset training command with a local all-monitors toggle", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    expect(
      screen.queryByRole("button", { name: /show preset description/i }),
    ).not.toBeInTheDocument();
    const trigger = screen.getByRole("button", {
      name: /training command for preset/i,
    });
    expect(trigger).toBeEnabled();

    await user.click(trigger);

    let dialog = await screen.findByRole("dialog", { name: /training command/i });
    expect(commandField(dialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist",
    );
    const overlayRoot = dialog.parentElement;
    const workspaceContent = document.getElementById("viewer-workspace-content");
    const sidebar = workspaceContent?.firstElementChild;
    if (!(overlayRoot instanceof HTMLElement) || !(sidebar instanceof HTMLElement)) {
      throw new Error("Expected training command overlay and sidebar to render");
    }
    expect(overlayRoot.parentElement).toBe(document.body);
    expect(overlayRoot).toHaveClass("fixed", "inset-0", "z-[60]");
    expect(sidebar).not.toContain(overlayRoot);
    expect((commandField(dialog) as HTMLTextAreaElement).value)
      .not.toContain("--monitors");

    const monitorToggle = within(dialog).getByRole("button", {
      name: /include all monitors/i,
    });
    const copyButton = within(dialog).getByRole("button", { name: /copy command/i });
    const footer = monitorToggle.closest("footer");
    if (!(footer instanceof HTMLElement)) {
      throw new Error("Expected monitor toggle to render in the dialog footer");
    }
    expect(footer).toContain(copyButton);
    expect(
      monitorToggle.compareDocumentPosition(copyButton) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(monitorToggle).toHaveAttribute("aria-pressed", "false");

    await user.click(monitorToggle);

    await waitFor(() => {
      expect(commandField(dialog)).toHaveValue(
        "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist --monitors linear sampler",
      );
    });
    expect(monitorToggle).toHaveAttribute("aria-pressed", "true");

    await user.click(
      within(dialog).getByRole("button", { name: /close training command/i }),
    );
    await user.click(trigger);

    dialog = await screen.findByRole("dialog", { name: /training command/i });
    expect(within(dialog).getByRole("button", { name: /include all monitors/i }))
      .toHaveAttribute("aria-pressed", "false");
    expect(commandField(dialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist",
    );
    await user.click(
      within(dialog).getByRole("button", { name: /close training command/i }),
    );

    const trainingDetails = await expandedTrainingDetails(user);
    const monitorsDropdown = await openTrainingMultiSelect(
      user,
      trainingDetails,
      "Training monitors",
    );
    expect(
      within(monitorsDropdown.listbox).getByRole("option", {
        name: /Linear layers/i,
      }),
    ).toHaveAttribute("aria-selected", "false");
    expect(
      within(monitorsDropdown.listbox).getByRole("option", {
        name: /Sampler usage/i,
      }),
    ).toHaveAttribute("aria-selected", "false");
  });

  it("opens a snapshot training command with the snapshot preset and overrides", async () => {
    installFetchMock({
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "linear-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { stack_hidden_dim: "128", stack_num_layers: "7" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await user.click(screen.getByRole("radio", { name: "Snapshots" }));
    expect(await screen.findByRole("combobox", { name: /^snapshot$/i }))
      .toHaveTextContent("Wide snapshot");

    const trigger = screen.getByRole("button", {
      name: /training command for snapshot/i,
    });
    expect(trigger).toBeEnabled();

    await user.click(trigger);

    const dialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(dialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist --config --stack-hidden-dim 128 --stack-num-layers 7",
    );
  });

  it("disables the target training command button without a selected dataset", async () => {
    installFetchMock({
      datasetsResponse: {
        modelType: "linears",
        model: "linear",
        datasets: [],
      },
    });
    renderViewer();

    await waitForTargetValue("model", "linear");
    expect(
      screen.getByRole("button", { name: /training command for preset/i }),
    ).toBeDisabled();
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

  it("keeps training dataset selection out of model preview requests", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const trainingDetails = await expandedTrainingDetails(user);
    await setTrainingMultiSelectOption(
      user,
      trainingDetails,
      "Training datasets",
      /Cifar 10/i,
    );
    await setTrainingMultiSelectOption(
      user,
      trainingDetails,
      "Training datasets",
      /Mnist/i,
      false,
    );
    await user.click(screen.getByRole("button", { name: /^model\b/i }));
    const dialog = await openFullConfig(user);
    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toMatchObject({ dataset: "Mnist" });
    });
  });

  it("keeps dataset selection out of the main sidebar", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /datasets\s+\d+\s*\/\s*\d+/i }),
    ).not.toBeInTheDocument();

    const trainingDetails = await expandedTrainingDetails(user);
    const datasetsDropdown = await openTrainingMultiSelect(
      user,
      trainingDetails,
      "Training datasets",
    );
    expect(
      within(datasetsDropdown.listbox).getByRole("option", { name: /Mnist/i }),
    ).toHaveAttribute("aria-selected", "true");
    expect(
      within(datasetsDropdown.listbox).getByRole("option", { name: /Cifar 10/i }),
    ).toHaveAttribute("aria-selected", "false");
  });

  it("renders experiment runs as a Target mode and keeps the active run selected", async () => {
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
            modelType: "transformer_encoder",
            model: "bert_linear",
            preset: "BERT_BASELINE",
            dataset: "ToyText",
            relativePath:
              "bert_experiment/bert_linear/BERT_BASELINE/ToyText/ccc_20260601_030405/version_0",
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const experimentsTab = await screen.findByRole("radio", { name: "Experiments" });
    await waitFor(() => expect(experimentsTab).not.toBeDisabled());
    await user.click(experimentsTab);
    expect(experimentsTab).toHaveAttribute("aria-checked", "true");

    const experimentControl = await screen.findByRole("combobox", {
      name: /^experiment$/i,
    });
    await waitFor(() => expect(experimentControl).not.toBeDisabled());
    expect(experimentControl).toHaveTextContent("Select experiment");

    await user.click(experimentControl);
    const experimentOptions = await screen.findByRole("listbox", {
      name: /^experiment options$/i,
    });
    expect(
      within(experimentOptions).getByRole("option", {
        name: "test_model",
      }),
    ).toBeInTheDocument();
    expect(
      within(experimentOptions).getByRole("option", {
        name: "test_model_2",
      }),
    ).toBeInTheDocument();
    expect(
      within(experimentOptions).queryByRole("option", {
        name: /bert_experiment/i,
      }),
    ).not.toBeInTheDocument();
    await user.keyboard("{Escape}");

    await selectTargetOption(user, "experiment", "test_model_2");
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: /^dataset$/i }))
        .not.toBeDisabled(),
    );
    await selectTargetOption(user, "dataset", "Cifar10");
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: /^preset$/i }))
        .not.toBeDisabled(),
    );
    const presetControl = await selectTargetOption(user, "preset", "BASELINE");

    expect(screen.getByRole("radio", { name: "Experiments" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "Cifar10",
        overrides: {},
        logRunId: "log-cifar",
      });
    });

    const requestCount = inspectBodies.length;
    await selectTargetOption(user, "preset", "BASELINE");
    expect(experimentControl).toHaveTextContent("test_model_2");
    expect(screen.getByRole("combobox", { name: /^dataset$/i }))
      .toHaveTextContent("Cifar10");
    expect(presetControl).toHaveTextContent("BASELINE");
    expect(inspectBodies).toHaveLength(requestCount);
  });

  it("refreshes experiment run options when the selected model changes", async () => {
    installFetchMock({
      logRunsResponse: {
        runs: [
          {
            ...logRunsResponse.runs[0],
            id: "linear-run",
            experiment: "exp_alpha",
            group: "exp_alpha",
            preset: "BASELINE",
            dataset: "Mnist",
            timestamp: "2026-06-01 01:00:00",
            runName: "linear_20260601_010000",
            relativePath:
              "exp_alpha/linear/BASELINE/Mnist/linear_20260601_010000/version_0",
          },
          {
            ...logRunsResponse.runs[1],
            id: "bert-run",
            experiment: "exp_bert",
            group: "exp_bert",
            modelType: "transformer_encoder",
            model: "bert_linear",
            preset: "BERT_BASELINE",
            dataset: "ToyText",
            timestamp: "2026-06-01 03:00:00",
            runName: "bert_20260601_030000",
            relativePath:
              "exp_bert/bert_linear/BERT_BASELINE/ToyText/bert_20260601_030000/version_0",
          },
        ],
      },
      logTagsByRun: {
        "linear-run": ["main_model.0.model/weights/mean"],
        "bert-run": ["main_model.0.model/weights/mean"],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const experimentsTab = await screen.findByRole("radio", { name: "Experiments" });
    await waitFor(() => expect(experimentsTab).not.toBeDisabled());
    await user.click(experimentsTab);
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: /^experiment$/i }))
        .not.toBeDisabled(),
    );
    await selectTargetOption(user, "experiment", "exp_alpha");
    await selectTargetOption(user, "dataset", "Mnist");
    await selectTargetOption(user, "preset", "BASELINE");

    await selectTargetOption(user, "model type", "Transformer encoder");
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: /^model$/i }))
        .toHaveTextContent("bert_linear"),
    );
    const refreshedExperimentsTab = screen.getByRole("radio", { name: "Experiments" });
    await waitFor(() => expect(refreshedExperimentsTab).not.toBeDisabled());
    await user.click(refreshedExperimentsTab);
    const refreshedExperimentControl = await screen.findByRole("combobox", {
      name: /^experiment$/i,
    });
    await waitFor(() => expect(refreshedExperimentControl).not.toBeDisabled());
    expect(refreshedExperimentControl).toHaveTextContent("Select experiment");

    await user.click(refreshedExperimentControl);
    const refreshedOptions = await screen.findByRole("listbox", {
      name: /^experiment options$/i,
    });
    expect(
      within(refreshedOptions).getByRole("option", {
        name: "exp_bert",
      }),
    ).toBeInTheDocument();
    expect(
      within(refreshedOptions).queryByRole("option", {
        name: /exp_alpha/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("selecting a historical run syncs preset and dataset, clears overrides, and refreshes preview", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    expect(screen.getByText(/1 overrides?/i)).toBeInTheDocument();

    const experimentsTab = await screen.findByRole("radio", { name: "Experiments" });
    await waitFor(() => expect(experimentsTab).not.toBeDisabled());
    await user.click(experimentsTab);
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: /^experiment$/i }))
        .not.toBeDisabled(),
    );
    const experimentControl = await selectTargetOption(
      user,
      "experiment",
      "test_model",
    );
    await selectTargetOption(user, "dataset", "Mnist");
    await selectTargetOption(user, "preset", "BASELINE");

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "Mnist",
        overrides: {},
        logRunId: "log-mnist",
      });
    });
    expect(experimentControl).toHaveTextContent("test_model");
    expect(screen.getByText("0 overrides")).toBeInTheDocument();
  });

});
