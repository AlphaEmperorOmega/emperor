import { useState } from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FullPageError } from "@/components/layout/page-error-status";
import { FullPageLoading, FullPageStatus } from "@/components/layout/page-status";
import { StatusCard } from "@/components/ui/status-card";
import { FeatureListDialog } from "@/features/workbench/components/feature-list-dialog";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { KeyValueRow } from "@/features/workbench/components/shared/key-value-row";
import { MetricCard } from "@/features/workbench/components/shared/metric-card";
import { SectionHeading } from "@/components/ui/section-heading";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { WorkbenchWorkspaceNav } from "@/features/workbench/components/workbench-workspace-nav";
import { IMPLEMENTED_FEATURES } from "@/lib/feature-catalog";

describe("FullPageStatus", () => {
  it("renders status content, detail text, icon, and action", () => {
    render(
      <FullPageStatus
        title="Backend offline"
        detail="Start the API server."
        icon={<span aria-hidden>!</span>}
        action={<button type="button">Retry</button>}
      />,
    );

    expect(screen.getByRole("heading", { name: "Backend offline" })).toBeInTheDocument();
    expect(screen.getByText("Start the API server.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders the loading title without optional detail text", () => {
    render(<FullPageLoading />);

    expect(screen.getByRole("heading", { name: "Loading Workbench…" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");
    expect(screen.queryByText(/Start the API/i)).not.toBeInTheDocument();
  });
});

describe("FullPageError", () => {
  it("uses the fallback error copy and calls retry", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<FullPageError onRetry={onRetry} />);

    expect(screen.getByText(/unexpected error/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try Again" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("renders a provided error message", () => {
    render(<FullPageError message="Exploded during render" onRetry={() => {}} />);

    expect(screen.getByText("Exploded during render")).toBeInTheDocument();
  });
});

describe("StatusCard", () => {
  it.each(["page", "overlay", "chart"] as const)(
    "renders the %s layout on the shared surface",
    (layout) => {
      const { container } = render(
        <StatusCard
          layout={layout}
          title="Loading scalars"
          detail="Waiting for TensorBoard data."
          busy
          actions={<button type="button">Retry</button>}
        />,
      );

      const action = screen.getByRole("button", { name: "Retry" });
      const surface = action.closest("div");
      expect(surface).toHaveClass(
        "rounded-panel",
        "border",
        "p-region",
        "shadow-panel",
      );
      expect(surface).not.toHaveClass("edge", "rounded-card");
      expect(screen.getByText("Loading scalars")).toBeInTheDocument();
      expect(screen.getByText("Waiting for TensorBoard data.")).toBeInTheDocument();
      expect(container.querySelector(".animate-spin")).toBeInTheDocument();
    },
  );

  it("renders default and danger inline layouts with roles and actions", () => {
    const { container, rerender } = render(
      <StatusCard
        title="Inline loading"
        detail="Still working."
        busy
        actions={<button type="button">Cancel</button>}
      />,
    );

    const inlineSurface = screen.getByRole("button", { name: "Cancel" }).closest("div");
    expect(inlineSurface).toHaveClass(
      "rounded-panel",
      "border-line",
      "bg-panel-2/70",
      "p-panel",
    );
    expect(inlineSurface).not.toHaveClass("edge", "rounded-card");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(container.querySelector(".animate-spin")).toBeInTheDocument();

    rerender(
      <StatusCard
        title="Inline error"
        detail="Request failed."
        tone="danger"
        actions={<button type="button">Retry</button>}
      />,
    );

    const alert = screen.getByRole("alert");
    expect(alert).toHaveClass(
      "rounded-panel",
      "border-danger-line",
      "bg-danger-soft",
      "text-danger-text",
    );
    expect(alert).not.toHaveClass("edge", "rounded-card");
    expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});

describe("WorkbenchWorkspaceNav", () => {
  it("marks the active workspace and emits workspace changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<WorkbenchWorkspaceNav activeWorkspace="model" onChange={onChange} />);

    const modelButton = screen.getByRole("button", { name: "Model" });
    const trainingButton = screen.getByRole("button", { name: "Training" });
    const logsButton = screen.getByRole("button", { name: "Logs" });
    const workspaceNav = screen.getByRole("navigation", { name: "Workspace" });
    const workspaceButtons = within(workspaceNav).getAllByRole("button");

    expect(workspaceButtons.map((button) => button.textContent?.trim())).toEqual([
      "Model",
      "Training",
      "Logs",
    ]);
    expect(within(workspaceNav).getByRole("list")).toBeInTheDocument();
    expect(within(workspaceNav).getAllByRole("listitem")).toHaveLength(3);

    expect(modelButton).toHaveAttribute("aria-current", "page");
    expect(modelButton.className).not.toContain("after:");
    expect(trainingButton).not.toHaveAttribute("aria-current");
    expect(logsButton).not.toHaveAttribute("aria-current");
    expect(modelButton).not.toHaveAttribute("aria-pressed");
    expect(trainingButton).not.toHaveAttribute("aria-pressed");
    expect(logsButton).not.toHaveAttribute("aria-pressed");
    for (const button of workspaceButtons) {
      const icon = button.querySelector("svg");
      expect(icon).not.toBeNull();
      expect(icon?.getAttribute("aria-hidden")).toBe("true");
    }

    await user.click(trainingButton);
    await user.click(logsButton);
    await user.click(modelButton);

    expect(onChange).toHaveBeenNthCalledWith(1, "training");
    expect(onChange).toHaveBeenNthCalledWith(2, "logs");
    expect(onChange).toHaveBeenNthCalledWith(3, "model");
  });

  it.each([
    ["running", "warn"],
    ["completed", "good"],
    ["failed", "danger"],
    ["cancelled", "danger"],
  ] as const)("renders the Training nav badge for %s jobs", (label, tone) => {
    render(
      <WorkbenchWorkspaceNav
        activeWorkspace="model"
        onChange={vi.fn()}
        trainingStatus={{ label, tone }}
      />,
    );

    const trainingButton = screen.getByRole("button", {
      name: new RegExp(`Training\\s+${label}`, "i"),
    });
    expect(trainingButton).toBeInTheDocument();
    expect(within(trainingButton).getByText(label)).toBeInTheDocument();
  });
});

describe("FeatureListDialog", () => {
  it("renders the feature catalog grouped in a modal dialog", () => {
    render(<FeatureListDialog onClose={() => {}} />);

    const dialog = screen.getByRole("dialog", { name: "Implemented Features" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(within(dialog).getByText(`${IMPLEMENTED_FEATURES.length} features`))
      .toBeInTheDocument();
    expect(within(dialog).getByText(IMPLEMENTED_FEATURES[0].title)).toBeInTheDocument();
  });

  it("calls onClose from the close button", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<FeatureListDialog onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: "Close implemented features" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("DialogShell", () => {
  it("moves focus into the dialog and traps tab navigation", async () => {
    const user = userEvent.setup();
    render(
      <>
        <button type="button">Before dialog</button>
        <DialogShell
          titleId="test-dialog-title"
          onClose={() => {}}
          header={<h2 id="test-dialog-title">Test Dialog</h2>}
        >
          <button type="button">First action</button>
          <button type="button">Second action</button>
        </DialogShell>
        <button type="button">After dialog</button>
      </>,
    );

    const firstAction = screen.getByRole("button", { name: "First action" });
    const secondAction = screen.getByRole("button", { name: "Second action" });

    await waitFor(() => expect(firstAction).toHaveFocus());
    await user.tab();
    expect(secondAction).toHaveFocus();
    await user.tab();
    expect(firstAction).toHaveFocus();
    await user.tab({ shift: true });
    expect(secondAction).toHaveFocus();
  });

  it("closes on Escape when allowed", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <DialogShell
        titleId="esc-dialog-title"
        onClose={onClose}
        header={<h2 id="esc-dialog-title">Esc Dialog</h2>}
      >
        <button type="button">Inside</button>
      </DialogShell>,
    );

    await waitFor(() => expect(screen.getByRole("button", { name: "Inside" })).toHaveFocus());
    await user.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("ignores Escape when close on Escape is disabled", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <DialogShell
        titleId="busy-dialog-title"
        onClose={onClose}
        closeOnEscape={false}
        header={<h2 id="busy-dialog-title">Busy Dialog</h2>}
      >
        <button type="button">Inside</button>
      </DialogShell>,
    );

    await user.keyboard("{Escape}");

    expect(onClose).not.toHaveBeenCalled();
  });

  it("restores focus to the opener after close", async () => {
    const user = userEvent.setup();

    function Harness() {
      const [open, setOpen] = useState(false);

      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open dialog
          </button>
          {open && (
            <DialogShell
              titleId="restore-dialog-title"
              onClose={() => setOpen(false)}
              header={<h2 id="restore-dialog-title">Restore Dialog</h2>}
            >
              <button type="button" onClick={() => setOpen(false)}>
                Close inside
              </button>
            </DialogShell>
          )}
        </>
      );
    }

    render(<Harness />);

    const opener = screen.getByRole("button", { name: "Open dialog" });
    await user.click(opener);
    await user.click(await screen.findByRole("button", { name: "Close inside" }));

    await waitFor(() => expect(opener).toHaveFocus());
  });

  it("lets only the topmost nested dialog handle Escape", async () => {
    const user = userEvent.setup();
    const onParentClose = vi.fn();
    const onChildClose = vi.fn();

    render(
      <DialogShell
        titleId="parent-dialog-title"
        onClose={onParentClose}
        header={<h2 id="parent-dialog-title">Parent Dialog</h2>}
        overlayChildren={
          <DialogShell
            titleId="child-dialog-title"
            onClose={onChildClose}
            header={<h2 id="child-dialog-title">Child Dialog</h2>}
            className="z-[60]"
          >
            <button type="button">Child action</button>
          </DialogShell>
        }
      >
        <button type="button">Parent action</button>
      </DialogShell>,
    );

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Child action" })).toHaveFocus(),
    );
    await user.keyboard("{Escape}");

    expect(onChildClose).toHaveBeenCalledTimes(1);
    expect(onParentClose).not.toHaveBeenCalled();
  });

  it("can render the subtle surface panel variant", () => {
    render(
      <DialogShell
        titleId="surface-dialog-title"
        panelVariant="surface"
        header={<h2 id="surface-dialog-title">Surface Dialog</h2>}
      >
        <button type="button">Inside surface</button>
      </DialogShell>,
    );

    const dialog = screen.getByRole("dialog", { name: "Surface Dialog" });
    expect(dialog.parentElement).toHaveClass(
      "safe-dialog-inset",
      "overscroll-contain",
    );
    expect(dialog).toHaveClass(
      "dialog-shell-panel",
      "min-w-0",
      "max-w-full",
      "rounded-dialog",
      "border",
      "border-line-hover",
      "bg-panel",
    );
    expect(dialog).not.toHaveClass("edge", "rounded-card", "bg-white/[0.018]");
  });
});

describe("InlineStatus", () => {
  it("renders children with default spacing and tone", () => {
    render(<InlineStatus>Waiting for config</InlineStatus>);

    const status = screen.getByText("Waiting for config");
    expect(status).toHaveClass(
      "rounded-panel",
      "border",
      "p-region",
      "border-line",
      "bg-panel-2/70",
      "text-ink-faint",
    );
  });

  it("applies compact spacing", () => {
    render(<InlineStatus compact>Compact status</InlineStatus>);

    const status = screen.getByText("Compact status");
    expect(status).toHaveClass("p-panel");
    expect(status).not.toHaveClass("p-region");
  });

  it("applies tone classes", () => {
    const { rerender } = render(<InlineStatus tone="danger">Danger</InlineStatus>);
    expect(screen.getByText("Danger")).toHaveClass(
      "border-danger-line",
      "bg-danger-soft",
      "text-danger-text",
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Danger");

    rerender(<InlineStatus tone="warning">Warning</InlineStatus>);
    expect(screen.getByText("Warning")).toHaveClass(
      "border-amber/40",
      "bg-amber/[0.12]",
      "text-amber",
    );

    rerender(<InlineStatus tone="success">Success</InlineStatus>);
    expect(screen.getByText("Success")).toHaveClass(
      "border-ok/30",
      "bg-ok/10",
      "text-ok",
    );
  });

  it("preserves explicit roles", () => {
    render(<InlineStatus role="alert">Failed to load</InlineStatus>);

    expect(screen.getByRole("alert")).toHaveTextContent("Failed to load");
  });

  it("renders a busy spinner", () => {
    render(<InlineStatus busy>Loading status</InlineStatus>);

    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveTextContent("Loading status");
    const spinner = status.querySelector("svg");
    expect(spinner).toHaveClass("h-4", "w-4", "animate-spin", "text-violet");
    expect(spinner).toHaveAttribute("aria-hidden", "true");
  });

  it("applies caller classes after defaults", () => {
    render(<InlineStatus className="p-2 text-xs custom-status">Custom status</InlineStatus>);

    const status = screen.getByText("Custom status");
    expect(status).toHaveClass("p-2", "text-xs", "custom-status");
    expect(status).not.toHaveClass("p-4", "text-sm");
  });
});

describe("SectionHeading", () => {
  it("renders an h2 title with icon and caller classes", () => {
    render(
      <SectionHeading
        as="h2"
        className="custom-heading text-ink"
        icon={<span data-testid="heading-icon" aria-hidden />}
        title="Metrics"
      />,
    );

    const heading = screen.getByRole("heading", { name: "Metrics" });
    expect(heading).toHaveClass(
      "flex",
      "min-w-0",
      "items-center",
      "gap-2",
      "type-label",
      "font-bold",
      "uppercase",
      "text-ink",
      "custom-heading",
    );
    expect(screen.getByTestId("heading-icon")).toBeInTheDocument();
  });

  it("renders count and actions", () => {
    render(
      <SectionHeading
        title="Logs"
        count={<span>12</span>}
        actions={<button type="button">Refresh</button>}
      />,
    );

    expect(screen.getByText("Logs")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
  });
});

describe("StatChip", () => {
  it("renders the default chip classes", () => {
    render(<StatChip>8 / 10</StatChip>);

    expect(screen.getByText("8 / 10")).toHaveClass(
      "rounded-chip",
      "border",
      "border-line",
      "bg-control",
      "px-2",
      "py-1",
      "font-mono",
      "tabular-nums",
      "text-xs",
      "text-ink-dim",
    );
  });

  it("applies soft tone, other tone classes, size classes, and caller classes", () => {
    const { rerender } = render(
      <StatChip tone="soft" className="custom-chip">
        Soft
      </StatChip>,
    );

    expect(screen.getByText("Soft")).toHaveClass(
      "border-line-soft",
      "bg-control-field",
      "type-meta",
      "font-semibold",
      "leading-none",
      "custom-chip",
    );

    rerender(
      <StatChip tone="violet" size="xs">
        Violet
      </StatChip>,
    );
    expect(screen.getByText("Violet")).toHaveClass(
      "border-violet/30",
      "bg-violet/10",
      "text-violet",
      "px-1.5",
      "py-0.5",
      "type-meta",
    );

    rerender(<StatChip tone="success">Success</StatChip>);
    expect(screen.getByText("Success")).toHaveClass("border-ok/30", "bg-ok/10", "text-ok");

    rerender(<StatChip tone="warning">Warning</StatChip>);
    expect(screen.getByText("Warning")).toHaveClass(
      "border-amber/40",
      "bg-amber/[0.12]",
      "text-amber",
    );

    rerender(<StatChip tone="danger">Danger</StatChip>);
    expect(screen.getByText("Danger")).toHaveClass(
      "border-danger-line",
      "bg-danger-soft",
      "text-danger-text",
    );
  });
});

describe("SurfacePanel", () => {
  it("renders the shared surface base, header slots, and merged classes", () => {
    render(
      <SurfacePanel
        icon={<span data-testid="surface-icon" aria-hidden />}
        title="Run Plan"
        detail={<span>3 runs</span>}
        actions={<button type="button">Refresh</button>}
        className="custom-surface py-3"
        headerClassName="custom-header min-h-0"
      >
        <span>Surface body</span>
      </SurfacePanel>,
    );

    const panel = screen.getByText("Surface body").closest(".custom-surface");
    expect(panel).toHaveClass(
      "grid",
      "min-w-0",
      "content-start",
      "gap-2",
      "rounded-panel",
      "border",
      "border-line",
      "bg-panel-2/70",
      "px-panel",
      "py-3",
    );
    expect(panel).not.toHaveClass("py-2");
    expect(screen.getByTestId("surface-icon")).toBeInTheDocument();
    expect(screen.getByText("Run Plan")).toBeInTheDocument();
    expect(screen.getByText("3 runs")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(screen.getByText("Run Plan").closest(".custom-header")).toHaveClass(
      "min-h-0",
    );
  });

  it("renders semantic sections with labelled region attributes", () => {
    render(
      <SurfacePanel as="section" aria-labelledby="surface-title">
        <h2 id="surface-title">Surface Region</h2>
        <span>Surface region body</span>
      </SurfacePanel>,
    );

    const region = screen.getByRole("region", { name: "Surface Region" });
    expect(region.tagName).toBe("SECTION");
    expect(region).toHaveAttribute("aria-labelledby", "surface-title");
    expect(region).toHaveClass(
      "rounded-panel",
      "border",
      "border-line",
      "bg-panel-2/70",
    );
  });

  it.each([
    ["compact", "gap-2", "px-panel", "py-2"],
    ["roomy", "gap-panel", "p-panel"],
    ["spacious", "gap-region", "p-region"],
    ["none", "gap-0", "p-0"],
  ] as const)("applies %s padding classes", (padding, ...classes) => {
    render(
      <SurfacePanel padding={padding} className={`surface-${padding}`}>
        {padding}
      </SurfacePanel>,
    );

    const panel = screen.getByText(padding);
    expect(panel).toHaveClass(...classes);
  });
});

describe("MetricCard", () => {
  it("renders label, value, detail, and caller classes", () => {
    render(
      <MetricCard
        icon={<span data-testid="metric-icon" aria-hidden />}
        label="Runs"
        value="42"
        valueTitle="42 runs"
        detail="complete"
        className="custom-card py-2.5"
        valueClassName="text-sm text-violet"
        detailClassName="text-ok"
      />,
    );

    const card = screen.getByText("Runs").closest(".custom-card");
    expect(screen.getByTestId("metric-icon")).toBeInTheDocument();
    expect(card).toHaveClass(
      "rounded-panel",
      "border",
      "border-line",
      "bg-panel-2/70",
      "px-panel",
      "py-2.5",
      "custom-card",
    );
    expect(card).not.toHaveClass("edge", "rounded-[12px]", "py-2");
    expect(screen.getByText("Runs")).toHaveClass(
      "flex",
      "min-w-0",
      "items-center",
      "gap-2",
      "type-label",
      "font-bold",
      "uppercase",
      "text-ink-dim",
    );
    expect(screen.getByText("42")).toHaveClass(
      "font-mono",
      "font-bold",
      "text-sm",
      "text-violet",
    );
    expect(screen.getByTitle("42 runs")).toHaveTextContent("42");
    expect(screen.getByText("complete")).toHaveClass("text-ok");
  });
});

describe("KeyValueRow", () => {
  it("renders the line variant with default value classes and caller classes", () => {
    render(
      <KeyValueRow
        label="Params"
        value="1,024"
        className="custom-row last:border-b-0"
        valueClassName="text-violet-text"
      />,
    );

    const row = screen.getByText("Params").parentElement;
    expect(row).toHaveClass(
      "grid",
      "grid-cols-[104px_minmax(0,1fr)]",
      "gap-panel",
      "border-b",
      "border-line-soft",
      "py-panel",
      "text-xs",
      "custom-row",
      "last:border-b-0",
    );
    expect(screen.getByText("1,024")).toHaveClass(
      "break-words",
      "text-right",
      "font-mono",
      "font-semibold",
      "tabular-nums",
      "text-violet-text",
    );
  });

  it("renders the card variant", () => {
    render(<KeyValueRow variant="card" label="result.json" value="present" />);

    const row = screen.getByText("result.json").parentElement;
    expect(row).toHaveClass(
      "grid-cols-[minmax(0,1fr)_auto]",
      "items-center",
      "rounded-control-md",
      "border-line-soft",
      "bg-control-field",
      "px-panel",
      "py-2",
    );
    expect(screen.getByText("present")).toHaveClass(
      "font-mono",
      "font-semibold",
      "text-ink",
    );
  });
});

describe("createWorkbenchContext", () => {
  it("returns values from its generated provider", () => {
    const [Provider, useValue] = createWorkbenchContext<{ label: string }>("TestContext");

    function Consumer() {
      return <span>{useValue().label}</span>;
    }

    render(
      <Provider value={{ label: "provided" }}>
        <Consumer />
      </Provider>,
    );

    expect(screen.getByText("provided")).toBeInTheDocument();
  });

  it("throws a clear error when rendered outside its provider", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const [, useValue] = createWorkbenchContext<{ label: string }>("MissingContext");

    function Consumer() {
      useValue();
      return null;
    }

    expect(() => render(<Consumer />)).toThrow(
      "MissingContext value is missing; render within its provider.",
    );
    consoleError.mockRestore();
  });
});
