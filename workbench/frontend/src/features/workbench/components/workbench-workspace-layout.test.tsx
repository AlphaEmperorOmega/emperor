import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  WorkbenchThreeRegionLayout,
  WorkbenchWideWorkspaceRegion,
  WorkbenchWorkspaceFrame,
  WorkbenchWorkspaceLoadingStatus,
} from "@/features/workbench/components/workbench-workspace-layout";
import { WorkbenchWideThreeRegionLayout } from "@/features/workbench/components/_workbench-wide-three-region-layout";

function region(name: string) {
  const element = document.querySelector(
    `[data-workbench-region="${name}"]`,
  );
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Expected ${name} Workbench region`);
  }
  return element;
}

describe("Workbench workspace layout", () => {
  it("stacks through tablet widths and owns desktop three-column occupancy", () => {
    render(
      <WorkbenchWorkspaceFrame>
        <WorkbenchThreeRegionLayout
          sidebar={<span>Sidebar content</span>}
          primary={<span>Primary content</span>}
          details={<span>Details content</span>}
        />
      </WorkbenchWorkspaceFrame>,
    );

    const frame = document.getElementById("workbench-workspace-content");
    const sidebar = region("sidebar");
    const primary = region("primary");
    const details = region("details");

    expect(frame).toHaveAttribute("data-workbench-layout", "workspace-frame");
    expect(frame).toHaveClass(
      "block",
      "overflow-x-hidden",
      "overflow-y-auto",
      "xl:grid",
      "xl:grid-cols-[320px_minmax(0,1fr)_320px]",
      "xl:overflow-hidden",
      "2xl:grid-cols-[344px_minmax(0,1fr)_332px]",
    );
    expect(sidebar.tagName).toBe("ASIDE");
    expect(sidebar).toHaveClass("border-b", "xl:border-b-0", "xl:border-r");
    expect(primary).toHaveClass(
      "min-h-[520px]",
      "sm:min-h-[640px]",
      "xl:min-h-0",
    );
    expect(details.tagName).toBe("ASIDE");
    expect(details).toHaveClass("border-t", "xl:border-l", "xl:border-t-0");
    expect(
      sidebar.compareDocumentPosition(primary) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      primary.compareDocumentPosition(details) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("gives wide workspaces and loading fallbacks the complete grid span", () => {
    render(
      <WorkbenchWorkspaceFrame>
        <WorkbenchWideWorkspaceRegion>
          <WorkbenchWorkspaceLoadingStatus label="Loading workspace" />
        </WorkbenchWideWorkspaceRegion>
      </WorkbenchWorkspaceFrame>,
    );

    const wide = region("wide");
    const status = screen.getByRole("status", { name: "Loading workspace" });

    expect(wide).toHaveClass(
      "min-h-[520px]",
      "sm:min-h-[640px]",
      "overflow-hidden",
      "xl:col-span-3",
      "xl:min-h-0",
    );
    expect(wide).toContainElement(status);
    expect(status).toHaveAttribute("aria-busy", "true");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(within(status).getByText("Preparing this workspace…")).toBeInTheDocument();
    expect(status.querySelector(".animate-spin")).toBeInTheDocument();
    expect(document.querySelectorAll("[data-workbench-region]")).toHaveLength(1);
  });

  it("stacks wide regions without workspace-wide horizontal scrolling", () => {
    render(
      <WorkbenchWideThreeRegionLayout
        notices={<span>Notice</span>}
        leading={<span>Setup</span>}
        primary={<span>Runs</span>}
        trailing={<span>Status</span>}
        leadingLabel="Setup region"
        primaryLabel="Runs region"
        trailingLabel="Status region"
      />,
    );

    const layout = document.querySelector(
      '[data-workbench-layout="wide-three-region"]',
    );
    const leading = region("wide-leading");
    const primary = region("wide-primary");
    const trailing = region("wide-trailing");
    const columns = leading.parentElement;

    expect(layout).toHaveClass(
      "block",
      "overflow-x-hidden",
      "overflow-y-auto",
      "xl:grid",
      "xl:overflow-y-hidden",
    );
    expect(columns).toHaveClass(
      "block",
      "min-w-0",
      "space-y-panel",
      "xl:grid",
      "xl:space-y-0",
      "xl:grid-cols-[minmax(280px,320px)_minmax(0,1fr)_minmax(280px,340px)]",
    );
    expect(leading.tagName).toBe("ASIDE");
    expect(leading).toHaveAttribute("aria-label", "Setup region");
    expect(primary.tagName).toBe("SECTION");
    expect(primary).toHaveAttribute("aria-label", "Runs region");
    expect(primary).toHaveClass("min-h-[600px]", "xl:min-h-0");
    expect(trailing.tagName).toBe("ASIDE");
    expect(trailing).toHaveAttribute("aria-label", "Status region");
    expect(trailing).toHaveAttribute("aria-live", "polite");
    expect(leading).toHaveClass("overflow-visible", "xl:overflow-y-auto");
    expect(trailing).toHaveClass("overflow-visible", "xl:overflow-y-auto");
    expect(within(leading).getByText("Setup")).toBeInTheDocument();
    expect(within(primary).getByText("Runs")).toBeInTheDocument();
    expect(within(trailing).getByText("Status")).toBeInTheDocument();
    expect(
      leading.compareDocumentPosition(primary) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      primary.compareDocumentPosition(trailing) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
