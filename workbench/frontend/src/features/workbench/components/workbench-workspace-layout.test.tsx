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
  it("owns narrow stacking and wide three-column region occupancy", () => {
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
      "grid-cols-1",
      "overflow-auto",
      "lg:grid-cols-[344px_minmax(0,1fr)_332px]",
      "lg:overflow-hidden",
    );
    expect(sidebar.tagName).toBe("ASIDE");
    expect(sidebar).toHaveClass("border-b", "lg:border-b-0", "lg:border-r");
    expect(primary).toHaveClass("min-h-[560px]", "lg:min-h-0");
    expect(details.tagName).toBe("ASIDE");
    expect(details).toHaveClass("border-t", "lg:border-l", "lg:border-t-0");
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
      "min-h-[560px]",
      "overflow-hidden",
      "lg:col-span-3",
      "lg:min-h-0",
    );
    expect(wide).toContainElement(status);
    expect(document.querySelectorAll("[data-workbench-region]")).toHaveLength(1);
  });

  it("owns the horizontally scrollable wide three-region protocol", () => {
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

    expect(layout).toHaveClass("overflow-x-auto", "overflow-y-hidden");
    expect(columns).toHaveClass(
      "min-w-[920px]",
      "grid-cols-[minmax(300px,340px)_minmax(22rem,1fr)_minmax(280px,360px)]",
    );
    expect(leading.tagName).toBe("ASIDE");
    expect(leading).toHaveAttribute("aria-label", "Setup region");
    expect(primary.tagName).toBe("MAIN");
    expect(primary).toHaveAttribute("aria-label", "Runs region");
    expect(trailing.tagName).toBe("ASIDE");
    expect(trailing).toHaveAttribute("aria-label", "Status region");
    expect(trailing).toHaveAttribute("aria-live", "polite");
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
