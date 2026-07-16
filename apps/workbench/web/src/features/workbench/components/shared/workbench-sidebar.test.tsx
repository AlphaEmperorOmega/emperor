import { render, screen, within } from "@testing-library/react";
import { Activity, Database } from "lucide-react";
import { describe, expect, it } from "vitest";
import {
  WorkbenchSidebarHeader,
  WorkbenchSidebarSection,
  WorkbenchSidebarStack,
} from "@/features/workbench/components/shared/workbench-sidebar";

describe("Workbench sidebar visual contract", () => {
  it("shares Training's stack, header, section rhythm, and icon treatment", () => {
    render(
      <WorkbenchSidebarStack>
        <WorkbenchSidebarHeader
          icon={<Activity data-testid="header-icon" />}
          title="Setup"
          actions={<span>01</span>}
        />
        <WorkbenchSidebarSection
          as="h3"
          title="Datasets"
          icon={<Database data-testid="section-icon" />}
          aside={<span>1 / 4</span>}
          divider="before"
        >
          <button type="button">Select datasets</button>
        </WorkbenchSidebarSection>
      </WorkbenchSidebarStack>,
    );

    const stack = document.querySelector("[data-workbench-sidebar]");
    const header = document.querySelector("[data-workbench-sidebar-header]");
    const section = screen.getByRole("heading", { name: "Datasets" }).closest("section");

    expect(stack).toHaveClass(
      "grid",
      "min-w-0",
      "content-start",
      "gap-region",
    );
    expect(header).toHaveClass(
      "min-h-control",
      "gap-2",
      "border-b",
      "border-line-soft",
      "pb-panel",
    );
    expect(screen.getByRole("heading", { name: "Setup" })).toBeInTheDocument();
    expect(within(header as HTMLElement).getByText("01")).toBeInTheDocument();
    expect(section).toHaveClass(
      "grid",
      "min-w-0",
      "gap-2",
      "border-t",
      "border-line-soft",
      "pt-panel",
    );
    expect(screen.getByTestId("header-icon").parentElement).toHaveClass(
      "shrink-0",
      "text-violet",
      "[&>svg]:h-[15px]",
      "[&>svg]:w-[15px]",
    );
    expect(screen.getByTestId("section-icon").parentElement).toHaveClass(
      "text-violet",
    );
    expect(within(section as HTMLElement).getByText("1 / 4")).toBeInTheDocument();
    expect(
      within(section as HTMLElement).getByRole("button", {
        name: "Select datasets",
      }),
    ).toBeInTheDocument();
  });
});
