import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import {
  ChartDataAction,
  type ChartDataColumn,
} from "@/features/workbench/components/shared/chart-data-dialog";

type FixtureRow = { index: number; value: number };

const columns: readonly ChartDataColumn<FixtureRow>[] = [
  { key: "index", label: "Index", render: (row) => row.index },
  { key: "value", label: "Value", align: "right", render: (row) => row.value },
];

describe("ChartDataAction", () => {
  it("opens a keyboard-accessible dialog and restores focus when it closes", async () => {
    const user = userEvent.setup();
    render(
      <ChartDataAction
        chartTitle="Accuracy"
        columns={columns}
        rows={[{ index: 1, value: 0.75 }]}
      />,
    );

    const action = screen.getByRole("button", { name: "View chart data" });
    await user.click(action);

    const dialog = screen.getByRole("dialog", { name: "Accuracy" });
    expect(dialog).toHaveAccessibleDescription(
      "Complete table of 1 returned row represented by this chart.",
    );
    expect(
      within(dialog).getByRole("button", { name: "Close chart data" }),
    ).toHaveFocus();
    expect(within(dialog).getByRole("table")).toHaveAccessibleName(
      "Accuracy: 1 returned row",
    );

    await user.keyboard("{Escape}");
    expect(dialog).not.toBeInTheDocument();
    expect(action).toHaveFocus();
  });

  it("paginates every returned row in fixed 100-row pages", async () => {
    const user = userEvent.setup();
    const rows = Array.from({ length: 205 }, (_, index) => ({
      index: index + 1,
      value: (index + 1) * 2,
    }));
    render(<ChartDataAction chartTitle="Loss" columns={columns} rows={rows} />);

    await user.click(screen.getByRole("button", { name: "View chart data" }));
    const dialog = screen.getByRole("dialog", { name: "Loss" });
    const table = within(dialog).getByRole("table");
    const tableBodyRows = () => table.querySelectorAll("tbody tr");

    expect(tableBodyRows()).toHaveLength(100);
    expect(within(dialog).getByText("Rows 1–100 of 205 · page 1 of 3"))
      .toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Next" }));
    expect(tableBodyRows()).toHaveLength(100);
    expect(within(tableBodyRows()[0] as HTMLElement).getByText("101"))
      .toBeInTheDocument();
    expect(within(dialog).getByText("Rows 101–200 of 205 · page 2 of 3"))
      .toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Next" }));
    expect(tableBodyRows()).toHaveLength(5);
    expect(within(tableBodyRows()[4] as HTMLElement).getByText("205"))
      .toBeInTheDocument();
    expect(within(dialog).getByText("Rows 201–205 of 205 · page 3 of 3"))
      .toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Next" })).toBeDisabled();
  });

  it("distinguishes complete returned rows from truncated source data", async () => {
    const user = userEvent.setup();
    render(
      <ChartDataAction
        chartTitle="Weights"
        columns={columns}
        rows={[{ index: 1, value: 2 }]}
        completeness={{
          incomplete: true,
          sourceRowCount: 8,
          reason: "Response capped at one row.",
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "View chart data" }));
    expect(screen.getByRole("note")).toHaveTextContent(
      "The API marked this dataset as incomplete",
    );
    expect(screen.getByRole("note")).toHaveTextContent("source reported 8 rows");
    expect(screen.getByRole("note")).toHaveTextContent(
      "Response capped at one row.",
    );
  });
});
