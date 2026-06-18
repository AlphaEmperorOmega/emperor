import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MultiSelectDropdown } from "@/features/viewer/components/screen/multi-select-dropdown";

const createAction = {
  label: "Create snapshot from baseline",
  tooltip: "Create a Config Snapshot from baseline defaults",
  icon: <span aria-hidden>+</span>,
  onAction: vi.fn(),
};

const options = [
  {
    value: "baseline",
    label: "Baseline",
    description: "baseline",
    meta: <span>2 fields</span>,
    actions: [createAction],
  },
  {
    value: "wide",
    label: "Wide",
    description: "wide",
  },
];

function renderDropdown({
  disabledValues = [],
  onAction = vi.fn(),
  onChange = vi.fn(),
  values = [],
}: {
  disabledValues?: string[];
  onAction?: (value: string) => void;
  onChange?: (values: string[]) => void;
  values?: string[];
} = {}) {
  render(
    <MultiSelectDropdown
      label="Targets"
      values={values}
      options={[
        {
          ...options[0],
          actions: [{ ...createAction, onAction }],
        },
        options[1],
      ]}
      onChange={onChange}
      disabledValues={disabledValues}
    />,
  );
  return { onAction, onChange };
}

async function openDropdown(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("combobox", { name: /^targets\b/i }));
  return screen.findByRole("listbox", { name: "Targets options" });
}

describe("MultiSelectDropdown", () => {
  it("renders row actions without adding them to the option accessible name", async () => {
    const user = userEvent.setup();
    renderDropdown();

    const listbox = await openDropdown(user);

    expect(
      within(listbox).getByRole("option", {
        name: "Baseline baseline 2 fields",
      }),
    ).toBeInTheDocument();
    expect(
      within(listbox).queryByRole("option", {
        name: /create snapshot from baseline/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      within(listbox).getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    ).toBeInTheDocument();
  });

  it("runs a mouse action without toggling the row selection", async () => {
    const user = userEvent.setup();
    const { onAction, onChange } = renderDropdown();

    const listbox = await openDropdown(user);
    await user.click(
      within(listbox).getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    );

    expect(onAction).toHaveBeenCalledWith("baseline");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("runs keyboard actions without toggling the row selection", async () => {
    const user = userEvent.setup();
    const { onAction, onChange } = renderDropdown();

    let listbox = await openDropdown(user);
    await user.tab();
    expect(
      within(listbox).getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    ).toHaveFocus();
    await user.keyboard("{Enter}");

    listbox = await openDropdown(user);
    await user.tab();
    expect(
      within(listbox).getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    ).toHaveFocus();
    await user.keyboard(" ");

    expect(onAction).toHaveBeenCalledTimes(2);
    expect(onAction).toHaveBeenNthCalledWith(1, "baseline");
    expect(onAction).toHaveBeenNthCalledWith(2, "baseline");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("shows row action tooltips on hover and focus", async () => {
    const user = userEvent.setup();
    renderDropdown();

    const listbox = await openDropdown(user);
    const action = within(listbox).getByRole("button", {
      name: "Create snapshot from baseline",
    });

    await user.hover(action);
    let tooltip = await screen.findByRole("tooltip");
    expect(tooltip).toHaveTextContent(
      "Create a Config Snapshot from baseline defaults",
    );
    expect(tooltip).not.toHaveClass("sr-only");

    await user.unhover(action);
    await waitFor(() => {
      expect(screen.getByRole("tooltip")).toHaveClass("sr-only");
    });

    await user.tab();
    expect(action).toHaveFocus();
    tooltip = await screen.findByRole("tooltip");
    expect(tooltip).toHaveTextContent(
      "Create a Config Snapshot from baseline defaults",
    );
    expect(tooltip).not.toHaveClass("sr-only");
    expect(action).toHaveAccessibleDescription(
      "Create a Config Snapshot from baseline defaults",
    );
  });

  it("closes before the row action callback runs", async () => {
    const user = userEvent.setup();
    const onAction = vi.fn(() => {
      expect(
        screen.queryByRole("listbox", { name: "Targets options" }),
      ).not.toBeInTheDocument();
    });
    renderDropdown({ onAction });

    const listbox = await openDropdown(user);
    await user.click(
      within(listbox).getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    );

    expect(onAction).toHaveBeenCalledWith("baseline");
  });

  it("keeps row actions available when row selection is disabled", async () => {
    const user = userEvent.setup();
    const { onAction, onChange } = renderDropdown({
      values: ["baseline"],
      disabledValues: ["baseline"],
    });

    const listbox = await openDropdown(user);
    expect(
      within(listbox).getByRole("option", { name: "Baseline baseline 2 fields" }),
    ).toHaveAttribute("aria-disabled", "true");

    await user.click(
      within(listbox).getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    );

    expect(onAction).toHaveBeenCalledWith("baseline");
    expect(onChange).not.toHaveBeenCalled();
  });
});
