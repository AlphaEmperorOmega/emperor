import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MultiSelectDropdown } from "@/features/workbench/components/screen/multi-select-dropdown";

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

function manyOptions(count: number) {
  return Array.from({ length: count }, (_, index) => {
    const number = String(index + 1).padStart(2, "0");
    return {
      value: `target-${number}`,
      label: `Target ${number}`,
      description: `target-${number}`,
    };
  });
}

function setScrollable(element: HTMLElement) {
  Object.defineProperty(element, "clientHeight", {
    configurable: true,
    value: 120,
  });
  Object.defineProperty(element, "scrollHeight", {
    configurable: true,
    value: 800,
  });
  element.scrollTop = 700;
}

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

function renderLazyDropdown() {
  const onChange = vi.fn();
  const renderWithOptions = (nextOptions: ReturnType<typeof manyOptions>) => (
    <MultiSelectDropdown
      label="Targets"
      values={[]}
      options={nextOptions}
      onChange={onChange}
      initialVisibleCount={5}
      pageSize={5}
    />
  );
  const rendered = render(renderWithOptions(manyOptions(18)));
  return {
    onChange,
    rerenderOptions: (nextOptions: ReturnType<typeof manyOptions>) =>
      rendered.rerender(renderWithOptions(nextOptions)),
  };
}

async function openDropdown(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /^targets\b/i }));
  return screen.findByRole("listbox", { name: "Targets options" });
}

describe("MultiSelectDropdown", () => {
  it("only references popup descendants while they are mounted", async () => {
    const user = userEvent.setup();
    renderDropdown();

    const control = screen.getByRole("button", { name: /^targets\b/i });
    expect(control).not.toHaveAttribute("aria-controls");
    expect(control).not.toHaveAttribute("aria-activedescendant");

    const popup = await openDropdown(user);
    const search = screen.getByRole("combobox", { name: "Search Targets" });
    expect(control).toHaveAttribute("aria-controls", popup.id);
    expect(document.getElementById(search.getAttribute("aria-activedescendant") ?? ""))
      .toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(popup).not.toBeInTheDocument();
    expect(control).not.toHaveAttribute("aria-controls");
    expect(control).not.toHaveAttribute("aria-activedescendant");
  });

  it("applies custom trigger classes after the default trigger styling", () => {
    const onChange = vi.fn();
    render(
      <MultiSelectDropdown
        label="Targets"
        values={["baseline"]}
        options={options}
        onChange={onChange}
        triggerClassName="compact-trigger h-7 px-1"
      />,
    );

    expect(screen.getByRole("button", { name: /^targets\b/i })).toHaveClass(
      "compact-trigger",
      "h-7",
      "px-1",
    );
  });

  it("renders active-option actions in a sibling toolbar, never inside the listbox", async () => {
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
      screen.getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    ).toBeInTheDocument();
    expect(within(listbox).queryByRole("button")).not.toBeInTheDocument();
    expect(
      screen.getByRole("toolbar", { name: "Baseline actions" }),
    ).toBeInTheDocument();
  });

  it("can wrap long option labels while exposing metadata as hover-only context", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectDropdown
        label="Experiments"
        values={[]}
        options={[
          {
            value: "same_prefix_with_a_very_long_suffix_alpha",
            label: "same_prefix_with_a_very_long_suffix_alpha",
            metaTooltip: "12 runs",
            wrapLabel: true,
          },
        ]}
        onChange={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /^experiments\b/i }));
    const listbox = await screen.findByRole("listbox", {
      name: "Experiments options",
    });
    const option = within(listbox).getByRole("option", {
      name: "same_prefix_with_a_very_long_suffix_alpha",
    });

    expect(option).toHaveAccessibleDescription("12 runs");
    expect(
      within(option).getByText("same_prefix_with_a_very_long_suffix_alpha"),
    ).toHaveClass("whitespace-normal", "break-words", "[overflow-wrap:anywhere]");
    expect(screen.getByRole("tooltip")).toHaveClass(
      "opacity-0",
      "group-hover:opacity-100",
      "group-focus:opacity-100",
    );
    expect(
      within(listbox).queryByRole("option", {
        name: /12 runs/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("runs a mouse action without toggling the row selection", async () => {
    const user = userEvent.setup();
    const { onAction, onChange } = renderDropdown();

    await openDropdown(user);
    await user.click(
      screen.getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    );

    expect(onAction).toHaveBeenCalledWith("baseline");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("runs keyboard actions without toggling the row selection", async () => {
    const user = userEvent.setup();
    const { onAction, onChange } = renderDropdown();

    const listbox = await openDropdown(user);
    await user.tab();
    expect(
      screen.getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    ).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(listbox).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
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

    await openDropdown(user);
    const action = screen.getByRole("button", {
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

  it("keeps focus and the popup while the toolbar action runs", async () => {
    const user = userEvent.setup();
    const onAction = vi.fn();
    renderDropdown({ onAction });

    const listbox = await openDropdown(user);
    await user.click(
      screen.getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    );

    expect(onAction).toHaveBeenCalledWith("baseline");
    expect(listbox).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Create snapshot from baseline" }),
    ).toHaveFocus();
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
      screen.getByRole("button", {
        name: "Create snapshot from baseline",
      }),
    );

    expect(onAction).toHaveBeenCalledWith("baseline");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("renders a bounded virtual window before scrolling", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    const listbox = await openDropdown(user);
    const renderedOptions = within(listbox).getAllByRole("option");

    expect(renderedOptions.length).toBeGreaterThan(0);
    expect(renderedOptions.length).toBeLessThan(18);
    expect(
      within(listbox).queryByRole("option", { name: /target 18/i }),
    ).not.toBeInTheDocument();
  });

  it("moves the virtual window when the listbox scrolls", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    const listbox = await openDropdown(user);
    setScrollable(listbox);
    fireEvent.scroll(listbox);

    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 18/i }),
      ).toBeInTheDocument();
    });
    expect(
      within(listbox).queryByRole("option", { name: /target 01/i }),
    ).not.toBeInTheDocument();
  });

  it("preserves the measured virtual position for value-equivalent options", async () => {
    const user = userEvent.setup();
    const { rerenderOptions } = renderLazyDropdown();

    const listbox = await openDropdown(user);
    setScrollable(listbox);
    fireEvent.scroll(listbox);
    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 18/i }),
      ).toBeInTheDocument();
    });

    rerenderOptions(manyOptions(18));

    expect(
      within(listbox).getByRole("option", { name: /target 18/i }),
    ).toBeInTheDocument();
  });

  it("resets the virtual window when the search changes", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    let listbox = await openDropdown(user);
    setScrollable(listbox);
    fireEvent.scroll(listbox);
    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 18/i }),
      ).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText("Search Targets"), "Target 1");

    listbox = screen.getByRole("listbox", { name: "Targets options" });
    expect(
      within(listbox).getByRole("option", { name: /target 10/i }),
    ).toBeInTheDocument();
    expect(within(listbox).queryByText(/loading more/i)).not.toBeInTheDocument();
  });

  it("keeps combobox focus while keyboard navigation reveals a distant row", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    const listbox = await openDropdown(user);
    const search = screen.getByRole("combobox", { name: "Search Targets" });

    await user.keyboard("{End}");
    await waitFor(() => {
      const lastOption = within(listbox).getByRole("option", {
        name: /target 18/i,
      });
      expect(search).toHaveAttribute("aria-activedescendant", lastOption.id);
    });
    expect(search).toHaveFocus();
  });
});
