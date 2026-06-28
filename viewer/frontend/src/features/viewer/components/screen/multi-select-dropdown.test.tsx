import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
  render(
    <MultiSelectDropdown
      label="Targets"
      values={[]}
      options={manyOptions(18)}
      onChange={onChange}
      initialVisibleCount={5}
      pageSize={5}
    />,
  );
  return { onChange };
}

async function openDropdown(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("combobox", { name: /^targets\b/i }));
  return screen.findByRole("listbox", { name: "Targets options" });
}

describe("MultiSelectDropdown", () => {
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

    expect(screen.getByRole("combobox", { name: /^targets\b/i })).toHaveClass(
      "compact-trigger",
      "h-7",
      "px-1",
    );
  });

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

    await user.click(screen.getByRole("combobox", { name: /^experiments\b/i }));
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
      "group-focus-within:opacity-100",
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

  it("renders the first lazy page before scrolling", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    const listbox = await openDropdown(user);

    expect(within(listbox).getAllByRole("option")).toHaveLength(5);
    expect(
      within(listbox).getByRole("option", { name: /target 05/i }),
    ).toBeInTheDocument();
    expect(
      within(listbox).queryByRole("option", { name: /target 06/i }),
    ).not.toBeInTheDocument();
  });

  it("loads the next lazy page when the listbox scrolls near the bottom", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    const listbox = await openDropdown(user);
    setScrollable(listbox);
    fireEvent.scroll(listbox);

    expect(
      within(listbox).getByRole("status", { name: /loading more targets/i }),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 10/i }),
      ).toBeInTheDocument();
    });
  });

  it("resets the lazy window after searching and can append filtered rows", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    let listbox = await openDropdown(user);
    setScrollable(listbox);
    fireEvent.scroll(listbox);
    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 10/i }),
      ).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText("Search Targets"), "Target 1");

    listbox = screen.getByRole("listbox", { name: "Targets options" });
    expect(
      within(listbox).getByRole("option", { name: /target 10/i }),
    ).toBeInTheDocument();
    expect(
      within(listbox).queryByRole("option", { name: /target 15/i }),
    ).not.toBeInTheDocument();

    setScrollable(listbox);
    fireEvent.scroll(listbox);
    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 15/i }),
      ).toBeInTheDocument();
    });
  });

  it("loads another page when keyboard navigation moves past the last rendered row", async () => {
    const user = userEvent.setup();
    renderLazyDropdown();

    const listbox = await openDropdown(user);
    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 01/i }),
      ).toHaveFocus();
    });

    await user.keyboard("{End}");
    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 05/i }),
      ).toHaveFocus();
    });
    await user.keyboard("{ArrowDown}");

    await waitFor(() => {
      expect(
        within(listbox).getByRole("option", { name: /target 06/i }),
      ).toHaveFocus();
    });
  });
});
