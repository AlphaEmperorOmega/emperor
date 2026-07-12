import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SelectOnlyDropdown } from "@/features/workbench/components/screen/select-only-dropdown";

const options = [
  { value: "linear", label: "Linear" },
  { value: "neuron", label: "Neuron" },
  { value: "attention", label: "Attention" },
];

describe("SelectOnlyDropdown", () => {
  it("only references popup descendants while they are mounted", async () => {
    const user = userEvent.setup();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={() => {}}
      />,
    );

    const control = screen.getByRole("combobox", { name: "Model" });
    expect(control).not.toHaveAttribute("aria-controls");
    expect(control).not.toHaveAttribute("aria-activedescendant");

    await user.click(control);
    const popup = screen.getByRole("listbox", { name: "Model options" });
    expect(control).toHaveAttribute("aria-controls", popup.id);
    expect(document.getElementById(control.getAttribute("aria-activedescendant") ?? ""))
      .toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(popup).not.toBeInTheDocument();
    expect(control).not.toHaveAttribute("aria-controls");
    expect(control).not.toHaveAttribute("aria-activedescendant");
  });

  it("filters options with search and shows an empty result message", async () => {
    const user = userEvent.setup();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={[
          ...options,
          {
            value: "mixture_of_attention_heads",
            label: "Mixture of Attention Heads",
            description: "moah preset",
          },
        ]}
        onChange={() => {}}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Model" }));
    const search = screen.getByRole("searchbox", { name: "Search Model" });

    await user.type(search, "moah");

    expect(
      screen.getByRole("option", { name: /mixture of attention heads/i }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Linear" })).not.toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "missing");

    expect(screen.getByText("No matching options")).toBeInTheDocument();
  });

  it("selects a filtered option by click", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Model" }));
    await user.type(screen.getByRole("searchbox", { name: "Search Model" }), "neu");
    await user.click(screen.getByRole("option", { name: "Neuron" }));

    expect(onChange).toHaveBeenCalledWith("neuron");
  });

  it("keeps keyboard focus separate from the selected option", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={onChange}
      />,
    );

    const control = screen.getByRole("combobox", { name: "Model" });
    await user.click(control);

    const listbox = screen.getByRole("listbox", { name: "Model options" });
    const selectedOption = within(listbox).getByRole("option", { name: "Linear" });
    const nextOption = within(listbox).getByRole("option", { name: "Neuron" });

    expect(selectedOption).toHaveAttribute("aria-selected", "true");
    expect(nextOption).toHaveAttribute("aria-selected", "false");

    await user.keyboard("{ArrowDown}");

    expect(control).toHaveAttribute("aria-activedescendant", nextOption.id);
    expect(selectedOption).toHaveAttribute("aria-selected", "true");
    expect(nextOption).toHaveAttribute("aria-selected", "false");

    await user.keyboard("{Enter}");

    expect(onChange).toHaveBeenCalledWith("neuron");
    expect(screen.queryByRole("listbox", { name: "Model options" })).not.toBeInTheDocument();
  });

  it("clears the search query after closing and reopening", async () => {
    const user = userEvent.setup();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={() => {}}
      />,
    );

    const control = screen.getByRole("combobox", { name: "Model" });
    await user.click(control);
    await user.type(screen.getByRole("searchbox", { name: "Search Model" }), "att");

    expect(screen.queryByRole("option", { name: "Linear" })).not.toBeInTheDocument();

    await user.keyboard("{Escape}");
    await user.click(control);

    expect(screen.getByRole("searchbox", { name: "Search Model" })).toHaveValue("");
    expect(screen.getByRole("option", { name: "Linear" })).toBeInTheDocument();
  });

  it("renders disabled options without allowing selection", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={[
          { value: "linear", label: "Linear" },
          { value: "neuron", label: "Neuron", disabled: true },
        ]}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Model" }));

    const disabledOption = screen.getByRole("option", { name: "Neuron" });
    expect(disabledOption).toHaveAttribute("aria-disabled", "true");

    await user.click(disabledOption);

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole("listbox", { name: "Model options" })).toBeInTheDocument();
  });

  it("honors an explicit disabled state when options are available", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={onChange}
        disabled
      />,
    );

    const control = screen.getByRole("combobox", { name: "Model" });
    expect(control).toBeDisabled();

    await user.click(control);

    expect(screen.queryByRole("listbox", { name: "Model options" })).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});
