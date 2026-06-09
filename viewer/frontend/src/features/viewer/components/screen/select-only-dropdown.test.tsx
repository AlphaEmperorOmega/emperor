import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";

const options = [
  { value: "linear", label: "Linear" },
  { value: "neuron", label: "Neuron" },
  { value: "attention", label: "Attention" },
];

describe("SelectOnlyDropdown", () => {
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
});
