import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { describe, expect, it, vi } from "vitest";
import { ConfigFieldSearch } from "@/features/workbench/components/config/config-field-search";
import { MultiSelectDropdown } from "@/features/workbench/components/screen/multi-select-dropdown";
import { SelectOnlyDropdown } from "@/features/workbench/components/screen/select-only-dropdown";
import { type RuntimeDefaultsSearchOptionPresentation } from "@/features/workbench/state/runtime-defaults/runtime-defaults-presentation";

const configOption: RuntimeDefaultsSearchOptionPresentation = {
  key: "hidden_dim",
  label: "Hidden Dim",
  configKey: "HIDDEN_DIM",
  flag: "--hidden-dim",
  type: "int",
  sectionTitle: "Model",
  rootSectionTitle: "Model",
  field: {
    schema: {
      key: "hidden_dim",
      configKey: "HIDDEN_DIM",
      flag: "--hidden-dim",
      label: "Hidden Dim",
      section: "Model",
      sectionPath: ["Model"],
      type: "int",
      default: 64,
      nullable: false,
      choices: [],
    },
    key: "hidden_dim",
    label: "Hidden Dim",
    value: "64",
    selectOptions: [],
    isModified: false,
    isPresetOwned: false,
    isEnabledValue: false,
  },
};

async function expectNoAccessibilityViolations(container: HTMLElement) {
  const result = await axe.run(container, {
    rules: {
      "color-contrast": { enabled: false },
    },
  });
  expect(result.violations).toEqual([]);
}

function SearchableControlsHarness() {
  const [model, setModel] = useState("linear");
  const [targets, setTargets] = useState<string[]>([]);
  const [configQuery, setConfigQuery] = useState("");

  return (
    <main>
      <SelectOnlyDropdown
        id="accessible-model"
        label="Model"
        value={model}
        options={[
          { value: "linear", label: "Linear" },
          { value: "attention", label: "Attention" },
        ]}
        onChange={setModel}
      />
      <MultiSelectDropdown
        id="accessible-targets"
        label="Targets"
        values={targets}
        options={[
          { value: "mnist", label: "Mnist" },
          { value: "cifar10", label: "Cifar10" },
        ]}
        onChange={setTargets}
      />
      <ConfigFieldSearch
        options={[configOption]}
        query={configQuery}
        selectedFieldKey={null}
        matchesQuery={(option, query) =>
          `${option.label} ${option.key}`
            .toLowerCase()
            .includes(query.toLowerCase())
        }
        onQueryChange={setConfigQuery}
        onClear={() => setConfigQuery("")}
        onSelect={vi.fn()}
        onFieldChange={vi.fn()}
        onFieldReset={vi.fn()}
      />
    </main>
  );
}

describe("searchable control accessibility", () => {
  it("has no automated violations while controls are closed", async () => {
    const { container } = render(<SearchableControlsHarness />);

    await expectNoAccessibilityViolations(container);
  });

  it("has no automated violations for open listbox and config-search states", async () => {
    const user = userEvent.setup();
    const { container } = render(<SearchableControlsHarness />);

    await user.click(screen.getByRole("button", { name: "Model" }));
    await expectNoAccessibilityViolations(container);

    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: /^Targets\b/ }));
    await expectNoAccessibilityViolations(container);

    await user.keyboard("{Escape}");
    await user.type(
      screen.getByRole("combobox", { name: "Search config fields" }),
      "hidden",
    );
    await expectNoAccessibilityViolations(container);
  });
});
