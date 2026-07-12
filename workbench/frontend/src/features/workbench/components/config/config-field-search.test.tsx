import { useState } from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ConfigFieldSearch } from "@/features/workbench/components/config/config-field-search";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { type RuntimeDefaultsSearchOptionPresentation } from "@/features/workbench/state/full-config/runtime-defaults-schema-presentation";

function option(index: number): RuntimeDefaultsSearchOptionPresentation {
  const key = `field_${index}`;
  const label = `Field ${index}`;
  return {
    key,
    label,
    configKey: key.toUpperCase(),
    flag: `--${key.replaceAll("_", "-")}`,
    type: "int",
    sectionTitle: "Model",
    rootSectionTitle: "Model",
    field: {
      schema: {
        key,
        configKey: key.toUpperCase(),
        flag: `--${key.replaceAll("_", "-")}`,
        label,
        section: "Model",
        sectionPath: ["Model"],
        type: "int",
        default: 1,
        nullable: false,
        choices: [],
      },
      key,
      label,
      value: String(index),
      selectOptions: [],
      isModified: index === 2,
      isPresetOwned: false,
      isEnabledValue: false,
    },
  };
}

const options = Array.from({ length: 10 }, (_, index) => option(index + 1));

function Harness({ source = options }: { source?: typeof options }) {
  const [query, setQuery] = useState("");
  return (
    <>
      <ConfigFieldSearch
        options={source}
        query={query}
        selectedFieldKey={null}
        matchesQuery={(candidate, value) =>
          `${candidate.label} ${candidate.key}`
            .toLowerCase()
            .includes(value.toLowerCase())
        }
        onQueryChange={setQuery}
        onClear={() => setQuery("")}
        onSelect={vi.fn()}
        onFieldChange={vi.fn()}
        onFieldReset={vi.fn()}
      />
      <button type="button">Outside search</button>
    </>
  );
}

describe("ConfigFieldSearch", () => {
  it("moves real focus through result titles, then tabs through the row editor", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const search = screen.getByRole("combobox", { name: "Search config fields" });

    await user.type(search, "field");
    const popup = screen.getByRole("dialog", { name: "Matching config fields" });
    expect(screen.getByRole("status")).toHaveTextContent(
      "10 matching config fields.",
    );

    await user.keyboard("{ArrowDown}");
    expect(within(popup).getByRole("button", { name: "Field 1" })).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(within(popup).getByRole("button", { name: "Field 2" })).toHaveFocus();
    await user.tab();
    expect(
      within(popup).getByRole("textbox", { name: "Field 2 current value" }),
    ).toHaveFocus();
    await user.tab();
    expect(
      within(popup).getByRole("button", {
        name: "Reset Field 2 search result override",
      }),
    ).toHaveFocus();
  });

  it("restores search focus on Escape and closes when focus leaves normally", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const search = screen.getByRole("combobox", { name: "Search config fields" });

    await user.type(search, "field");
    await user.keyboard("{ArrowDown}");
    await user.keyboard("{Escape}");
    expect(search).toHaveFocus();
    expect(screen.queryByRole("dialog", { name: "Matching config fields" }))
      .not.toBeInTheDocument();

    await user.click(search);
    expect(screen.getByRole("dialog", { name: "Matching config fields" }))
      .toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Outside search" }));
    expect(screen.queryByRole("dialog", { name: "Matching config fields" }))
      .not.toBeInTheDocument();
  });

  it("dismisses from an editor without closing its parent dialog", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <DialogShell ariaLabel="Parent config dialog" onClose={onClose}>
        <Harness />
      </DialogShell>,
    );
    const search = screen.getByRole("combobox", {
      name: "Search config fields",
    });

    await user.type(search, "field");
    await user.keyboard("{ArrowDown}");
    await user.tab();
    expect(
      screen.getByRole("textbox", { name: "Field 1 current value" }),
    ).toHaveFocus();

    await user.keyboard("{Escape}");

    expect(onClose).not.toHaveBeenCalled();
    expect(
      screen.getByRole("dialog", { name: "Parent config dialog" }),
    ).toBeInTheDocument();
    expect(search).toHaveFocus();
    expect(
      screen.queryByRole("dialog", { name: "Matching config fields" }),
    ).not.toBeInTheDocument();
  });

  it("loads and focuses the next lazy result page with ArrowDown", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const search = screen.getByRole("combobox", { name: "Search config fields" });

    await user.type(search, "field");
    await user.keyboard("{ArrowUp}");
    expect(screen.getByRole("button", { name: "Field 8" })).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Field 9" })).toHaveFocus();
    });
  });

  it("announces an empty result set without exposing a visual active row", async () => {
    const user = userEvent.setup();
    render(<Harness source={[]} />);
    const search = screen.getByRole("combobox", { name: "Search config fields" });

    await user.type(search, "missing");
    expect(screen.getByRole("status")).toHaveTextContent(
      "0 matching config fields.",
    );
    expect(screen.getByText("No matching fields")).toBeInTheDocument();
    await user.keyboard("{ArrowDown}");
    expect(search).toHaveFocus();
  });
});
