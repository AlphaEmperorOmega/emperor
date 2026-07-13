import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { useSearchableDialogInteraction } from "@/features/workbench/components/shared/use-searchable-dialog";
import { useSearchablePopupInteraction } from "@/features/workbench/components/shared/use-searchable-dropdown";

type TestOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

const testOptions: TestOption[] = [
  { value: "alpha", label: "Alpha" },
  { value: "beta", label: "Beta", disabled: true },
  { value: "gamma", label: "Gamma" },
  { value: "delta", label: "Delta" },
];

function ListboxHarness({
  mode,
  onActivate,
  initialVisibleCount = testOptions.length,
  options = testOptions,
}: {
  mode: "single-select" | "multi-select";
  onActivate: (option: TestOption) => void;
  initialVisibleCount?: number;
  options?: TestOption[];
}) {
  const interaction = useSearchablePopupInteraction<
    TestOption,
    HTMLButtonElement
  >({
    mode,
    id: `${mode}-test`,
    idSuffix: "unused",
    options,
    optionKey: (option) => option.value,
    optionSearchText: (option) => `${option.label} ${option.value}`,
    selectedKey: mode === "single-select" ? "alpha" : undefined,
    isOptionDisabled: (option) => Boolean(option.disabled),
    onActivate,
    pagination:
      mode === "multi-select"
        ? { initialVisibleCount, pageSize: initialVisibleCount }
        : undefined,
  });
  const { ids, state, root, trigger, search, collection } = interaction;

  return (
    <>
      <button type="button">Outside</button>
      <div ref={root.ref} onBlur={root.onBlur}>
        <button
          ref={trigger.ref}
          type="button"
          role="combobox"
          aria-label="Test choices"
          aria-expanded={state.isOpen}
          aria-controls={ids.popup}
          aria-activedescendant={ids.active}
          onClick={trigger.onClick}
          onKeyDown={trigger.onKeyDown}
        >
          Choices
        </button>
        {state.isOpen && (
          <div>
            <input
              ref={search.ref}
              type="search"
              aria-label="Search choices"
              value={state.query}
              onChange={search.onChange}
              onKeyDown={search.onKeyDown}
            />
            <div
              ref={collection.ref}
              id={ids.popup}
              role="listbox"
              aria-label="Test options"
              onScroll={collection.onScroll}
            >
              {state.options.map((option, index) => (
                <button
                  {...collection.option(index, option)}
                  key={option.value}
                  id={`${ids.popup}-option-${index}`}
                  type="button"
                  role="option"
                  aria-selected="false"
                  aria-disabled={option.disabled || undefined}
                >
                  {option.label}
                </button>
              ))}
              {state.loading && <span role="status">Loading</span>}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function SearchDialogHarness({
  onActivate,
  onClear,
}: {
  onActivate: (option: TestOption) => void;
  onClear: () => void;
}) {
  const [query, setQuery] = useState("");
  const interaction = useSearchableDialogInteraction<TestOption>({
    id: "dialog-test",
    idSuffix: "unused",
    options: testOptions,
    optionKey: (option) => option.value,
    matchesQuery: (option, nextQuery) =>
      option.label.toLowerCase().includes(nextQuery.toLowerCase()),
    query,
    onQueryChange: setQuery,
    onClear: () => {
      onClear();
      setQuery("");
    },
    onActivate,
    initialVisibleCount: 2,
    pageSize: 2,
  });
  const { ids, state, root, search, popup, collection, actions } = interaction;

  return (
    <div ref={root.ref} onFocus={root.onFocus} onBlur={root.onBlur}>
      <input
        ref={search.ref}
        role="combobox"
        aria-label="Search records"
        aria-expanded={state.isOpen}
        aria-controls={ids.popup}
        value={query}
        onChange={search.onChange}
        onKeyDown={search.onKeyDown}
      />
      {query && (
        <button
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={search.clear}
        >
          Clear
        </button>
      )}
      {state.isOpen && (
        <div
          ref={popup.ref}
          id={ids.popup}
          role="dialog"
          aria-label="Matching records"
          onKeyDown={popup.onKeyDown}
          onScroll={collection.onScroll}
        >
          {state.visibleOptions.map((option, index) => (
            <button
              {...actions.optionTitle(option, index)}
              key={option.value}
              type="button"
              data-active={index === state.activeIndex || undefined}
              onClick={() => actions.activate(option)}
            >
              {option.label}
            </button>
          ))}
          {state.isLoadingMore && <span role="status">Loading</span>}
        </div>
      )}
    </div>
  );
}

describe("useSearchablePopupInteraction", () => {
  it("owns listbox focus, disabled activation, selection, and focus restoration", async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();
    render(<ListboxHarness mode="single-select" onActivate={onActivate} />);

    const trigger = screen.getByRole("combobox", { name: "Test choices" });
    await user.click(trigger);
    expect(screen.getByRole("searchbox", { name: "Search choices" })).toHaveFocus();

    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Beta" })).toHaveFocus();
    });
    await user.keyboard("{Enter}");
    expect(onActivate).not.toHaveBeenCalled();
    expect(screen.getByRole("listbox", { name: "Test options" })).toBeInTheDocument();

    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Gamma" })).toHaveFocus();
    });
    await user.keyboard("{Enter}");
    expect(onActivate).toHaveBeenCalledWith(testOptions[2]);
    expect(screen.queryByRole("listbox", { name: "Test options" })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("resets search and dismisses a listbox on an outside pointer", async () => {
    const user = userEvent.setup();
    render(<ListboxHarness mode="single-select" onActivate={() => {}} />);

    const trigger = screen.getByRole("combobox", { name: "Test choices" });
    await user.click(trigger);
    await user.type(screen.getByRole("searchbox", { name: "Search choices" }), "gam");
    expect(screen.getAllByRole("option")).toHaveLength(1);

    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole("listbox", { name: "Test options" })).not.toBeInTheDocument();

    await user.click(trigger);
    expect(screen.getByRole("searchbox", { name: "Search choices" })).toHaveValue("");
    expect(screen.getAllByRole("option")).toHaveLength(testOptions.length);
  });

  it("loads and focuses the next page during option keyboard navigation", async () => {
    const user = userEvent.setup();
    render(
      <ListboxHarness
        mode="multi-select"
        onActivate={() => {}}
        initialVisibleCount={2}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Test choices" }));
    const listbox = screen.getByRole("listbox", { name: "Test options" });
    expect(within(listbox).getAllByRole("option")).toHaveLength(2);

    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(within(listbox).getByRole("option", { name: "Alpha" })).toHaveFocus();
    });
    await user.keyboard("{End}");
    await waitFor(() => {
      expect(within(listbox).getByRole("option", { name: "Beta" })).toHaveFocus();
    });
    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(within(listbox).getByRole("option", { name: "Gamma" })).toHaveFocus();
    });
  });

  it("resets active identity when the matching option lifecycle changes", async () => {
    const user = userEvent.setup();
    const rendered = render(
      <ListboxHarness mode="multi-select" onActivate={() => {}} />,
    );

    const trigger = screen.getByRole("combobox", { name: "Test choices" });
    await user.click(trigger);
    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Alpha" })).toHaveFocus();
    });
    await user.keyboard("{End}");
    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Delta" })).toHaveFocus();
      expect(trigger).toHaveAttribute(
        "aria-activedescendant",
        screen.getByRole("option", { name: "Delta" }).id,
      );
    });

    rendered.rerender(
      <ListboxHarness
        mode="multi-select"
        onActivate={() => {}}
        options={[testOptions[2], testOptions[0]]}
      />,
    );
    await waitFor(() => {
      expect(trigger).toHaveAttribute(
        "aria-activedescendant",
        screen.getByRole("option", { name: "Gamma" }).id,
      );
    });
  });

  it("moves real dialog focus through lazy rows while paging and activating", async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();
    const onClear = vi.fn();
    render(<SearchDialogHarness onActivate={onActivate} onClear={onClear} />);

    const search = screen.getByRole("combobox", { name: "Search records" });
    await user.type(search, "a");
    let dialog = screen.getByRole("dialog", { name: "Matching records" });
    expect(within(dialog).getAllByRole("button")).toHaveLength(2);

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Matching records" })).not.toBeInTheDocument();
    expect(search).toHaveValue("a");
    expect(search).toHaveFocus();

    await user.keyboard("{ArrowDown}");
    dialog = screen.getByRole("dialog", { name: "Matching records" });
    await user.keyboard("{ArrowDown}");
    await user.keyboard("{ArrowDown}");
    await waitFor(() => {
      expect(within(dialog).getByRole("button", { name: "Gamma" })).toHaveAttribute(
        "data-active",
        "true",
      );
    });
    expect(within(dialog).getByRole("button", { name: "Gamma" })).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(onActivate).toHaveBeenCalledWith(testOptions[2]);
    expect(screen.queryByRole("dialog", { name: "Matching records" })).not.toBeInTheDocument();
    expect(search).toHaveValue("a");

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(onClear).toHaveBeenCalledOnce();
    expect(search).toHaveValue("");
    expect(search).toHaveFocus();
  });
});
