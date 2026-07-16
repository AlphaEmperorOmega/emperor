import { fireEvent, render, screen, within } from "@testing-library/react";
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
  const interaction = useSearchablePopupInteraction<TestOption>({
    mode,
    id: `${mode}-test`,
    idSuffix: "unused",
    options,
    optionKey: (option) => option.value,
    optionSearchText: (option) => `${option.label} ${option.value}`,
    selectedKey: mode === "single-select" ? "alpha" : undefined,
    isOptionDisabled: (option) => Boolean(option.disabled),
    onActivate,
    pagination: { initialVisibleCount, pageSize: initialVisibleCount },
  });
  const {
    controlId,
    searchId,
    popupId,
    activeOptionId,
    isOpen,
    query,
    activeIndex,
    virtualOptions,
    beforeHeight,
    afterHeight,
    rootRef,
    triggerRef,
    searchRef,
    collectionRef,
    measureOption,
    handleRootBlur,
    handleTriggerClick,
    handleTriggerKeyDown,
    handleSearchChange,
    handleSearchKeyDown,
    handleCollectionScroll,
    handleOptionMouseDown,
    handleOptionMouseEnter,
    handleOptionClick,
  } = interaction;

  return (
    <>
      <button type="button">Outside</button>
      <div ref={rootRef} onBlur={handleRootBlur}>
        <button
          ref={triggerRef}
          id={controlId}
          type="button"
          aria-label="Test choices"
          aria-haspopup="listbox"
          aria-expanded={isOpen}
          aria-controls={isOpen ? popupId : undefined}
          onClick={handleTriggerClick}
          onKeyDown={handleTriggerKeyDown}
        >
          Choices
        </button>
        {isOpen && (
          <div>
            <input
              ref={searchRef}
              id={searchId}
              role="combobox"
              aria-label="Search choices"
              aria-expanded={isOpen}
              aria-controls={popupId}
              aria-activedescendant={activeOptionId}
              value={query}
              onChange={handleSearchChange}
              onKeyDown={handleSearchKeyDown}
            />
            <div
              ref={collectionRef}
              id={popupId}
              role="listbox"
              aria-label="Test options"
              onScroll={handleCollectionScroll}
            >
              {beforeHeight > 0 && (
                <div aria-hidden style={{ height: beforeHeight }} />
              )}
              {virtualOptions.map(({ option, index, key }) => (
                <div
                  ref={measureOption}
                  key={key}
                  id={`${popupId}-option-${encodeURIComponent(key)}`}
                  data-virtual-option-key={key}
                  role="option"
                  tabIndex={-1}
                  aria-selected="false"
                  aria-disabled={option.disabled || undefined}
                  data-active={index === activeIndex || undefined}
                  onMouseDown={handleOptionMouseDown}
                  onMouseEnter={() => handleOptionMouseEnter(index)}
                  onClick={() => handleOptionClick(index)}
                >
                  {option.label}
                </div>
              ))}
              {afterHeight > 0 && (
                <div aria-hidden style={{ height: afterHeight }} />
              )}
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
  const {
    controlId,
    popupId,
    activeOptionId,
    isOpen,
    activeIndex,
    virtualOptions,
    beforeHeight,
    afterHeight,
    rootRef,
    searchRef,
    collectionRef,
    measureOption,
    handleRootFocus,
    handleRootBlur,
    handleSearchFocus,
    handleSearchChange,
    handleSearchKeyDown,
    handlePopupKeyDown,
    handleCollectionScroll,
    handleOptionMouseEnter,
    clearSearch,
    activate,
  } = interaction;

  return (
    <div ref={rootRef} onFocus={handleRootFocus} onBlur={handleRootBlur}>
      <input
        ref={searchRef}
        id={controlId}
        role="combobox"
        aria-label="Search records"
        aria-expanded={isOpen}
        aria-controls={isOpen ? popupId : undefined}
        aria-activedescendant={isOpen ? activeOptionId : undefined}
        value={query}
        onFocus={handleSearchFocus}
        onChange={handleSearchChange}
        onKeyDown={handleSearchKeyDown}
      />
      {query && (
        <button
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={clearSearch}
        >
          Clear
        </button>
      )}
      {isOpen && (
        <div
          ref={collectionRef}
          id={popupId}
          role="dialog"
          aria-label="Matching records"
          onKeyDown={handlePopupKeyDown}
          onScroll={handleCollectionScroll}
        >
          {beforeHeight > 0 && (
            <div aria-hidden style={{ height: beforeHeight }} />
          )}
          {virtualOptions.map(({ option, index, key }) => (
            <button
              ref={measureOption}
              key={key}
              id={`${popupId}-option-${encodeURIComponent(key)}`}
              type="button"
              tabIndex={-1}
              data-virtual-option-key={key}
              data-active={index === activeIndex || undefined}
              onMouseEnter={() => handleOptionMouseEnter(index)}
              onClick={() => activate(option)}
            >
              {option.label}
            </button>
          ))}
          {afterHeight > 0 && (
            <div aria-hidden style={{ height: afterHeight }} />
          )}
        </div>
      )}
    </div>
  );
}

describe("useSearchablePopupInteraction", () => {
  it("keeps focus on the combobox, blocks disabled activation, and restores the trigger", async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();
    render(<ListboxHarness mode="single-select" onActivate={onActivate} />);

    const trigger = screen.getByRole("button", { name: "Test choices" });
    await user.click(trigger);
    const search = screen.getByRole("combobox", { name: "Search choices" });
    expect(search).toHaveFocus();

    await user.keyboard("{ArrowDown}");
    expect(search).toHaveAttribute(
      "aria-activedescendant",
      screen.getByRole("option", { name: "Beta" }).id,
    );
    await user.keyboard("{Enter}");
    expect(onActivate).not.toHaveBeenCalled();

    await user.keyboard("{ArrowDown}");
    expect(search).toHaveAttribute(
      "aria-activedescendant",
      screen.getByRole("option", { name: "Gamma" }).id,
    );
    await user.keyboard("{Enter}");
    expect(onActivate).toHaveBeenCalledWith(testOptions[2]);
    expect(screen.queryByRole("listbox", { name: "Test options" }))
      .not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("resets search and dismisses on an outside pointer", async () => {
    const user = userEvent.setup();
    render(<ListboxHarness mode="single-select" onActivate={() => {}} />);

    const trigger = screen.getByRole("button", { name: "Test choices" });
    await user.click(trigger);
    await user.type(
      screen.getByRole("combobox", { name: "Search choices" }),
      "gam",
    );
    expect(screen.getAllByRole("option")).toHaveLength(1);

    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole("listbox", { name: "Test options" }))
      .not.toBeInTheDocument();

    await user.click(trigger);
    expect(screen.getByRole("combobox", { name: "Search choices" }))
      .toHaveValue("");
  });

  it("uses active-descendant navigation across a virtualized collection", async () => {
    const user = userEvent.setup();
    render(
      <ListboxHarness
        mode="multi-select"
        onActivate={() => {}}
        initialVisibleCount={2}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Test choices" }));
    const search = screen.getByRole("combobox", { name: "Search choices" });
    const listbox = screen.getByRole("listbox", { name: "Test options" });
    expect(within(listbox).getAllByRole("option").length)
      .toBeLessThanOrEqual(testOptions.length);

    await user.keyboard("{End}");
    const delta = within(listbox).getByRole("option", { name: "Delta" });
    expect(search).toHaveFocus();
    expect(search).toHaveAttribute("aria-activedescendant", delta.id);
  });

  it("resets active identity when matching option identity changes", async () => {
    const user = userEvent.setup();
    const rendered = render(
      <ListboxHarness mode="multi-select" onActivate={() => {}} />,
    );

    await user.click(screen.getByRole("button", { name: "Test choices" }));
    const search = screen.getByRole("combobox", { name: "Search choices" });
    await user.keyboard("{End}");

    rendered.rerender(
      <ListboxHarness
        mode="multi-select"
        onActivate={() => {}}
        options={[testOptions[2], testOptions[0]]}
      />,
    );
    expect(search).toHaveAttribute(
      "aria-activedescendant",
      screen.getByRole("option", { name: "Gamma" }).id,
    );
  });

  it("keeps dialog-search focus while navigating and activating", async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();
    const onClear = vi.fn();
    render(<SearchDialogHarness onActivate={onActivate} onClear={onClear} />);

    const search = screen.getByRole("combobox", { name: "Search records" });
    await user.type(search, "a");
    const dialog = screen.getByRole("dialog", { name: "Matching records" });
    expect(search).toHaveFocus();

    await user.keyboard("{End}");
    const delta = within(dialog).getByRole("button", { name: "Delta" });
    expect(search).toHaveAttribute("aria-activedescendant", delta.id);
    expect(search).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(onActivate).toHaveBeenCalledWith(testOptions[3]);
    expect(screen.queryByRole("dialog", { name: "Matching records" }))
      .not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(onClear).toHaveBeenCalledOnce();
    expect(search).toHaveValue("");
    expect(search).toHaveFocus();
  });
});
