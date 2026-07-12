import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useConfigDialogSections } from "@/features/workbench/components/config/use-config-dialog-sections";

const originalMatchMedia = window.matchMedia;
const sections = [{ title: "Optimization" }];

afterEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: originalMatchMedia,
  });
});

function installMotionPreference(reduced: boolean) {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn().mockReturnValue({ matches: reduced }),
  });
}

describe("useConfigDialogSections", () => {
  it.each([
    { reduced: false, behavior: "smooth" as const },
    { reduced: true, behavior: "auto" as const },
  ])("uses $behavior scrolling when reduced motion is $reduced", ({
    reduced,
    behavior,
  }) => {
    installMotionPreference(reduced);
    const scrollIntoView = vi.fn();
    const section = document.createElement("section");
    section.scrollIntoView = scrollIntoView;
    const { result } = renderHook(() => useConfigDialogSections(sections));

    act(() => {
      result.current.sectionRefs.current.Optimization = section;
      result.current.jumpToSection("Optimization");
    });

    expect(scrollIntoView).toHaveBeenCalledWith({
      block: "start",
      behavior,
    });
  });
});
