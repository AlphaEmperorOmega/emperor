import { useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { sectionTitlesFromSignature } from "@/lib/config";

// Manages which config sections are open in the full-config dialog, keeping the
// open set in sync with the section list and supporting jump-to-section (which
// opens the target, then scrolls to it).
export function useConfigDialogSections(
  sections: Array<{ title: string }>,
  autoOpenKey?: string,
  initialOpenTitles?: string[],
) {
  const sectionTitles = useMemo(() => sections.map((section) => section.title), [sections]);
  const initialOpenTitleSignature = useMemo(
    () => (initialOpenTitles ?? sectionTitles).join(String.fromCharCode(0)),
    [initialOpenTitles, sectionTitles],
  );
  const [openSectionTitles, setOpenSectionTitles] = useState(
    () => new Set(sectionTitlesFromSignature(initialOpenTitleSignature)),
  );
  const previousAutoOpenKey = useRef(autoOpenKey);
  const areAllSectionsOpen =
    sectionTitles.length > 0 && sectionTitles.every((title) => openSectionTitles.has(title));
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    const initialOpen = sectionTitlesFromSignature(initialOpenTitleSignature).filter(
      (title) => sectionTitles.includes(title),
    );
    const sectionTitleSet = new Set(sectionTitles);
    const shouldResetOpenSections = previousAutoOpenKey.current !== autoOpenKey;
    previousAutoOpenKey.current = autoOpenKey;

    setOpenSectionTitles((current) => {
      if (shouldResetOpenSections) {
        return new Set(initialOpen);
      }
      const next = new Set<string>();
      for (const title of current) {
        if (sectionTitleSet.has(title)) {
          next.add(title);
        }
      }
      for (const title of initialOpen) {
        next.add(title);
      }
      return next;
    });
  }, [sectionTitles, initialOpenTitleSignature, autoOpenKey]);

  function toggleSection(title: string) {
    setOpenSectionTitles((current) => {
      const next = new Set(current);
      if (next.has(title)) {
        next.delete(title);
      } else {
        next.add(title);
      }
      return next;
    });
  }

  function toggleAllSections() {
    setOpenSectionTitles(areAllSectionsOpen ? new Set<string>() : new Set(sectionTitles));
  }

  function setOpenSections(titles: string[]) {
    setOpenSectionTitles(new Set(titles));
  }

  function jumpToSection(title: string) {
    flushSync(() => {
      setOpenSectionTitles((current) => {
        if (current.has(title)) {
          return current;
        }
        const next = new Set(current);
        next.add(title);
        return next;
      });
    });
    const reduceMotion = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    sectionRefs.current[title]?.scrollIntoView({
      block: "start",
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }

  return {
    openSectionTitles,
    areAllSectionsOpen,
    sectionRefs,
    toggleSection,
    toggleAllSections,
    setOpenSections,
    jumpToSection,
  };
}
