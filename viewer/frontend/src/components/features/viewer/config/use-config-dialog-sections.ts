import { useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { type ConfigSection, sectionTitlesFromSignature } from "@/lib/config";

// Manages which config sections are open in the full-config dialog, keeping the
// open set in sync with the section list and supporting jump-to-section (which
// opens the target, then scrolls to it).
export function useConfigDialogSections(sections: ConfigSection[], autoOpenKey?: string) {
  const sectionTitles = useMemo(() => sections.map((section) => section.title), [sections]);
  // NUL separator pairs with sectionTitlesFromSignature (which splits on it) and
  // cannot collide with a section title.
  const sectionTitleSignature = sectionTitles.join(String.fromCharCode(0));
  const [openSectionTitles, setOpenSectionTitles] = useState(() => new Set(sectionTitles));
  const areAllSectionsOpen =
    sectionTitles.length > 0 && sectionTitles.every((title) => openSectionTitles.has(title));
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    setOpenSectionTitles(new Set(sectionTitlesFromSignature(sectionTitleSignature)));
  }, [sectionTitleSignature, autoOpenKey]);

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
    sectionRefs.current[title]?.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  return {
    openSectionTitles,
    areAllSectionsOpen,
    sectionRefs,
    toggleSection,
    toggleAllSections,
    jumpToSection,
  };
}
