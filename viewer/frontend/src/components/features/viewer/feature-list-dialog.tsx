import { X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  IMPLEMENTED_FEATURES,
  type ImplementedFeature,
} from "@/lib/feature-catalog";

type FeatureGroup = {
  category: string;
  features: ImplementedFeature[];
};

const FEATURE_GROUPS = IMPLEMENTED_FEATURES.reduce<FeatureGroup[]>((groups, feature) => {
  const group = groups.find((candidate) => candidate.category === feature.category);
  if (group) {
    group.features.push(feature);
    return groups;
  }

  groups.push({ category: feature.category, features: [feature] });
  return groups;
}, []);

export function FeatureListDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="feature-list-title"
        className="edge flex max-h-[calc(100vh-1.5rem)] w-full max-w-4xl flex-col overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]"
      >
        <header className="sticky top-0 z-10 border-b border-line bg-panel/85 px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 id="feature-list-title" className="text-base font-semibold text-ink">
                Implemented Features
              </h2>
              <p className="mt-1 text-xs text-ink-faint">
                Static viewer feature catalog
              </p>
            </div>
            <div className="flex shrink-0 items-center justify-end gap-2">
              <Badge>{IMPLEMENTED_FEATURES.length} features</Badge>
              <button
                type="button"
                aria-label="Close implemented features"
                onClick={onClose}
                className="flex h-9 w-9 items-center justify-center rounded-[10px] border border-line bg-white/[0.035] text-ink-faint transition hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </div>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto bg-bg-2/80 px-4 py-4 sm:px-5">
          <div className="grid gap-5">
            {FEATURE_GROUPS.map((group, index) => {
              const categoryId = `feature-category-${index}`;
              return (
                <section key={group.category} aria-labelledby={categoryId}>
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <h3
                      id={categoryId}
                      className="text-xs font-bold uppercase tracking-[0.09em] text-ink-faint"
                    >
                      {group.category}
                    </h3>
                    <Badge>{group.features.length}</Badge>
                  </div>
                  <ul className="overflow-hidden rounded-[10px] border border-line bg-white/[0.02]">
                    {group.features.map((feature) => (
                      <li
                        key={feature.id}
                        className="grid gap-1 border-b border-line-soft px-3 py-2.5 last:border-b-0 sm:grid-cols-[minmax(180px,0.42fr)_minmax(0,1fr)] sm:gap-4"
                      >
                        <h4 className="text-sm font-semibold leading-5 text-ink">
                          {feature.title}
                        </h4>
                        <p className="text-sm leading-5 text-ink-dim">
                          {feature.description}
                        </p>
                      </li>
                    ))}
                  </ul>
                </section>
              );
            })}
          </div>
        </div>
      </section>
    </div>
  );
}
