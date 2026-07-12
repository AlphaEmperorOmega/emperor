import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { type LogRun, type LogTextSummary } from "@/lib/api";
import { type LogValidationExampleImage } from "@/features/workbench/state/logs/log-diagnostics";
import { cn } from "@/lib/utils";

export function LogValidationExamplesPanel({
  images,
  texts,
  runsById,
  enabled = true,
  isLoading = false,
  onVisible,
}: {
  images: LogValidationExampleImage[];
  texts: LogTextSummary[];
  runsById: Map<string, LogRun>;
  enabled?: boolean;
  isLoading?: boolean;
  onVisible?: () => void;
}) {
  const sectionRef = useRef<HTMLElement | null>(null);
  const [hasEnteredView, setHasEnteredView] = useState(false);

  useEffect(() => {
    if (!enabled || hasEnteredView) {
      return;
    }
    const node = sectionRef.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setHasEnteredView(true);
      onVisible?.();
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setHasEnteredView(true);
          onVisible?.();
          observer.disconnect();
        }
      },
      { rootMargin: "320px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [enabled, hasEnteredView, onVisible]);

  if (!enabled) {
    return null;
  }

  if (!hasEnteredView || isLoading) {
    return (
      <section
        ref={sectionRef}
        role={isLoading ? "status" : undefined}
        aria-label={isLoading ? "Loading validation examples…" : undefined}
        aria-busy={isLoading || undefined}
        className="grid min-h-40 place-items-center rounded-control-md border border-line-soft bg-white/[0.018] p-4"
      >
        {isLoading && (
          <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />
        )}
      </section>
    );
  }

  if (images.length === 0 && texts.length === 0) {
    return null;
  }

  return (
    <section ref={sectionRef} className="grid gap-3">
      <div className="text-xs text-ink-faint">
        Most-confident wrong predictions from the latest validation epoch
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {images.map((image) => {
          const run = runsById.get(image.runId);
          const text = image.textSummary?.text;
          const runName = run?.runName ?? image.runId;
          return (
            <div
              key={`${image.runId}:${image.tag}`}
              className="grid gap-3 rounded-control-md border border-line bg-white/[0.018] p-3"
            >
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-ink">
                  {run?.runName ?? image.runId}
                </div>
                <div className="truncate font-mono type-meta text-ink-faint">
                  step {image.step} · {image.tag}
                </div>
              </div>
              {image.truncated || !image.dataUrl ? (
                <div className="grid min-h-44 place-items-center rounded-chip border border-line-soft bg-black/20 p-4 text-center">
                  <div className="grid gap-1">
                    <div className="text-sm font-semibold text-ink">Payload omitted</div>
                    <div className="max-w-sm text-xs leading-5 text-ink-faint">
                      {image.truncationReason ??
                        "This validation image exceeded the workbench payload budget."}
                    </div>
                  </div>
                </div>
              ) : (
                <Image
                  src={image.dataUrl}
                  alt={`Most-confident wrong validation predictions for ${runName}`}
                  width={960}
                  height={540}
                  sizes="(min-width: 1280px) 50vw, 100vw"
                  unoptimized
                  className="w-full rounded-chip border border-line-soft bg-black/20 object-contain"
                />
              )}
              {text && (
                <pre
                  className={cn(
                    "max-h-36 overflow-auto rounded-chip border border-line-soft",
                    "bg-black/25 p-2 font-mono type-meta leading-5 text-ink-faint",
                  )}
                >
                  {text}
                </pre>
              )}
              {image.textSummary?.truncated && (
                <div className="rounded-chip border border-line-soft bg-white/[0.018] px-2 py-1 text-xs text-ink-faint">
                  {image.textSummary.truncationReason ??
                    "Text summary truncated for display."}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
