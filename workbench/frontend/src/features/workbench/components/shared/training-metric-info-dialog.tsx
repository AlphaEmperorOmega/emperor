import { X } from "lucide-react";
import { createPortal } from "react-dom";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";

type TrainingMetricScope = "training" | "validation" | "test" | "generic";

type TrainingMetricInfo = {
  displayName: string;
  tracks: string;
  matters: string;
  interpretation: string;
};

const metricScopeLabels: Record<TrainingMetricScope, string> = {
  training: "Training",
  validation: "Validation",
  test: "Test",
  generic: "Generic",
};

function trainingMetricTokens(metricName: string) {
  return metricName.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

function trainingMetricScope(metricName: string): TrainingMetricScope {
  const tokens = trainingMetricTokens(metricName);

  if (tokens.some((token) => ["validation", "valid", "val", "eval"].includes(token))) {
    return "validation";
  }
  if (tokens.some((token) => ["training", "train"].includes(token))) {
    return "training";
  }
  if (tokens.some((token) => ["testing", "test"].includes(token))) {
    return "test";
  }
  return "generic";
}

function readableMetricName(metricName: string) {
  return metricName.replace(/[/_.-]+/g, " ").replace(/\s+/g, " ").trim();
}

function metricInfoFor(metricName: string): TrainingMetricInfo {
  const tokens = trainingMetricTokens(metricName);
  const scope = trainingMetricScope(metricName);
  const scopeLabel = metricScopeLabels[scope];
  const scopedName = (family: string) =>
    scope === "generic" ? family : `${scopeLabel} ${family.toLowerCase()}`;
  const isAccuracy = tokens.some(
    (token) => token === "accuracy" || token === "acc",
  );
  const isLoss = tokens.some(
    (token) =>
      token === "loss" ||
      token.endsWith("loss") ||
      token === "nll" ||
      token === "crossentropy",
  );
  const isCalibration = tokens.some(
    (token) => token === "calibration" || token === "ece",
  );
  const isConfidence = tokens.includes("confidence");

  if (isAccuracy) {
    return {
      displayName: scopedName("Accuracy"),
      tracks:
        scope === "generic"
          ? "Accuracy is the percentage of examples the model got right. For example, 0.80 means about 8 out of 10 predictions were correct."
          : `${scopeLabel} accuracy is the percentage of ${scope} examples the model got right.`,
      matters:
        scope === "training"
          ? "It quickly shows whether the model is learning the examples it trains on."
          : "It quickly shows whether the model is making better predictions on data outside the current update step.",
      interpretation:
        scope === "training"
          ? "Higher is better, but compare it with validation or test accuracy. High training accuracy with low validation accuracy usually means the model is memorizing instead of generalizing."
          : "Higher is better. Compare it with training accuracy to see whether the model also works on held-out data.",
    };
  }

  if (isLoss) {
    return {
      displayName: scopedName("Loss"),
      tracks:
        scope === "generic"
          ? "Loss is a mistake score reported by the training loop. Lower means the predictions are closer to the expected answers."
          : `${scopeLabel} loss is a mistake score measured on the ${scope} data. Lower means the predictions are closer to the expected answers.`,
      matters:
        scope === "training"
          ? "This is usually the number the optimizer is directly trying to reduce while the model learns."
          : "Loss can show problems accuracy hides, such as the model being very confident about wrong answers.",
      interpretation:
        scope === "training"
          ? "Lower is generally better; if training loss falls while validation loss rises, the model may be overfitting."
          : "Lower is generally better; watch for plateaus or increases after earlier improvement.",
    };
  }

  if (isCalibration) {
    return {
      displayName: scopedName("Calibration error"),
      tracks:
        "Calibration error checks whether the model's confidence matches reality. If the model says 80% confidence, it should be right about 80% of the time.",
      matters:
        "A model can have decent accuracy and still be overconfident. Calibration helps you judge whether its probabilities are trustworthy.",
      interpretation:
        "Lower is better. Use it with accuracy: accuracy says how often the model is right, calibration says whether its confidence is honest.",
    };
  }

  if (isConfidence) {
    return {
      displayName: scopedName("Confidence"),
      tracks:
        "Confidence is how sure the model says it is about its predictions. A higher mean confidence means the model is making more certain predictions on average.",
      matters:
        "Confidence helps explain whether accuracy changes come with useful certainty or with risky overconfidence.",
      interpretation:
        "Confidence should be read with accuracy and calibration. High confidence with low accuracy means the model may be confidently wrong.",
    };
  }

  const displayName = readableMetricName(metricName) || "Custom metric";
  return {
    displayName,
    tracks:
      "A custom metric reported by the training loop for this run. Its exact definition comes from the training code that emitted it.",
    matters:
      "Custom metrics often capture task-specific behavior that generic accuracy or loss cannot fully describe.",
    interpretation:
      "Compare it against earlier steps and sibling runs, and use the metric's project definition to decide whether higher or lower is better.",
  };
}

export function TrainingMetricInfoDialog({
  metricKey,
  valueLabel,
  valueTitle = "Current value",
  onClose,
}: {
  metricKey: string;
  valueLabel: string;
  valueTitle?: string;
  onClose: () => void;
}) {
  const info = metricInfoFor(metricKey);

  if (typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <DialogShell
      titleId="training-metric-info-title"
      describedBy="training-metric-info-description"
      size="sm"
      panelVariant="surface"
      onClose={onClose}
      className="z-[90] bg-black/65 p-4 sm:p-4"
      panelClassName="grid max-h-none max-w-xl gap-4 overflow-visible p-4 sm:max-h-none sm:p-5"
      header={
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-xs font-bold uppercase tracking-label text-ink-faint">
              Metric info
            </p>
            <h2
              id="training-metric-info-title"
              className="mt-1 break-words font-mono text-base font-semibold text-ink"
            >
              {metricKey}
            </h2>
            <p
              id="training-metric-info-description"
              className="mt-1 text-sm text-ink-dim"
            >
              {info.displayName}
            </p>
          </div>
          <IconButton
            label="Close metric info"
            onClick={onClose}
            variant="edge"
            icon={<X className="h-4 w-4" aria-hidden />}
          />
        </header>
      }
      footer={
        <footer className="flex justify-end border-t border-line-soft pt-3">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </footer>
      }
    >
      <div className="grid gap-3 text-sm leading-6 text-ink-dim">
        <div className="grid gap-1 rounded-control-md border border-line-soft bg-white/[0.018] px-3 py-2.5">
          <span className="text-xs font-bold uppercase tracking-label text-ink-faint">
            {valueTitle}
          </span>
          <span className="font-mono text-base font-semibold text-ink">
            {valueLabel}
          </span>
        </div>
        <section className="grid gap-1">
          <h3 className="text-xs font-bold uppercase tracking-label text-ink-faint">
            What it tracks
          </h3>
          <p>{info.tracks}</p>
        </section>
        <section className="grid gap-1">
          <h3 className="text-xs font-bold uppercase tracking-label text-ink-faint">
            Why it matters
          </h3>
          <p>{info.matters}</p>
        </section>
        <section className="grid gap-1">
          <h3 className="text-xs font-bold uppercase tracking-label text-ink-faint">
            Interpretation
          </h3>
          <p>{info.interpretation}</p>
        </section>
      </div>
    </DialogShell>,
    document.body,
  );
}
