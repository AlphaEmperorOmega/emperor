import { type ReactNode } from "react";
import {
  useViewerState,
  type GraphViewContextValue,
  type HistoricalRunsContextValue,
  type TargetConfigContextValue,
  type TrainingContextValue,
} from "@/components/features/viewer/use-viewer-state";
import { createViewerContext } from "@/components/features/viewer/providers/create-context";

const [TargetConfigProvider, useTargetConfig] =
  createViewerContext<TargetConfigContextValue>("TargetConfigContext");
const [GraphViewProvider, useGraphView] =
  createViewerContext<GraphViewContextValue>("GraphViewContext");
const [HistoricalRunsProvider, useHistoricalRuns] =
  createViewerContext<HistoricalRunsContextValue>("HistoricalRunsContext");
const [TrainingProvider, useTraining] =
  createViewerContext<TrainingContextValue>("TrainingContext");

export { useTargetConfig, useGraphView, useHistoricalRuns, useTraining };

export type ViewerProvidersProps = {
  /** Wired to the logs workspace so a new job's folder appears in its run list. */
  onJobStarted?: (logFolder: string) => void;
  children: ReactNode;
};

/**
 * Runs the viewer orchestration engine once and distributes its four domain
 * slices through nested contexts, so panels read exactly the slice they need
 * instead of receiving it drilled down through props.
 */
export function ViewerProviders({ onJobStarted, children }: ViewerProvidersProps) {
  const { target, graph, history, training } = useViewerState({ onJobStarted });
  return (
    <TargetConfigProvider value={target}>
      <GraphViewProvider value={graph}>
        <HistoricalRunsProvider value={history}>
          <TrainingProvider value={training}>{children}</TrainingProvider>
        </HistoricalRunsProvider>
      </GraphViewProvider>
    </TargetConfigProvider>
  );
}
