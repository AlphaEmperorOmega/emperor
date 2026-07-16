import { act, fireEvent, render, screen } from "@testing-library/react";
import { useEffect, useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WorkbenchWorkspaceActivities } from "@/features/workbench/components/workbench-workspaces";
import { type WorkbenchWorkspace } from "@/types/workbench";

function StatefulWorkspace({
  label,
  onCleanup,
  onSetup,
}: {
  label: string;
  onCleanup: () => void;
  onSetup: () => void;
}) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    onSetup();
    return onCleanup;
  }, [onCleanup, onSetup]);

  return (
    <button type="button" onClick={() => setCount((current) => current + 1)}>
      {label}: {count}
    </button>
  );
}

function WorkspaceHarness({
  activeWorkspace,
  onLogsCleanup,
  onLogsSetup,
}: {
  activeWorkspace: WorkbenchWorkspace;
  onLogsCleanup: () => void;
  onLogsSetup: () => void;
}) {
  return (
    <WorkbenchWorkspaceActivities
      activeWorkspace={activeWorkspace}
      deferredWorkspaceOrder={["logs", "training"]}
      model={<div>Model workspace</div>}
      logs={
        <StatefulWorkspace
          label="Logs state"
          onCleanup={onLogsCleanup}
          onSetup={onLogsSetup}
        />
      }
      training={<div>Training workspace</div>}
    />
  );
}

afterEach(() => vi.useRealTimers());

describe("Workbench workspace Activity boundaries", () => {
  it("retains local UI state while hidden and reconnects effects when shown", () => {
    const onLogsCleanup = vi.fn();
    const onLogsSetup = vi.fn();
    const rendered = render(
      <WorkspaceHarness
        activeWorkspace="logs"
        onLogsCleanup={onLogsCleanup}
        onLogsSetup={onLogsSetup}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Logs state: 0" }));
    expect(screen.getByRole("button", { name: "Logs state: 1" })).toBeVisible();
    expect(onLogsSetup).toHaveBeenCalledTimes(1);

    rendered.rerender(
      <WorkspaceHarness
        activeWorkspace="model"
        onLogsCleanup={onLogsCleanup}
        onLogsSetup={onLogsSetup}
      />,
    );
    expect(onLogsCleanup).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Logs state: 1")).not.toBeVisible();

    rendered.rerender(
      <WorkspaceHarness
        activeWorkspace="logs"
        onLogsCleanup={onLogsCleanup}
        onLogsSetup={onLogsSetup}
      />,
    );
    expect(screen.getByRole("button", { name: "Logs state: 1" })).toBeVisible();
    expect(onLogsSetup).toHaveBeenCalledTimes(2);
  });

  it("keeps app-scoped polling active while Training UI effects are hidden", () => {
    vi.useFakeTimers();
    const onPoll = vi.fn();
    const onTrainingCleanup = vi.fn();
    const onTrainingSetup = vi.fn();

    function AppScopedTraining({
      activeWorkspace,
    }: {
      activeWorkspace: WorkbenchWorkspace;
    }) {
      useEffect(() => {
        const timer = setInterval(onPoll, 50);
        return () => clearInterval(timer);
      }, []);
      return (
        <WorkbenchWorkspaceActivities
          activeWorkspace={activeWorkspace}
          deferredWorkspaceOrder={["training"]}
          model={<div>Model workspace</div>}
          logs={<div>Logs workspace</div>}
          training={
            <StatefulWorkspace
              label="Training state"
              onCleanup={onTrainingCleanup}
              onSetup={onTrainingSetup}
            />
          }
        />
      );
    }

    const rendered = render(<AppScopedTraining activeWorkspace="training" />);
    act(() => vi.advanceTimersByTime(100));
    expect(onPoll).toHaveBeenCalledTimes(2);
    expect(onTrainingSetup).toHaveBeenCalledTimes(1);

    rendered.rerender(<AppScopedTraining activeWorkspace="model" />);
    expect(onTrainingCleanup).toHaveBeenCalledTimes(1);
    act(() => vi.advanceTimersByTime(100));
    expect(onPoll).toHaveBeenCalledTimes(4);
  });
});
