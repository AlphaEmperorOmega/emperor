import { useState } from "react";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  AddConfigSnapshotDialog,
  ConfigSnapshotsTray,
} from "@/features/workbench/components/config/config-snapshots-tray";
import {
  type ConfigSnapshotMutationOutcome,
  type ConfigSnapshotMutationStatus,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";
import type { ConfigField } from "@/lib/api/models";
import {
  type ConfigSnapshot,
  type ConfigSnapshotCreateResult,
} from "@/lib/config-snapshots";

const field: ConfigField = {
  key: "hidden_size",
  configKey: "HIDDEN_SIZE",
  flag: "--hidden-size",
  label: "Hidden size",
  section: "Model",
  sectionPath: ["Model"],
  type: "int",
  default: 64,
  nullable: false,
  choices: [],
  locked: false,
};

const snapshot: ConfigSnapshot = {
  id: "snapshot-1",
  name: "Snapshot one",
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  overrides: { hidden_size: "128" },
  createdAt: "2026-06-01T00:00:00.000Z",
};

const idleMutation: ConfigSnapshotMutationStatus = {
  phase: "idle",
  kind: null,
  snapshotId: null,
  error: "",
  canRetry: false,
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function mutationStatus(
  phase: ConfigSnapshotMutationStatus["phase"],
  kind: ConfigSnapshotMutationStatus["kind"],
  error = "",
): ConfigSnapshotMutationStatus {
  return {
    phase,
    kind,
    snapshotId: kind === "create" ? null : snapshot.id,
    error,
    canRetry: phase === "failed",
  };
}

describe("Config Snapshot mutation rendering", () => {
  it("keeps the save dialog open through rejection and authoritative retry", async () => {
    const firstAttempt = deferred<ConfigSnapshotCreateResult>();
    const retryAttempt = deferred<ConfigSnapshotCreateResult>();
    const closed = vi.fn();

    function Harness() {
      const [open, setOpen] = useState(true);
      const [mutation, setMutation] = useState(idleMutation);
      if (!open) {
        return null;
      }
      return (
        <AddConfigSnapshotDialog
          modelType="linears"
          model="linear"
          preset="baseline"
          fields={[field]}
          overrides={{ hidden_size: "128" }}
          snapshots={[]}
          initialName="Tuned"
          mutation={mutation}
          onAdd={async () => {
            setMutation(mutationStatus("pending", "create"));
            const result = await firstAttempt.promise;
            setMutation(
              result.ok
                ? mutationStatus("succeeded", "create")
                : mutationStatus("failed", "create", result.error),
            );
            return result;
          }}
          onRetry={async () => {
            setMutation(mutationStatus("pending", "create"));
            const result = await retryAttempt.promise;
            setMutation(
              result.ok
                ? mutationStatus("succeeded", "create")
                : mutationStatus("failed", "create", result.error),
            );
            return result;
          }}
          onDismissMutation={() => setMutation(idleMutation)}
          onClose={() => {
            closed();
            setOpen(false);
          }}
        />
      );
    }

    render(<Harness />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Add Snapshot" }));

    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Close add config snapshot" }),
    ).toBeDisabled();
    expect(closed).not.toHaveBeenCalled();

    await act(async () => {
      firstAttempt.resolve({ ok: false, error: "Persistence rejected." });
      await firstAttempt.promise;
    });
    expect(screen.getByRole("alert")).toHaveTextContent("Persistence rejected.");
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry Save" }));
    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
    expect(closed).not.toHaveBeenCalled();

    await act(async () => {
      retryAttempt.resolve({ ok: true, snapshot });
      await retryAttempt.promise;
    });
    expect(closed).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("keeps rename editing active until retry succeeds", async () => {
    const firstAttempt = deferred<ConfigSnapshotMutationOutcome>();
    const retryAttempt = deferred<ConfigSnapshotMutationOutcome>();

    function Harness() {
      const [mutation, setMutation] = useState(idleMutation);
      return (
        <ConfigSnapshotsTray
          groups={[{ preset: "baseline", snapshots: [snapshot] }]}
          selectedPreset="baseline"
          selectedTrainingSnapshotIds={[]}
          overrides={snapshot.overrides}
          canManage
          mutation={mutation}
          onLoad={vi.fn()}
          onRename={async () => {
            setMutation(mutationStatus("pending", "rename"));
            const result = await firstAttempt.promise;
            setMutation(
              result.ok
                ? mutationStatus("succeeded", "rename")
                : mutationStatus("failed", "rename", result.error),
            );
            return result;
          }}
          onRemove={vi.fn()}
          onRetryMutation={async () => {
            setMutation(mutationStatus("pending", "rename"));
            const result = await retryAttempt.promise;
            setMutation(
              result.ok
                ? mutationStatus("succeeded", "rename")
                : mutationStatus("failed", "rename", result.error),
            );
            return result;
          }}
          onDismissMutation={() => setMutation(idleMutation)}
          onToggleSelection={vi.fn()}
        />
      );
    }

    render(<Harness />);
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Rename snapshot Snapshot one" }),
    );
    await user.click(screen.getByRole("button", { name: "Save snapshot name" }));

    expect(screen.getByLabelText("Snapshot name")).toBeDisabled();
    expect(screen.getByText("Renaming Config Snapshot…")).toBeInTheDocument();

    await act(async () => {
      firstAttempt.resolve({
        ok: false,
        kind: "rename",
        snapshotId: snapshot.id,
        error: "Rename rejected.",
        retryable: true,
      });
      await firstAttempt.promise;
    });
    expect(screen.getByLabelText("Snapshot name")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Rename rejected.");

    await user.click(screen.getByRole("button", { name: "Retry Change" }));
    await act(async () => {
      retryAttempt.resolve({
        ok: true,
        kind: "rename",
        snapshotId: snapshot.id,
        record: null,
      });
      await retryAttempt.promise;
    });
    expect(screen.queryByLabelText("Snapshot name")).not.toBeInTheDocument();
  });

  it("keeps a snapshot visible until removal retry is authoritative", async () => {
    const firstAttempt = deferred<ConfigSnapshotMutationOutcome>();
    const retryAttempt = deferred<ConfigSnapshotMutationOutcome>();

    function Harness() {
      const [records, setRecords] = useState([snapshot]);
      const [mutation, setMutation] = useState(idleMutation);
      return (
        <ConfigSnapshotsTray
          groups={records.length > 0 ? [{ preset: "baseline", snapshots: records }] : []}
          selectedPreset="baseline"
          selectedTrainingSnapshotIds={[]}
          overrides={{}}
          canManage
          mutation={mutation}
          onLoad={vi.fn()}
          onRename={vi.fn()}
          onRemove={async () => {
            setMutation(mutationStatus("pending", "remove"));
            const result = await firstAttempt.promise;
            setMutation(
              result.ok
                ? mutationStatus("succeeded", "remove")
                : mutationStatus("failed", "remove", result.error),
            );
            if (result.ok) {
              setRecords([]);
            }
            return result;
          }}
          onRetryMutation={async () => {
            setMutation(mutationStatus("pending", "remove"));
            const result = await retryAttempt.promise;
            setMutation(
              result.ok
                ? mutationStatus("succeeded", "remove")
                : mutationStatus("failed", "remove", result.error),
            );
            if (result.ok) {
              setRecords([]);
            }
            return result;
          }}
          onDismissMutation={() => setMutation(idleMutation)}
          onToggleSelection={vi.fn()}
        />
      );
    }

    render(<Harness />);
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Remove snapshot Snapshot one" }),
    );
    expect(screen.getByText("Snapshot one")).toBeInTheDocument();

    await act(async () => {
      firstAttempt.resolve({
        ok: false,
        kind: "remove",
        snapshotId: snapshot.id,
        error: "Removal rejected.",
        retryable: true,
      });
      await firstAttempt.promise;
    });
    expect(screen.getByText("Snapshot one")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Removal rejected.");

    await user.click(screen.getByRole("button", { name: "Retry Change" }));
    await act(async () => {
      retryAttempt.resolve({
        ok: true,
        kind: "remove",
        snapshotId: snapshot.id,
        record: null,
      });
      await retryAttempt.promise;
    });
    expect(screen.queryByText("Snapshot one")).not.toBeInTheDocument();
  });
});
