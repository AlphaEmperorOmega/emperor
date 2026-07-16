import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  isWorkbenchProtectedAccessReady: vi.fn(),
  useConfigSnapshotEditor: vi.fn(),
  useConfigSnapshotEditorState: vi.fn(),
  useConfigSnapshotRecords: vi.fn(),
  useModelPackageInspection: vi.fn(),
  useTrainingConfiguration: vi.fn(),
  useWorkbenchCapabilities: vi.fn(),
  useWorkbenchConnection: vi.fn(),
}));

vi.mock("@/features/workbench/providers/workbench-providers", () => ({
  useConfigSnapshotEditor: mocks.useConfigSnapshotEditor,
  useConfigSnapshotRecords: mocks.useConfigSnapshotRecords,
  useModelPackageInspection: mocks.useModelPackageInspection,
}));

vi.mock(
  "@/features/workbench/providers/workbench-connection-provider",
  () => ({
    isWorkbenchProtectedAccessReady: mocks.isWorkbenchProtectedAccessReady,
    useWorkbenchCapabilities: mocks.useWorkbenchCapabilities,
    useWorkbenchConnection: mocks.useWorkbenchConnection,
  }),
);

vi.mock("@/features/workbench/providers/training-execution-context", () => ({
  useTrainingConfiguration: mocks.useTrainingConfiguration,
}));

vi.mock(
  "@/features/workbench/state/config-snapshots/use-config-snapshot-editor",
  () => ({
    useConfigSnapshotEditorState: mocks.useConfigSnapshotEditorState,
  }),
);

import type { ConfigField } from "@/lib/api/models";
import { type ConfigSection } from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { useFullConfigSession } from "@/features/workbench/state/full-config/use-full-config-session";

const hiddenDim: ConfigField = {
  key: "hidden_dim",
  configKey: "HIDDEN_DIM",
  flag: "--hidden-dim",
  label: "Hidden Dim",
  section: "Model",
  sectionPath: ["Model"],
  type: "int",
  default: 64,
  nullable: false,
  choices: [],
  locked: false,
};

const modelSections: ConfigSection[] = [
  { title: "Model", fields: [hiddenDim] },
];
const trainingSections: ConfigSection[] = [
  { title: "Training", fields: [{ ...hiddenDim, section: "Training" }] },
];
const editorSections: ConfigSection[] = [
  { title: "Snapshot", fields: [{ ...hiddenDim, section: "Snapshot" }] },
];

const snapshot: ConfigSnapshot = {
  id: "snapshot-1",
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  name: "Wide",
  overrides: { hidden_dim: "128" },
  createdAt: "2026-06-01T00:00:00.000Z",
};

const idleMutation = {
  phase: "idle" as const,
  kind: null,
  snapshotId: null,
  error: "",
  canRetry: false,
};

function installAdapters({
  selectedEditorSnapshot = snapshot,
  editorCloseResult = true,
  launcherSession = null,
}: {
  selectedEditorSnapshot?: ConfigSnapshot | null;
  editorCloseResult?: boolean;
  launcherSession?: object | null;
} = {}) {
  const modelActions = {
    editRuntimeDefault: vi.fn(),
    clearRuntimeDefault: vi.fn(),
    resetRuntimeDefaults: vi.fn(),
    refreshInspection: vi.fn(),
  };
  const libraryActions = {
    selectTarget: vi.fn(),
    rename: vi.fn(),
    remove: vi.fn(),
    retryMutation: vi.fn(),
    dismissMutation: vi.fn(),
  };
  const launcherActions = {
    beginDraft: vi.fn(() => true),
    beginEdit: vi.fn(),
    beginDuplicate: vi.fn(),
    close: vi.fn(),
  };
  const editorActions = {
    updateOverride: vi.fn(),
    clearOverride: vi.fn(),
    reset: vi.fn(),
    load: vi.fn(),
    rename: vi.fn(),
    remove: vi.fn(),
    retryMutation: vi.fn(),
    dismissMutation: vi.fn(),
    save: vi.fn(async () => ({ ok: true, snapshot })),
    retrySave: vi.fn(async () => ({ ok: true, snapshot })),
    close: vi.fn(() => editorCloseResult),
  };
  const trainingActions = {
    updateOverride: vi.fn(),
    clearOverride: vi.fn(),
    resetOverrides: vi.fn(),
    includeSnapshot: vi.fn(),
    excludeSnapshot: vi.fn(),
  };

  mocks.useWorkbenchCapabilities.mockReturnValue({
    capabilities: { configSnapshotsEnabled: true },
  });
  mocks.useWorkbenchConnection.mockReturnValue({});
  mocks.isWorkbenchProtectedAccessReady.mockReturnValue(true);
  mocks.useModelPackageInspection.mockReturnValue({
    browser: {
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "baseline",
      selectedDatasets: ["Mnist"],
    },
    options: { configSections: modelSections },
    runtimeDefaults: {
      fieldCount: 1,
      active: { hidden_dim: "96" },
      inactiveLockedCount: 1,
    },
    status: { schema: { isLoading: false } },
    actions: modelActions,
  });
  mocks.useConfigSnapshotRecords.mockReturnValue({
    records: {
      all: [snapshot],
      allGroups: [{ preset: "baseline", snapshots: [snapshot] }],
    },
    mutation: idleMutation,
    actions: libraryActions,
  });
  mocks.useConfigSnapshotEditor.mockReturnValue({
    session: launcherSession,
    actions: launcherActions,
  });
  mocks.useConfigSnapshotEditorState.mockReturnValue({
    session: {
      modelType: "experts",
      model: "expert",
      preset: "wide",
      selectedSnapshot: selectedEditorSnapshot ?? undefined,
      records: [snapshot],
      recordGroups: [{ preset: "baseline", snapshots: [snapshot] }],
      configSections: editorSections,
      fieldCount: 1,
      draft: { hidden_dim: "160" },
      status: { isLoading: true, mutation: idleMutation },
    },
    actions: editorActions,
  });
  mocks.useTrainingConfiguration.mockReturnValue({
    selectedModelType: "experts",
    selectedModel: "mixture",
    selectedPrimaryPreset: "fast",
    selectedSnapshotIds: ["snapshot-1"],
    selectedMonitors: ["loss"],
    configSections: trainingSections,
    fieldCount: 1,
    bulkOverrides: { hidden_dim: "192" },
    inactiveLockedOverrideCount: 2,
    schemaLoading: false,
    updateOverride: trainingActions.updateOverride,
    clearOverride: trainingActions.clearOverride,
    resetOverrides: trainingActions.resetOverrides,
    includeSnapshot: trainingActions.includeSnapshot,
    excludeSnapshot: trainingActions.excludeSnapshot,
  });

  return {
    editorActions,
    launcherActions,
    libraryActions,
    modelActions,
    trainingActions,
  };
}

beforeEach(() => {
  for (const mock of Object.values(mocks)) {
    mock.mockReset();
  }
});

describe("useFullConfigSession", () => {
  it("adapts Model Runtime Defaults and opens a transient snapshot save", async () => {
    const adapters = installAdapters();
    const onClose = vi.fn();
    const { result } = renderHook(() =>
      useFullConfigSession({ mode: "default", scope: "model", onClose }),
    );

    expect(result.current).toMatchObject({
      kind: "model",
      identity: {
        modelType: "linears",
        model: "linear",
        preset: "baseline",
      },
      runtimeDefaults: {
        sections: modelSections,
        overrides: { hidden_dim: "96" },
        lockedOverrideCount: 1,
        isLoading: false,
      },
      controls: {
        canReset: true,
        canAddSnapshot: true,
        canUpdatePreview: true,
      },
    });
    act(() => result.current.actions.editOverride("hidden_dim", "128"));
    expect(adapters.modelActions.editRuntimeDefault).toHaveBeenCalledWith(
      "hidden_dim",
      "128",
    );
    act(() => result.current.actions.resetOverrides());
    expect(adapters.modelActions.resetRuntimeDefaults).toHaveBeenCalledTimes(1);
    act(() => result.current.actions.loadSnapshot("snapshot-1"));
    expect(adapters.libraryActions.selectTarget).toHaveBeenCalledWith(
      "snapshot-1",
    );

    expect(result.current.actions.openSnapshotSave()).toBe(true);
    expect(adapters.launcherActions.beginDraft).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      overrides: { hidden_dim: "96" },
    });
    await act(async () => {
      await result.current.actions.saveSnapshot("Model snapshot");
    });
    expect(adapters.editorActions.save).toHaveBeenCalledWith("Model snapshot");
    expect(adapters.editorActions.close).toHaveBeenCalledTimes(1);
    expect(result.current.trainingCommand).toContain(
      "--model-type linears --model linear --preset baseline",
    );
  });

  it("adapts Training Runtime Defaults without exposing Model commands", () => {
    const adapters = installAdapters();
    const onClose = vi.fn();
    const { result } = renderHook(() =>
      useFullConfigSession({ mode: "default", scope: "training", onClose }),
    );

    expect(result.current).toMatchObject({
      kind: "training",
      identity: {
        modelType: "experts",
        model: "mixture",
        preset: "fast",
      },
      runtimeDefaults: {
        sections: trainingSections,
        overrides: { hidden_dim: "192" },
        lockedOverrideCount: 2,
      },
      controls: {
        canAddSnapshot: false,
        canSaveSnapshot: false,
        canUpdatePreview: false,
        showTrainingCommand: false,
        showSnapshotLibrary: false,
      },
    });
    act(() => result.current.actions.editOverride("hidden_dim", "224"));
    expect(adapters.trainingActions.updateOverride).toHaveBeenCalledWith(
      "hidden_dim",
      "224",
    );
    act(() => result.current.actions.resetOverrides());
    expect(adapters.trainingActions.resetOverrides).toHaveBeenCalledTimes(1);
    act(() => result.current.actions.close());
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(adapters.editorActions.close).not.toHaveBeenCalled();
  });

  it("lets Config Snapshot draft mode override Training scope and routes editor actions", async () => {
    const adapters = installAdapters();
    const onClose = vi.fn();
    const { result } = renderHook(() =>
      useFullConfigSession({
        mode: "snapshotDraft",
        scope: "training",
        onClose,
      }),
    );

    expect(result.current).toMatchObject({
      kind: "snapshot-draft",
      identity: {
        modelType: "experts",
        model: "expert",
        preset: "wide",
      },
      runtimeDefaults: {
        sections: editorSections,
        overrides: { hidden_dim: "160" },
        lockedOverrideCount: 0,
        isLoading: true,
      },
      controls: { canSaveSnapshot: true },
    });
    act(() => result.current.actions.clearOverride("hidden_dim"));
    expect(adapters.editorActions.clearOverride).toHaveBeenCalledWith(
      "hidden_dim",
    );
    act(() => result.current.actions.renameSnapshot("snapshot-1", "Renamed"));
    expect(adapters.editorActions.rename).toHaveBeenCalledWith(
      "snapshot-1",
      "Renamed",
    );
    await act(async () => {
      await result.current.actions.saveSnapshot("Draft snapshot");
    });
    expect(adapters.editorActions.save).toHaveBeenCalledWith("Draft snapshot");
    expect(adapters.editorActions.close).not.toHaveBeenCalled();
    act(() => result.current.actions.close());
    expect(adapters.editorActions.close).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("blocks Snapshot edit close while pending ownership refuses it", () => {
    const adapters = installAdapters({ editorCloseResult: false });
    const onClose = vi.fn();
    const { result } = renderHook(() =>
      useFullConfigSession({ mode: "snapshotEdit", scope: "model", onClose }),
    );

    expect(result.current.kind).toBe("snapshot-edit");
    expect(result.current.controls.canSaveSnapshot).toBe(true);
    act(() => result.current.actions.removeSnapshot("snapshot-1"));
    expect(adapters.editorActions.remove).toHaveBeenCalledWith("snapshot-1");
    act(() => result.current.actions.close());
    expect(adapters.editorActions.close).toHaveBeenCalledTimes(1);
    expect(onClose).not.toHaveBeenCalled();
  });

  it("disables Snapshot edit save when its authoritative record disappeared", () => {
    installAdapters({ selectedEditorSnapshot: null });
    const { result } = renderHook(() =>
      useFullConfigSession({
        mode: "snapshotEdit",
        scope: "model",
        onClose: vi.fn(),
      }),
    );

    expect(result.current.controls.canSaveSnapshot).toBe(false);
  });
});
