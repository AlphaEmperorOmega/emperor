import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  useConfigSchemaQuery: vi.fn(),
  useConfigSnapshotRecords: vi.fn(),
  create: vi.fn(),
  rename: vi.fn(),
  update: vi.fn(),
  remove: vi.fn(),
}));

vi.mock("@/features/workbench/state/use-workbench-queries", () => ({
  useConfigSchemaQuery: mocks.useConfigSchemaQuery,
}));

vi.mock(
  "@/features/workbench/state/config-snapshots/use-config-snapshot-records",
  () => ({
    useConfigSnapshotRecords: mocks.useConfigSnapshotRecords,
  }),
);

import { type ConfigField } from "@/lib/api";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import {
  useConfigSnapshotEditorState,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-editor";

let fields: ConfigField[] = [];
let records: ConfigSnapshot[] = [];

function field(
  overrides: Partial<ConfigField> & Pick<ConfigField, "key">,
): ConfigField {
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replaceAll("_", "-")}`,
    label: overrides.label ?? overrides.key,
    section: overrides.section ?? "Model",
    sectionPath: overrides.sectionPath ?? [overrides.section ?? "Model"],
    type: overrides.type ?? "int",
    default: "default" in overrides ? overrides.default ?? null : 64,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

function snapshot(overrides: Partial<ConfigSnapshot> = {}): ConfigSnapshot {
  return {
    id: overrides.id ?? "snapshot-1",
    name: overrides.name ?? "Snapshot one",
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    overrides: overrides.overrides ?? {},
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
  };
}

beforeEach(() => {
  fields = [field({ key: "hidden_size" })];
  records = [];
  mocks.create.mockReset();
  mocks.rename.mockReset();
  mocks.update.mockReset();
  mocks.remove.mockReset();
  mocks.useConfigSchemaQuery.mockReset().mockImplementation(
    (modelType: string, model: string, preset: string) => ({
      data: { modelType, model, preset, fields },
      isLoading: false,
      isSuccess: Boolean(modelType && model && preset),
      isError: false,
      error: null,
    }),
  );
  mocks.useConfigSnapshotRecords.mockReset().mockImplementation(() => ({
    records,
    status: {
      isLoading: false,
      isReady: true,
      isError: false,
      error: null,
    },
    actions: {
      create: mocks.create,
      rename: mocks.rename,
      update: mocks.update,
      remove: mocks.remove,
    },
  }));
});

describe("Config Snapshot editor session", () => {
  it("opens an explicit cross-model draft without changing an Inspection target", async () => {
    const { result } = renderHook(() => useConfigSnapshotEditorState());

    act(() => {
      expect(
        result.current.actions.beginDraft({
          modelType: "experts",
          model: "linear",
          preset: "expert-baseline",
        }),
      ).toBe(true);
      result.current.actions.updateOverride("hidden_size", "128");
    });

    await waitFor(() => {
      expect(result.current.session).toMatchObject({
        modelType: "experts",
        model: "linear",
        preset: "expert-baseline",
        draft: { hidden_size: "128" },
      });
    });
    expect(mocks.useConfigSchemaQuery).toHaveBeenLastCalledWith(
      "experts",
      "linear",
      "expert-baseline",
      { enabled: true },
    );

    act(() => {
      expect(result.current.actions.save("Expert tuned").ok).toBe(true);
    });
    expect(mocks.create).toHaveBeenCalledWith({
      modelType: "experts",
      model: "linear",
      preset: "expert-baseline",
      name: "Expert tuned",
      overrides: { hidden_size: "128" },
    });
  });

  it("normalizes and updates an existing snapshot in its isolated edit session", async () => {
    fields = [
      field({
        key: "weight_option_flag",
        type: "bool",
        default: false,
        choices: [true, false],
      }),
      field({
        key: "weight_option",
        type: "class",
        default: null,
        nullable: true,
        choices: ["SingleModelDynamicWeightConfig"],
      }),
    ];
    const record = snapshot({
      id: "adaptive",
      overrides: { weight_option_flag: "true" },
    });
    records = [record];
    const { result } = renderHook(() => useConfigSnapshotEditorState());

    act(() => {
      result.current.actions.beginEdit(record);
    });
    await waitFor(() => {
      expect(result.current.session.draft).toEqual({
        weight_option_flag: "true",
        weight_option: "SingleModelDynamicWeightConfig",
      });
    });

    act(() => {
      expect(result.current.actions.save("Adaptive edited").ok).toBe(true);
    });
    expect(mocks.update).toHaveBeenCalledWith({
      id: "adaptive",
      input: {
        name: "Adaptive edited",
        overrides: {
          weight_option_flag: "true",
          weight_option: "SingleModelDynamicWeightConfig",
        },
      },
    });
    expect(mocks.create).not.toHaveBeenCalled();
  });

  it("duplicates a snapshot through create while retaining the source values", async () => {
    const record = snapshot({
      id: "source",
      name: "Source",
      overrides: { hidden_size: "256" },
    });
    records = [record];
    const { result } = renderHook(() => useConfigSnapshotEditorState());

    act(() => {
      result.current.actions.beginDuplicate(record);
    });
    await waitFor(() => {
      expect(result.current.session.draft).toEqual({ hidden_size: "256" });
    });
    act(() => {
      result.current.actions.updateOverride("hidden_size", "512");
    });
    act(() => {
      expect(result.current.actions.save("Source copy").ok).toBe(true);
    });

    expect(mocks.create).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Source copy",
        overrides: { hidden_size: "512" },
      }),
    );
    expect(mocks.update).not.toHaveBeenCalled();
  });
});
