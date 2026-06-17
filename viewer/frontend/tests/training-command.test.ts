import { describe, expect, it } from "vitest";
import { buildTrainingCommand } from "@/lib/training-command";
import { type ConfigField } from "@/lib/api";
import { type ConfigSection } from "@/lib/config";

function field(overrides: Partial<ConfigField> & Pick<ConfigField, "key" | "flag">): ConfigField {
  return {
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    label: overrides.label ?? overrides.key,
    section: overrides.section ?? "General",
    type: overrides.type ?? "string",
    default: overrides.default ?? "",
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    ...overrides,
  };
}

const sections: ConfigSection[] = [
  {
    title: "Layer",
    fields: [
      field({ key: "hidden_dim", flag: "--hidden-dim", type: "int", default: 256 }),
      field({ key: "activation", flag: "--activation", default: "GELU" }),
    ],
  },
  {
    title: "Trainer",
    fields: [
      field({
        key: "dropout_schedule",
        flag: "--dropout-schedule",
        default: null,
        nullable: true,
      }),
      field({ key: "run_name", flag: "--run-name" }),
    ],
  },
];

describe("buildTrainingCommand", () => {
  it("omits --config when no overrides are set", () => {
    expect(
      buildTrainingCommand({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        sections,
        overrides: {},
      }),
    ).toBe(
      "source experiment.sh --model-type linears --model linear --preset baseline",
    );
  });

  it("uses --presets when multiple run presets are selected", () => {
    expect(
      buildTrainingCommand({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        presets: ["baseline", "gating"],
        sections,
        overrides: {},
      }),
    ).toBe(
      "source experiment.sh --model-type linears --model linear --presets baseline gating",
    );
  });

  it("emits overrides in config schema order", () => {
    expect(
      buildTrainingCommand({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        sections,
        overrides: {
          activation: "RELU",
          hidden_dim: "128",
        },
      }),
    ).toBe(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --hidden-dim 128 --activation RELU",
    );
  });

  it("serializes nullable empty overrides as None", () => {
    expect(
      buildTrainingCommand({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        sections,
        overrides: {
          dropout_schedule: "",
        },
      }),
    ).toBe(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --dropout-schedule None",
    );
  });

  it("quotes values with spaces and shell-sensitive characters", () => {
    expect(
      buildTrainingCommand({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        sections,
        overrides: {
          run_name: "Bob's config; rm -rf /",
        },
      }),
    ).toBe(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --run-name 'Bob'\"'\"'s config; rm -rf /'",
    );
  });
});
