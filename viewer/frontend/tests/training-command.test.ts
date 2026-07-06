import { describe, expect, it } from "vitest";
import { buildTrainingCommand } from "@/lib/training-command";
import { type ConfigField } from "@/lib/api";
import { type ConfigSection } from "@/lib/config";

function field(overrides: Partial<ConfigField> & Pick<ConfigField, "key" | "flag">): ConfigField {
  return {
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    label: overrides.label ?? overrides.key,
    section: overrides.section ?? "General",
    description: overrides.description ?? "",
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
      field({ key: "stack_hidden_dim", flag: "--stack-hidden-dim", type: "int", default: 256 }),
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

  it("emits datasets, log folder, and monitors before config overrides", () => {
    expect(
      buildTrainingCommand({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        datasets: ["Mnist"],
        logFolder: "monitor_run",
        monitors: ["linear", "halting"],
        sections,
        overrides: {
          stack_hidden_dim: "128",
        },
      }),
    ).toBe(
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist --logdir monitor_run --monitors linear halting --config --stack-hidden-dim 128",
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
          stack_hidden_dim: "128",
        },
      }),
    ).toBe(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --stack-hidden-dim 128 --activation RELU",
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

  it("emits router stack overrides without an independent flag", () => {
    const routerSections: ConfigSection[] = [
      {
        title: "Router Stack Options",
        fields: [
          field({
            key: "router_stack_hidden_dim",
            flag: "--router-stack-hidden-dim",
            type: "int",
            default: 32,
          }),
          field({
            key: "router_bias_flag",
            flag: "--router-bias-flag",
            type: "bool",
            default: true,
            choices: [true, false],
          }),
        ],
      },
    ];

    const command = buildTrainingCommand({
      modelType: "experts",
      model: "linear",
      preset: "baseline",
      sections: routerSections,
      overrides: {
        router_stack_hidden_dim: "64",
        router_bias_flag: "false",
      },
    });

    expect(command).toBe(
      "source experiment.sh --model-type experts --model linear --preset baseline --config --router-stack-hidden-dim 64 --router-bias-flag false",
    );
    expect(command).not.toContain("--router-stack-independent-flag");
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
