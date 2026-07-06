import { describe, expect, it } from "vitest";

import { type ConfigField } from "@/lib/api";
import { type ConfigSection } from "@/lib/config";
import { buildTrainingCommand } from "@/lib/training-command";

function field(key: string): ConfigField {
  return {
    key,
    configKey: key.toUpperCase(),
    flag: `--${key.replace(/_/g, "-")}`,
    label: key,
    section: "Router Options",
    type: "bool",
    default: false,
    nullable: false,
    choices: [true, false],
    locked: false,
  };
}

describe("training command generation", () => {
  it("includes router controller override flags", () => {
    const sections: ConfigSection[] = [
      {
        title: "Router Gate Options",
        fields: [field("router_gate_flag")],
      },
      {
        title: "Router Memory Options",
        fields: [field("router_memory_flag")],
      },
      {
        title: "Router Recurrent Layer Options",
        fields: [field("router_recurrent_flag")],
      },
    ];

    const command = buildTrainingCommand({
      modelType: "experts",
      model: "linear_adaptive",
      preset: "baseline",
      sections,
      overrides: {
        router_gate_flag: "true",
        router_memory_flag: "true",
        router_recurrent_flag: "true",
      },
    });

    expect(command).toContain("--config");
    expect(command).toContain("--router-gate-flag true");
    expect(command).toContain("--router-memory-flag true");
    expect(command).toContain("--router-recurrent-flag true");
  });
});
