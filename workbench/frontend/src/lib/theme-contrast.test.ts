import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import tailwindConfig, { workbenchCssVariables } from "../../tailwind.config";
import { buildEmperorTheme } from "@/lib/echarts/theme";
import { workbenchGraphEdgeVisual } from "@/lib/graph/visuals";
import {
  workbenchVisualTokens,
  type WorkbenchVisualTokenName,
} from "@/lib/visual-tokens";

const MINIMUM_NORMAL_TEXT_CONTRAST = 4.5;
const primarySurfaceNames = [
  "bg",
  "bg2",
  "panel",
  "panel2",
  "cardA",
  "cardB",
] as const satisfies readonly WorkbenchVisualTokenName[];

const cssTokenNames = {
  "--amber": "amber",
  "--bg": "bg",
  "--bg-2": "bg2",
  "--blue": "blue",
  "--card-a": "cardA",
  "--card-b": "cardB",
  "--cyan": "cyan",
  "--graph-grid": "graphGrid",
  "--ink": "ink",
  "--ink-dim": "inkDim",
  "--ink-faint": "inkFaint",
  "--line": "line",
  "--line-soft": "lineSoft",
  "--ok": "ok",
  "--panel": "panel",
  "--panel-2": "panel2",
  "--violet": "violet",
  "--violet-deep": "violetDeep",
} as const satisfies Record<string, WorkbenchVisualTokenName>;

const themeColors = tailwindConfig.theme?.extend?.colors as Record<
  string,
  string
>;
const globalsCss = readFileSync(
  fileURLToPath(new URL("../../app/globals.css", import.meta.url)),
  "utf8",
);

function relativeLuminance(hexColor: string) {
  const channels = hexColor
    .slice(1)
    .match(/.{2}/g)
    ?.map((channel) => Number.parseInt(channel, 16) / 255);
  if (!channels || channels.length !== 3) {
    throw new Error(`Expected a six-digit hex color, received ${hexColor}`);
  }

  const [red, green, blue] = channels.map((channel) =>
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
  );
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(firstColor: string, secondColor: string) {
  const firstLuminance = relativeLuminance(firstColor);
  const secondLuminance = relativeLuminance(secondColor);
  const lighter = Math.max(firstLuminance, secondLuminance);
  const darker = Math.min(firstLuminance, secondLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

describe("Workbench visual-token adapters", () => {
  it("generates CSS custom properties from the token owner", () => {
    for (const [cssName, tokenName] of Object.entries(cssTokenNames)) {
      expect(workbenchCssVariables[cssName as keyof typeof workbenchCssVariables]).toBe(
        workbenchVisualTokens[tokenName],
      );
      expect(globalsCss).not.toMatch(new RegExp(`${cssName}:\\s*(?:#|rgba?\\()`, "i"));
    }
    expect(workbenchCssVariables["--grad"]).toBe(
      `linear-gradient(135deg, ${workbenchVisualTokens.gradientStart} 0%, ${workbenchVisualTokens.gradientMiddle} 48%, ${workbenchVisualTokens.gradientEnd} 100%)`,
    );
    expect(globalsCss).toContain("radial-gradient(var(--graph-grid)");
  });

  it("derives every named Tailwind color from a semantic token", () => {
    const tokenValues = new Set<string>(Object.values(workbenchVisualTokens));
    for (const [name, value] of Object.entries(themeColors)) {
      expect(tokenValues.has(value), `${name} ${value}`).toBe(true);
    }
    expect(themeColors).toMatchObject({
      accent: workbenchVisualTokens.violet,
      amberline: workbenchVisualTokens.amber,
      border: workbenchVisualTokens.line,
      muted: workbenchVisualTokens.inkDim,
      subtle: workbenchVisualTokens.lineSoft,
      surface: workbenchVisualTokens.bg2,
    });
  });

  it("translates the same semantic facts into ECharts and graph canvas styles", () => {
    const theme = buildEmperorTheme("fixture-mono");
    const categoryAxis = theme.categoryAxis as {
      axisLabel: { color: string };
      axisLine: { lineStyle: { color: string } };
      splitLine: { lineStyle: { color: string } };
    };
    const tooltip = theme.tooltip as {
      backgroundColor: string;
      borderColor: string;
      axisPointer: {
        label: { backgroundColor: string; color: string };
      };
    };
    const dataZoom = theme.dataZoom as {
      fillerColor: string;
      moveHandleStyle: { color: string };
    };

    expect(categoryAxis.axisLabel.color).toBe(workbenchVisualTokens.inkFaint);
    expect(categoryAxis.axisLine.lineStyle.color).toBe(workbenchVisualTokens.line);
    expect(categoryAxis.splitLine.lineStyle.color).toBe(
      workbenchVisualTokens.lineSoft,
    );
    expect(tooltip).toMatchObject({
      backgroundColor: workbenchVisualTokens.panel,
      borderColor: workbenchVisualTokens.line,
      axisPointer: {
        label: {
          backgroundColor: workbenchVisualTokens.violet,
          color: workbenchVisualTokens.bg,
        },
      },
    });
    expect(dataZoom).toMatchObject({
      fillerColor: workbenchVisualTokens.accentFill,
      moveHandleStyle: { color: workbenchVisualTokens.inkFaint },
    });
    expect(workbenchGraphEdgeVisual()).toMatchObject({
      markerEnd: { color: workbenchVisualTokens.gradientMiddle },
      style: { stroke: workbenchVisualTokens.violetDeep },
    });
  });

  it("keeps normal faint text at WCAG AA contrast on every primary surface", () => {
    for (const surfaceName of primarySurfaceNames) {
      const surface = workbenchVisualTokens[surfaceName];
      expect(
        contrastRatio(workbenchVisualTokens.inkFaint, surface),
        `inkFaint ${workbenchVisualTokens.inkFaint} on ${surfaceName} ${surface}`,
      ).toBeGreaterThanOrEqual(MINIMUM_NORMAL_TEXT_CONTRAST);
    }
  });
});
