import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import postcss from "postcss";
import tailwindcss from "tailwindcss";
import { type Config } from "tailwindcss";
import { describe, expect, it } from "vitest";
import tailwindConfig, {
  workbenchCssVariables,
  workbenchTailwindColorAliases,
} from "../../tailwind.config";
import { buildEmperorTheme } from "@/lib/echarts/theme";
import { workbenchGraphEdgeVisual } from "@/lib/graph/visuals";
import {
  workbenchBorderTokens,
  workbenchControlTokens,
  workbenchElevationTokens,
  workbenchFocusTokens,
  workbenchFontTokens,
  workbenchGradientTokens,
  workbenchMotionTokens,
  workbenchOpacityTokens,
  workbenchRadiusTokens,
  workbenchSpacingTokens,
  workbenchTrackingTokens,
  workbenchTypographyTokens,
  workbenchVisualTokens,
  workbenchVisualizationTokens,
  type WorkbenchVisualTokenName,
} from "@/lib/visual-tokens";

const MINIMUM_NORMAL_TEXT_CONTRAST = 4.5;
const MINIMUM_FOCUS_CONTRAST = 3;
const primarySurfaceNames = [
  "bg",
  "bg2",
  "panel",
  "panel2",
  "cardA",
  "cardB",
] as const satisfies readonly WorkbenchVisualTokenName[];

const themeColors = tailwindConfig.theme?.extend?.colors as unknown as Record<
  string,
  (options: { opacityValue?: string }) => string
>;
const themeSpacing = tailwindConfig.theme?.extend?.spacing as Record<
  string,
  string
>;
const themeRadii = tailwindConfig.theme?.extend?.borderRadius as Record<
  string,
  string
>;
const themeFontSizes = tailwindConfig.theme?.extend?.fontSize as Record<
  string,
  [string, { lineHeight: string; letterSpacing?: string }]
>;
const themeShadows = tailwindConfig.theme?.extend?.boxShadow as Record<
  string,
  string
>;
const themeTracking = tailwindConfig.theme?.extend?.letterSpacing as Record<
  string,
  string
>;
const themeOpacity = tailwindConfig.theme?.extend?.opacity as Record<
  string,
  string
>;
const themeDurations = tailwindConfig.theme?.extend
  ?.transitionDuration as Record<string, string>;
const themeEasing = tailwindConfig.theme?.extend
  ?.transitionTimingFunction as Record<string, string>;
const themeAnimations = tailwindConfig.theme?.extend?.animation as Record<
  string,
  string
>;
const globalsCss = readFileSync(
  fileURLToPath(new URL("../../app/globals.css", import.meta.url)),
  "utf8",
);

type AuditSource = { path: string; source: string };

function productionComponentSources(directory: URL): AuditSource[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const resource = new URL(`${entry.name}${entry.isDirectory() ? "/" : ""}`, directory);
    if (entry.isDirectory()) {
      return productionComponentSources(resource);
    }
    if (!entry.name.endsWith(".tsx") || entry.name.includes(".test.")) {
      return [];
    }
    return [{ path: fileURLToPath(resource), source: readFileSync(resource, "utf8") }];
  });
}

function auditSource(resource: URL): AuditSource {
  return {
    path: fileURLToPath(resource),
    source: readFileSync(resource, "utf8"),
  };
}

const productionAuditSources = [
  ...productionComponentSources(new URL("../../app/", import.meta.url)),
  ...productionComponentSources(new URL("../components/", import.meta.url)),
  ...productionComponentSources(
    new URL("../features/workbench/components/", import.meta.url),
  ),
  auditSource(new URL("../../app/globals.css", import.meta.url)),
  auditSource(new URL("../../tailwind.config.ts", import.meta.url)),
  auditSource(new URL("./visual-tokens.ts", import.meta.url)),
  auditSource(new URL("./charts.ts", import.meta.url)),
  auditSource(new URL("./echarts/theme.ts", import.meta.url)),
  auditSource(new URL("./graph/visuals.ts", import.meta.url)),
  auditSource(
    new URL(
      "../features/workbench/state/graph-monitor/use-monitor-chart-queries.ts",
      import.meta.url,
    ),
  ),
  auditSource(
    new URL(
      "../features/workbench/state/logs/_logs-chart-state.ts",
      import.meta.url,
    ),
  ),
];

function kebabCase(value: string) {
  return value.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
}

function expectFlatVariables(prefix: string, tokens: Readonly<Record<string, string>>) {
  for (const [name, value] of Object.entries(tokens)) {
    expect(workbenchCssVariables[`--${prefix}-${kebabCase(name)}`]).toBe(value);
  }
}

function colorTuple(color: string): [number, number, number, number] {
  const hex = color.match(/^#([\da-f]{6})$/i)?.[1];
  if (hex) {
    return [
      Number.parseInt(hex.slice(0, 2), 16),
      Number.parseInt(hex.slice(2, 4), 16),
      Number.parseInt(hex.slice(4, 6), 16),
      1,
    ];
  }
  const rgba = color.match(
    /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)$/i,
  );
  if (!rgba) {
    throw new Error(`Unsupported test color ${color}`);
  }
  return [
    Number(rgba[1]),
    Number(rgba[2]),
    Number(rgba[3]),
    Number(rgba[4] ?? 1),
  ];
}

function composite(foreground: string, background: string) {
  const [red, green, blue, alpha] = colorTuple(foreground);
  const [backRed, backGreen, backBlue] = colorTuple(background);
  const channel = (front: number, back: number) =>
    Math.round(front * alpha + back * (1 - alpha));
  return [channel(red, backRed), channel(green, backGreen), channel(blue, backBlue)];
}

function relativeLuminance(channels: readonly number[]) {
  const [red, green, blue] = channels.map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(firstColor: string, secondColor: string) {
  const first = colorTuple(firstColor).slice(0, 3);
  const second = colorTuple(secondColor).slice(0, 3);
  const firstLuminance = relativeLuminance(first);
  const secondLuminance = relativeLuminance(second);
  const lighter = Math.max(firstLuminance, secondLuminance);
  const darker = Math.min(firstLuminance, secondLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

describe("Workbench design-system adapters", () => {
  it("projects every typed token into CSS custom properties", () => {
    for (const [name, value] of Object.entries(workbenchVisualTokens)) {
      const cssName = kebabCase(name);
      expect(workbenchCssVariables[`--color-${cssName}`]).toBe(value);
      expect(workbenchCssVariables[`--color-${cssName}-rgb`]).toMatch(
        /^\d+ \d+ \d+$/,
      );
      expect(Number(workbenchCssVariables[`--color-${cssName}-alpha`])).toBeGreaterThan(
        0,
      );
    }
    expectFlatVariables("space", workbenchSpacingTokens);
    expectFlatVariables("font-fallback", workbenchFontTokens);
    expectFlatVariables("tracking", workbenchTrackingTokens);
    expectFlatVariables("control", workbenchControlTokens);
    expectFlatVariables("radius", workbenchRadiusTokens);
    expectFlatVariables("border", workbenchBorderTokens);
    expectFlatVariables("elevation", workbenchElevationTokens);
    expectFlatVariables("focus", workbenchFocusTokens);
    expectFlatVariables("gradient", workbenchGradientTokens);
    expectFlatVariables("opacity", workbenchOpacityTokens);
    expectFlatVariables("motion", workbenchMotionTokens);

    for (const [name, role] of Object.entries(workbenchTypographyTokens)) {
      expect(workbenchCssVariables[`--type-${name}-size`]).toBe(role.fontSize);
      expect(workbenchCssVariables[`--type-${name}-leading`]).toBe(
        role.lineHeight,
      );
      expect(workbenchCssVariables[`--type-${name}-tracking`]).toBe(
        role.letterSpacing,
      );
    }
  });

  it("maps Tailwind colors to RGB variables with opacity modifiers", () => {
    for (const [alias, token] of Object.entries(workbenchTailwindColorAliases)) {
      const adapter = themeColors[alias];
      const cssName = kebabCase(token);
      expect(adapter({})).toBe(
        `rgb(var(--color-${cssName}-rgb) / var(--color-${cssName}-alpha))`,
      );
      expect(adapter({ opacityValue: "0.37" })).toBe(
        `rgb(var(--color-${cssName}-rgb) / calc(var(--color-${cssName}-alpha) * 0.37))`,
      );
    }
  });

  it("multiplies explicit Tailwind opacity by a semantic color's intrinsic alpha", () => {
    const adapter = themeColors["accent-soft"];
    expect(workbenchVisualTokens.accentSoft).toBe("rgba(169,154,255,0.12)");
    expect(adapter({ opacityValue: "0.5" })).toBe(
      "rgb(var(--color-accent-soft-rgb) / calc(var(--color-accent-soft-alpha) * 0.5))",
    );
    expect(
      Number(workbenchCssVariables["--color-accent-soft-alpha"]) * 0.5,
    ).toBeCloseTo(0.06);
  });

  it("generates utilities that preserve intrinsic alpha composition", async () => {
    const result = await postcss([
      tailwindcss({
        ...tailwindConfig,
        content: [
          {
            raw: '<div class="bg-accent-soft/50 border-accent-line/[0.25]"></div>',
            extension: "html",
          },
        ],
      } as Config),
    ]).process("@tailwind utilities;", { from: undefined });

    expect(result.css).toContain(
      "background-color: rgb(var(--color-accent-soft-rgb) / calc(var(--color-accent-soft-alpha) * var(--opacity-disabled)))",
    );
    expect(result.css).toContain(
      "border-color: rgb(var(--color-accent-line-rgb) / calc(var(--color-accent-line-alpha) * 0.25))",
    );
  });

  it("maps spacing, controls, radii, typography, and elevation to variables", () => {
    expect(themeSpacing).toMatchObject({
      "control-sm": "var(--control-compact)",
      control: "var(--control-default)",
      "control-lg": "var(--control-comfortable)",
      touch: "var(--control-touch)",
      panel: "var(--space-panel)",
      region: "var(--space-region)",
      shell: "var(--space-shell)",
      "shell-wide": "var(--space-shell-wide)",
    });
    expect(themeRadii).toMatchObject({
      chip: "var(--radius-chip)",
      control: "var(--radius-control)",
      panel: "var(--radius-panel)",
      card: "var(--radius-card)",
      dialog: "var(--radius-dialog)",
    });
    for (const name of Object.keys(workbenchTypographyTokens)) {
      expect(themeFontSizes[name]?.[0]).toBe(`var(--type-${name}-size)`);
      expect(themeFontSizes[name]?.[1].lineHeight).toBe(
        `var(--type-${name}-leading)`,
      );
    }
    expect(themeFontSizes).toMatchObject({
      xs: ["var(--type-label-size)", expect.any(Object)],
      sm: ["var(--type-body-size)", expect.any(Object)],
      base: ["var(--type-title-size)", expect.any(Object)],
    });
    for (const name of Object.keys(workbenchTrackingTokens)) {
      expect(themeTracking[name]).toBe(`var(--tracking-${name})`);
    }
    for (const name of Object.keys(workbenchElevationTokens)) {
      const alias = kebabCase(name);
      expect(themeShadows[alias]).toBe(`var(--elevation-${alias})`);
    }
    for (const value of [
      ...Object.values(workbenchSpacingTokens),
      ...Object.values(workbenchControlTokens),
    ]) {
      expect((Number.parseFloat(value) * 16) % 4, value).toBe(0);
    }
    expect(workbenchControlTokens.touch).toBe("2.75rem");
    expect(themeOpacity).toMatchObject({
      0: "var(--opacity-hidden)",
      50: "var(--opacity-disabled)",
      82: "var(--opacity-resting)",
      100: "var(--opacity-visible)",
    });
    expect(themeDurations).toMatchObject({
      100: "var(--motion-duration-fast)",
      150: "var(--motion-duration)",
    });
    expect(themeEasing.out).toBe("var(--motion-ease-out)");
    expect(themeAnimations).toMatchObject({
      spin: expect.stringContaining("var(--motion-duration-spin)"),
      pulse: expect.stringContaining("var(--motion-duration-pulse)"),
    });
  });

  it("derives ECharts, graph, and visualization palettes from the registry", () => {
    const theme = buildEmperorTheme("fixture-mono");
    const categoryAxis = theme.categoryAxis as {
      axisLabel: { color: string };
      axisLine: { lineStyle: { color: string } };
      splitLine: { lineStyle: { color: string } };
    };
    const tooltip = theme.tooltip as {
      backgroundColor: string;
      borderColor: string;
      axisPointer: { label: { backgroundColor: string; color: string } };
    };
    expect(theme.color).toBe(workbenchVisualizationTokens.scalarSeriesColors);
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
    expect(workbenchGraphEdgeVisual()).toMatchObject({
      markerEnd: { color: workbenchVisualTokens.gradientMiddle },
      style: { stroke: workbenchVisualTokens.violetDeep },
    });
  });

  it("keeps semantic text and focus states above WCAG contrast thresholds", () => {
    for (const surfaceName of primarySurfaceNames) {
      const surface = workbenchVisualTokens[surfaceName];
      for (const textName of ["ink", "inkDim", "inkFaint"] as const) {
        expect(
          contrastRatio(workbenchVisualTokens[textName], surface),
          `${textName} on ${surfaceName}`,
        ).toBeGreaterThanOrEqual(MINIMUM_NORMAL_TEXT_CONTRAST);
      }
    }
    for (const textName of [
      "violet",
      "violetText",
      "cyan",
      "ok",
      "amber",
      "danger",
      "dangerText",
    ] as const) {
      expect(
        contrastRatio(workbenchVisualTokens[textName], workbenchVisualTokens.panel),
        `${textName} on panel`,
      ).toBeGreaterThanOrEqual(MINIMUM_NORMAL_TEXT_CONTRAST);
    }
    const compositedFocus = composite(
      workbenchVisualTokens.focus,
      workbenchVisualTokens.bg,
    );
    const focusLuminance = relativeLuminance(compositedFocus);
    const backgroundLuminance = relativeLuminance(
      colorTuple(workbenchVisualTokens.bg).slice(0, 3),
    );
    expect(
      (Math.max(focusLuminance, backgroundLuminance) + 0.05) /
        (Math.min(focusLuminance, backgroundLuminance) + 0.05),
    ).toBeGreaterThanOrEqual(MINIMUM_FOCUS_CONTRAST);

    expect(
      contrastRatio(
        workbenchVisualTokens.white,
        workbenchVisualTokens.selectedControl,
      ),
      "white on selected-control",
    ).toBeGreaterThanOrEqual(MINIMUM_NORMAL_TEXT_CONTRAST);
    const selectedControlHover = composite(
      "rgba(112,88,220,0.9)",
      workbenchVisualTokens.panel2,
    );
    expect(
      contrastRatio(
        workbenchVisualTokens.white,
        `rgb(${selectedControlHover.join(",")})`,
      ),
      "white on selected-control/90 over panel-2",
    ).toBeGreaterThanOrEqual(MINIMUM_NORMAL_TEXT_CONTRAST);
  });

  it("uses the selected-control surface for white-text buttons and view controls", () => {
    const selectedControlSources = productionAuditSources.filter(({ source }) =>
      source.includes("bg-selected-control"),
    );
    expect(selectedControlSources.map(({ path }) => path)).toEqual(
      expect.arrayContaining([
        expect.stringContaining("/components/ui/button.tsx"),
        expect.stringContaining("/components/view-mode-button.tsx"),
        expect.stringContaining("/components/logs/logs-chart-panel.tsx"),
      ]),
    );
    for (const { path, source } of selectedControlSources) {
      expect(source, path).toMatch(/bg-selected-control(?:\/90)?[^"\n]*text-white|bg-selected-control text-white/);
    }
    for (const { path, source } of productionAuditSources) {
      expect(source, path).not.toMatch(
        /bg-(?:violet|violet-deep)(?=\s)[^"\n]*text-white/,
      );
    }
  });

  it("audits all 125 production component and visual-support files", () => {
    expect(new Set(productionAuditSources.map(({ path }) => path)).size).toBe(125);
  });

  it("keeps raw UI colors, arbitrary shadows, and arbitrary type out of production", () => {
    const tokenOwner = fileURLToPath(new URL("./visual-tokens.ts", import.meta.url));
    for (const { path, source } of productionAuditSources) {
      if (path !== tokenOwner) {
        expect(source, path).not.toMatch(
          /#[\da-f]{3,8}\b|\brgba?\(\s*\d|\bhsla?\(\s*\d/i,
        );
      }
      expect(source, path).not.toMatch(/shadow-\[[^\]]+\]/);
      expect(source, path).not.toMatch(/text-\[[^\]]+\]/);
      expect(source, path).not.toMatch(/(?:tracking|leading|font)-\[[^\]]+\]/);
      expect(source, path).not.toMatch(
        /\b(?:bg|text|border|ring|from|via|to)-\[(?:#|rgba?|hsla?|linear-gradient|radial-gradient)[^\]]+\]/,
      );
      expect(source, path).not.toMatch(/\btransition-all\b|transition\s*:\s*all\b/i);
      expect(source, path).not.toMatch(
        /<(?:div|span)\b[^>]*\bonClick\s*=/,
      );
      expect(source, path).not.toMatch(/\bautoFocus\b/);
      expect(source, path).not.toMatch(/user-scalable\s*=\s*no|maximum-scale\s*=\s*1/i);
      expect(source, path).not.toMatch(
        /md:(?:h|min-h|min-w|w)-(?:5|6|7|\[30px\])/,
      );
    }
  });

  it("marks every Lucide icon as decorative or explicitly meaningful", () => {
    for (const { path, source } of productionAuditSources.filter(({ path }) =>
      path.endsWith(".tsx"),
    )) {
      const imports = Array.from(
        source.matchAll(
          /import\s*\{([^}]*)\}\s*from\s*["']lucide-react["']/g,
        ),
      ).flatMap((match) =>
        match[1]
          .split(",")
          .map((name) => name.trim().split(/\s+as\s+/).at(-1))
          .filter((name): name is string => Boolean(name)),
      );
      for (const name of imports) {
        const iconPattern = new RegExp(`<${name}\\b([\\s\\S]*?)(?:\\/>|>)`, "g");
        for (const icon of source.matchAll(iconPattern)) {
          expect(icon[0], `${path} <${name}>`).toMatch(
            /aria-hidden|aria-label|role=|title=/,
          );
        }
      }
    }
  });

  it("publishes reduced motion, safe areas, dark controls, and responsive targets", () => {
    expect(globalsCss).toContain("@media (prefers-reduced-motion: reduce)");
    expect(globalsCss).toContain("scroll-behavior: auto !important");
    expect(globalsCss).toContain(".safe-dialog-inset");
    expect(globalsCss).toContain(".dialog-shell-panel");
    expect(globalsCss).toContain("@media (max-width: 639px)");
    expect(globalsCss).toContain("border-radius: 0");
    expect(globalsCss).toContain(".safe-header-inset");
    for (const edge of ["top", "right", "bottom", "left"]) {
      expect(globalsCss).toContain(`env(safe-area-inset-${edge})`);
    }
    expect(globalsCss).toContain("color-scheme: dark");
    expect(globalsCss).toContain("touch-action: manipulation");
    expect(globalsCss).toContain("width: var(--control-touch)");
    expect(globalsCss).toContain("overflow: hidden");
  });
});
