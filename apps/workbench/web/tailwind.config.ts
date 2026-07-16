import type { Config } from "tailwindcss";
import plugin from "tailwindcss/plugin";
import {
  workbenchCssVariables,
  workbenchElevationTokens,
  workbenchTailwindColor,
  workbenchTrackingTokens,
  workbenchTypographyTokens,
  type WorkbenchVisualTokenName,
} from "./src/lib/visual-tokens";

export { workbenchCssVariables } from "./src/lib/visual-tokens";

export const workbenchTailwindColorAliases = {
  white: "white",
  black: "black",
  bg: "bg",
  "bg-2": "bg2",
  panel: "panel",
  "panel-2": "panel2",
  "card-a": "cardA",
  "card-b": "cardB",
  ink: "ink",
  "ink-dim": "inkDim",
  "ink-faint": "inkFaint",
  violet: "violet",
  "violet-deep": "violetDeep",
  "violet-text": "violetText",
  "violet-muted": "violetMuted",
  "selected-control": "selectedControl",
  blue: "blue",
  cyan: "cyan",
  ok: "ok",
  amber: "amber",
  line: "line",
  "line-soft": "lineSoft",
  "line-hover": "lineHover",
  "modified-field": "modifiedField",
  border: "line",
  surface: "bg2",
  control: "control",
  "control-muted": "controlMuted",
  "control-subtle": "controlSubtle",
  "control-active": "controlActive",
  "control-hover": "controlHover",
  "control-field": "controlField",
  "control-track": "controlTrack",
  muted: "inkDim",
  accent: "violet",
  amberline: "amber",
  danger: "danger",
  "danger-text": "dangerText",
  "danger-detail": "dangerDetail",
  "danger-hover": "dangerHover",
  focus: "focus",
  subtle: "lineSoft",
  faint: "faint",
  "accent-soft": "accentSoft",
  "accent-line": "accentLine",
  "accent-edge": "accentEdge",
  "danger-soft": "dangerSoft",
  "danger-line": "dangerLine",
  "structure-overlay": "structureOverlay",
} as const satisfies Record<string, WorkbenchVisualTokenName>;

const colors = Object.fromEntries(
  Object.entries(workbenchTailwindColorAliases).map(([name, token]) => [
    name,
    workbenchTailwindColor(token),
  ]),
) as unknown as Record<string, string>;

function typographyRole(
  name: keyof typeof workbenchTypographyTokens,
): [
  string,
  { lineHeight: string; letterSpacing: string },
] {
  return [
    `var(--type-${name}-size)`,
    {
      lineHeight: `var(--type-${name}-leading)`,
      letterSpacing: `var(--type-${name}-tracking)`,
    },
  ];
}

const config: Config = {
  // Tests import production components, so test-only class strings are not
  // retained in the shipped stylesheet.
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors,
      borderColor: {
        line: colors.line,
        "line-soft": colors["line-soft"],
      },
      borderRadius: {
        indicator: "var(--radius-indicator)",
        chip: "var(--radius-chip)",
        "control-sm": "var(--radius-control-sm)",
        "control-md": "var(--radius-control-md)",
        control: "var(--radius-control)",
        ctl: "var(--radius-control)",
        panel: "var(--radius-panel)",
        "control-group": "var(--radius-panel)",
        card: "var(--radius-card)",
        dialog: "var(--radius-dialog)",
        full: "var(--radius-round)",
      },
      spacing: {
        "control-sm": "var(--control-compact)",
        control: "var(--control-default)",
        "control-lg": "var(--control-comfortable)",
        touch: "var(--control-touch)",
        panel: "var(--space-panel)",
        region: "var(--space-region)",
        shell: "var(--space-shell)",
        "shell-wide": "var(--space-shell-wide)",
      },
      fontSize: {
        xs: typographyRole("label"),
        sm: typographyRole("body"),
        base: typographyRole("title"),
        micro: typographyRole("micro"),
        caption: typographyRole("caption"),
        meta: typographyRole("meta"),
        label: typographyRole("label"),
        compact: typographyRole("compact"),
        body: typographyRole("body"),
        title: typographyRole("title"),
        heading: typographyRole("heading"),
        display: typographyRole("display"),
      },
      letterSpacing: Object.fromEntries(
        Object.keys(workbenchTrackingTokens).map((name) => [
          name,
          `var(--tracking-${name})`,
        ]),
      ),
      fontFamily: {
        sans: ["var(--font-sans)", "var(--font-fallback-sans)"],
        mono: ["var(--font-mono)", "var(--font-fallback-mono)"],
      },
      backgroundImage: {
        grad: "var(--gradient-primary)",
        "config-preset": "var(--gradient-config-preset)",
        "config-preset-header": "var(--gradient-config-preset-header)",
        "config-preset-header-hover":
          "var(--gradient-config-preset-header-hover)",
        "config-navigation": "var(--gradient-config-navigation)",
        "config-navigation-hover": "var(--gradient-config-navigation-hover)",
        "cluster-active": "var(--gradient-cluster-active)",
        minimap: "var(--gradient-minimap)",
        "cluster-panel": "var(--gradient-cluster-panel)",
        "component-info": "var(--gradient-component-info)",
      },
      boxShadow: Object.fromEntries(
        Object.keys(workbenchElevationTokens).map((name) => [
          name.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase(),
          `var(--elevation-${name
            .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
            .toLowerCase()})`,
        ]),
      ),
      opacity: {
        0: "var(--opacity-hidden)",
        30: "var(--opacity-faint)",
        35: "var(--opacity-low)",
        45: "var(--opacity-restrained)",
        50: "var(--opacity-disabled)",
        60: "var(--opacity-soft)",
        65: "var(--opacity-muted)",
        70: "var(--opacity-quiet)",
        75: "var(--opacity-subdued)",
        82: "var(--opacity-resting)",
        90: "var(--opacity-strong)",
        100: "var(--opacity-visible)",
      },
      transitionDuration: {
        100: "var(--motion-duration-fast)",
        150: "var(--motion-duration)",
        fast: "var(--motion-duration-fast)",
        DEFAULT: "var(--motion-duration)",
        slow: "var(--motion-duration-slow)",
      },
      transitionTimingFunction: {
        out: "var(--motion-ease-out)",
        precision: "var(--motion-ease-out)",
      },
      animation: {
        spin: "spin var(--motion-duration-spin) linear infinite",
        pulse:
          "pulse var(--motion-duration-pulse) var(--motion-ease-standard) infinite",
      },
    },
  },
  plugins: [
    plugin(({ addBase }) => {
      addBase({ ":root": workbenchCssVariables });
    }),
  ],
};

export default config;
