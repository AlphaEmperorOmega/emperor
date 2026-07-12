import type { Config } from "tailwindcss";
import plugin from "tailwindcss/plugin";
import { workbenchVisualTokens } from "./src/lib/visual-tokens";

export const workbenchCssVariables = Object.freeze({
  "--amber": workbenchVisualTokens.amber,
  "--bg": workbenchVisualTokens.bg,
  "--bg-2": workbenchVisualTokens.bg2,
  "--blue": workbenchVisualTokens.blue,
  "--card-a": workbenchVisualTokens.cardA,
  "--card-b": workbenchVisualTokens.cardB,
  "--cyan": workbenchVisualTokens.cyan,
  "--grad": `linear-gradient(135deg, ${workbenchVisualTokens.gradientStart} 0%, ${workbenchVisualTokens.gradientMiddle} 48%, ${workbenchVisualTokens.gradientEnd} 100%)`,
  "--graph-grid": workbenchVisualTokens.graphGrid,
  "--ink": workbenchVisualTokens.ink,
  "--ink-dim": workbenchVisualTokens.inkDim,
  "--ink-faint": workbenchVisualTokens.inkFaint,
  "--line": workbenchVisualTokens.line,
  "--line-soft": workbenchVisualTokens.lineSoft,
  "--ok": workbenchVisualTokens.ok,
  "--panel": workbenchVisualTokens.panel,
  "--panel-2": workbenchVisualTokens.panel2,
  "--violet": workbenchVisualTokens.violet,
  "--violet-deep": workbenchVisualTokens.violetDeep,
});

const config: Config = {
  // Production CSS is generated only from app/source files. Vitest component
  // tests import those components directly, so scanning tests would only keep
  // test-only class strings in the shipped stylesheet.
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: workbenchVisualTokens.bg,
        "bg-2": workbenchVisualTokens.bg2,
        panel: workbenchVisualTokens.panel,
        "panel-2": workbenchVisualTokens.panel2,
        "card-a": workbenchVisualTokens.cardA,
        "card-b": workbenchVisualTokens.cardB,
        ink: workbenchVisualTokens.ink,
        "ink-dim": workbenchVisualTokens.inkDim,
        "ink-faint": workbenchVisualTokens.inkFaint,
        violet: workbenchVisualTokens.violet,
        "violet-deep": workbenchVisualTokens.violetDeep,
        blue: workbenchVisualTokens.blue,
        cyan: workbenchVisualTokens.cyan,
        ok: workbenchVisualTokens.ok,
        amber: workbenchVisualTokens.amber,
        "violet-text": workbenchVisualTokens.violetText,
        "violet-muted": workbenchVisualTokens.violetMuted,
        line: workbenchVisualTokens.line,
        "line-soft": workbenchVisualTokens.lineSoft,
        "line-hover": workbenchVisualTokens.lineHover,
        border: workbenchVisualTokens.line,
        surface: workbenchVisualTokens.bg2,
        control: workbenchVisualTokens.control,
        "control-muted": workbenchVisualTokens.controlMuted,
        "control-subtle": workbenchVisualTokens.controlSubtle,
        "control-active": workbenchVisualTokens.controlActive,
        "control-hover": workbenchVisualTokens.controlHover,
        "control-field": workbenchVisualTokens.controlField,
        "control-track": workbenchVisualTokens.controlTrack,
        muted: workbenchVisualTokens.inkDim,
        accent: workbenchVisualTokens.violet,
        amberline: workbenchVisualTokens.amber,
        danger: workbenchVisualTokens.danger,
        "danger-text": workbenchVisualTokens.dangerText,
        "danger-detail": workbenchVisualTokens.dangerDetail,
        "danger-hover": workbenchVisualTokens.dangerHover,
        focus: workbenchVisualTokens.focus,
        subtle: workbenchVisualTokens.lineSoft,
        faint: workbenchVisualTokens.faint,
        "accent-soft": workbenchVisualTokens.accentSoft,
        "accent-line": workbenchVisualTokens.accentLine,
        "accent-edge": workbenchVisualTokens.accentEdge,
        "danger-soft": workbenchVisualTokens.dangerSoft,
        "danger-line": workbenchVisualTokens.dangerLine,
      },
      borderColor: {
        line: workbenchVisualTokens.line,
        "line-soft": workbenchVisualTokens.lineSoft,
      },
      borderRadius: {
        card: "16px",
        ctl: "11px",
        control: "10px",
        "control-sm": "7px",
        "control-md": "8px",
        "control-group": "13px",
        chip: "8px",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      backgroundImage: {
        grad: `linear-gradient(135deg,${workbenchVisualTokens.gradientStart} 0%,${workbenchVisualTokens.gradientMiddle} 48%,${workbenchVisualTokens.gradientEnd} 100%)`,
        "control-chrome": `linear-gradient(155deg,${workbenchVisualTokens.cardA},${workbenchVisualTokens.cardB})`,
        "control-selected":
          "linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))",
        "grad-soft":
          "linear-gradient(135deg, rgba(169,139,255,0.55), rgba(124,141,255,0.18) 45%, rgba(255,255,255,0.05) 68%, rgba(111,195,255,0.42))",
        "edge-grad":
          `linear-gradient(155deg, var(--tw-gradient-from, ${workbenchVisualTokens.cardA}), var(--tw-gradient-to, ${workbenchVisualTokens.cardB}))`,
      },
      boxShadow: {
        panel: "0 12px 34px -24px rgba(0,0,0,0.85)",
        primary:
          "0 6px 22px -6px rgba(124,109,255,0.6), inset 0 1px 0 rgba(255,255,255,0.28)",
        "control-active": "0 4px 12px -4px rgba(124,109,255,0.7)",
        "control-checked": "0 3px 10px -3px rgba(124,109,255,0.8)",
        "switch-checked": "0 4px 14px -6px rgba(124,109,255,0.85)",
        "node-sel":
          "0 0 0 1px rgba(146,113,255,0.18), 0 18px 50px -20px rgba(124,92,255,0.7), 0 0 60px -12px rgba(124,92,255,0.35)",
        "status-ok": "0 0 10px 1px rgba(86,214,160,0.8)",
        "status-danger": "0 0 10px 1px rgba(251,113,133,0.55)",
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
