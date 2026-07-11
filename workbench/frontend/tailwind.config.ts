import type { Config } from "tailwindcss";

const config: Config = {
  // Production CSS is generated only from app/source files. Vitest component
  // tests import those components directly, so scanning tests would only keep
  // test-only class strings in the shipped stylesheet.
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#06060c",
        "bg-2": "#090910",
        panel: "#0c0c15",
        "panel-2": "#0f0f1a",
        "card-a": "#161622",
        "card-b": "#0e0e18",
        ink: "#ecebf5",
        "ink-dim": "#9c9cb6",
        "ink-faint": "#80809b",
        violet: "#a78bfa",
        "violet-deep": "#8b5cf6",
        blue: "#6ea8ff",
        cyan: "#7fd0ff",
        ok: "#56d6a0",
        amber: "#ffd166",
        "violet-text": "#d7c9ff",
        "violet-muted": "#cdbcff",
        line: "rgba(255,255,255,0.07)",
        "line-soft": "rgba(255,255,255,0.045)",
        "line-hover": "rgba(255,255,255,0.15)",
        border: "rgba(255,255,255,0.07)",
        surface: "#090910",
        control: "rgba(255,255,255,0.035)",
        "control-muted": "rgba(255,255,255,0.025)",
        "control-subtle": "rgba(255,255,255,0.03)",
        "control-active": "rgba(255,255,255,0.055)",
        "control-hover": "rgba(255,255,255,0.07)",
        "control-field": "rgba(0,0,0,0.25)",
        "control-track": "rgba(255,255,255,0.06)",
        muted: "#9c9cb6",
        accent: "#a78bfa",
        amberline: "#ffd166",
        danger: "#fb7185",
        "danger-text": "#fda4af",
        "danger-detail": "#fecdd3",
        "danger-hover": "#7f1d2d",
        focus: "rgba(167,139,250,0.32)",
        subtle: "rgba(255,255,255,0.045)",
        faint: "rgba(255,255,255,0.12)",
        "accent-soft": "rgba(146,113,255,0.14)",
        "accent-line": "rgba(146,113,255,0.32)",
        "accent-edge": "rgba(146,113,255,0.26)",
        "danger-soft": "rgba(127,29,44,0.24)",
        "danger-line": "rgba(251,113,133,0.34)",
      },
      borderColor: {
        line: "rgba(255,255,255,0.07)",
        "line-soft": "rgba(255,255,255,0.045)",
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
        grad: "linear-gradient(135deg,#a98bff 0%,#7c8dff 48%,#6fc3ff 100%)",
        "control-chrome": "linear-gradient(155deg,#161622,#0e0e18)",
        "control-selected":
          "linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))",
        "grad-soft":
          "linear-gradient(135deg, rgba(169,139,255,0.55), rgba(124,141,255,0.18) 45%, rgba(255,255,255,0.05) 68%, rgba(111,195,255,0.42))",
        "edge-grad":
          "linear-gradient(155deg, var(--tw-gradient-from, #161622), var(--tw-gradient-to, #0e0e18))",
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
  plugins: [],
};

export default config;
