import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}", "./tests/**/*.{ts,tsx}"],
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
        "ink-faint": "#62627c",
        violet: "#a78bfa",
        "violet-deep": "#8b5cf6",
        blue: "#6ea8ff",
        cyan: "#7fd0ff",
        ok: "#56d6a0",
        amber: "#ffd166",
        line: "rgba(255,255,255,0.07)",
        "line-soft": "rgba(255,255,255,0.045)",
        border: "rgba(255,255,255,0.07)",
        surface: "#090910",
        muted: "#9c9cb6",
        accent: "#a78bfa",
        amberline: "#ffd166",
        danger: "#fb7185",
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
        chip: "8px",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      backgroundImage: {
        grad: "linear-gradient(135deg,#a98bff 0%,#7c8dff 48%,#6fc3ff 100%)",
        "grad-soft":
          "linear-gradient(135deg, rgba(169,139,255,0.55), rgba(124,141,255,0.18) 45%, rgba(255,255,255,0.05) 68%, rgba(111,195,255,0.42))",
        "edge-grad":
          "linear-gradient(155deg, var(--tw-gradient-from, #161622), var(--tw-gradient-to, #0e0e18))",
      },
      boxShadow: {
        panel: "0 12px 34px -24px rgba(0,0,0,0.85)",
        primary:
          "0 6px 22px -6px rgba(124,109,255,0.6), inset 0 1px 0 rgba(255,255,255,0.28)",
        "node-sel":
          "0 0 0 1px rgba(146,113,255,0.18), 0 18px 50px -20px rgba(124,92,255,0.7), 0 0 60px -12px rgba(124,92,255,0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
