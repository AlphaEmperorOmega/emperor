import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}", "./tests/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "#d8ded9",
        surface: "#f3f5f2",
        panel: "#ffffff",
        ink: "#18221d",
        muted: "#66716b",
        accent: "#15705f",
        amberline: "#a96615",
        danger: "#b4232b",
        focus: "#15705f26",
        subtle: "#e3e7e2",
        faint: "#bdc7c1",
        "accent-soft": "#edf8f3",
        "accent-line": "#9fcfbd",
        "accent-edge": "#b9cfc7",
        "danger-soft": "#fff6f4",
        "danger-line": "#efb9bd",
      },
      boxShadow: {
        panel: "0 1px 2px rgba(24, 34, 29, 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;
