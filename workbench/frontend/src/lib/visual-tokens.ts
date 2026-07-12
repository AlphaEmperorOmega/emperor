type StringTokenSet = Readonly<Record<string, string>>;

function kebabCase(value: string) {
  return value.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
}

function colorChannels(value: string) {
  const hex = value.match(/^#([\da-f]{6})$/i)?.[1];
  if (hex) {
    return {
      alpha: "1",
      rgb: [0, 2, 4]
        .map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16))
        .join(" "),
    };
  }

  const rgba = value.match(
    /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)$/i,
  );
  if (!rgba) {
    throw new Error(`Unsupported Workbench color token: ${value}`);
  }
  return {
    alpha: rgba[4] ?? "1",
    rgb: `${rgba[1]} ${rgba[2]} ${rgba[3]}`,
  };
}

function cssVariables(prefix: string, tokens: StringTokenSet) {
  return Object.fromEntries(
    Object.entries(tokens).map(([name, value]) => [
      `--${prefix}-${kebabCase(name)}`,
      value,
    ]),
  );
}

/**
 * Canonical Workbench color registry. Raw UI color values belong here only;
 * DOM, Tailwind, canvas, ECharts, React Flow, and Three.js consume adapters
 * derived from these semantic facts.
 */
export const workbenchVisualTokens = Object.freeze({
  accentEdge: "rgba(169,154,255,0.34)",
  accentFill: "rgba(169,154,255,0.13)",
  accentLine: "rgba(169,154,255,0.38)",
  accentSoft: "rgba(169,154,255,0.12)",
  amber: "#f6c85f",
  bg: "#07080d",
  bg2: "#0a0b12",
  black: "#000000",
  blue: "#7aa7ff",
  cardA: "#151823",
  cardB: "#10121a",
  chartAmber: "#f59e0b",
  chartBlue: "#60a5fa",
  chartCyan: "#22d3ee",
  chartEmerald: "#34d399",
  chartPink: "#f472b6",
  chartRose: "#fb7185",
  chartTeal: "#2dd4bf",
  chartViolet: "#7c6dff",
  chartVioletSoft: "#a78bfa",
  chartYellow: "#facc15",
  checkpointMarker: "rgba(244,242,250,0.46)",
  clusterBounds: "#9ca3af",
  clusterGhost: "#64748b",
  clusterGrown: "#22d3ee",
  clusterInitial: "#8da2ff",
  clusterReach: "#94a3b8",
  clusterReachWireframe: "#cbd5e1",
  clusterRecent: "#f59e0b",
  clusterSelected: "#f8fafc",
  control: "rgba(244,242,250,0.045)",
  controlActive: "rgba(244,242,250,0.075)",
  controlField: "rgba(4,5,9,0.52)",
  controlHover: "rgba(244,242,250,0.085)",
  controlMuted: "rgba(244,242,250,0.025)",
  controlSubtle: "rgba(244,242,250,0.035)",
  controlTrack: "rgba(244,242,250,0.08)",
  cyan: "#67d7ff",
  danger: "#ff748d",
  dangerDetail: "#ffd0d8",
  dangerHover: "#8f2639",
  dangerLine: "rgba(255,116,141,0.38)",
  dangerSoft: "rgba(143,38,57,0.22)",
  dangerText: "#ff9daf",
  faint: "rgba(244,242,250,0.14)",
  flowHandle: "rgba(146,113,255,0.7)",
  flowHandleFill: "#0b0b14",
  focus: "rgba(169,154,255,0.72)",
  gradientEnd: "#67d7ff",
  gradientMiddle: "#858fff",
  gradientStart: "#b29cff",
  graphGrid: "rgba(244,242,250,0.055)",
  heatmapCorrect: "#40c97f",
  heatmapIncorrect: "#ef5656",
  heatmapLabel: "rgba(210,219,235,0.58)",
  heatmapValue: "rgba(255,255,255,0.94)",
  ink: "#f4f2fa",
  inkDim: "#b7bbcc",
  inkFaint: "#9296aa",
  line: "rgba(244,242,250,0.105)",
  lineHover: "rgba(244,242,250,0.2)",
  lineSoft: "rgba(244,242,250,0.065)",
  modifiedField: "#100719",
  ok: "#5ee1a5",
  panel: "#0d0f17",
  panel2: "#11131d",
  scene: "#05070d",
  selectedControl: "#7058dc",
  structureOverlay: "rgba(8,9,15,0.92)",
  violet: "#a99aff",
  violetDeep: "#8f7cff",
  violetMuted: "#d0c7ff",
  violetText: "#ddd7ff",
  white: "#ffffff",
});

export type WorkbenchVisualTokenName = keyof typeof workbenchVisualTokens;

/** A 4 px spatial rhythm for dense controls and larger shell regions. */
export const workbenchSpacingTokens = Object.freeze({
  unit: "0.25rem",
  compact: "0.5rem",
  panel: "0.75rem",
  region: "1rem",
  shell: "1.25rem",
  shellWide: "1.5rem",
  section: "2rem",
});

export type WorkbenchSpacingTokenName = keyof typeof workbenchSpacingTokens;

export const workbenchControlTokens = Object.freeze({
  compact: "2rem",
  default: "2.25rem",
  comfortable: "2.5rem",
  touch: "2.75rem",
});

export type WorkbenchControlTokenName = keyof typeof workbenchControlTokens;

export const workbenchTypographyTokens = Object.freeze({
  micro: Object.freeze({
    fontSize: "0.5625rem",
    lineHeight: "0.75rem",
    letterSpacing: "0.04em",
  }),
  caption: Object.freeze({
    fontSize: "0.625rem",
    lineHeight: "0.875rem",
    letterSpacing: "0.03em",
  }),
  meta: Object.freeze({
    fontSize: "0.6875rem",
    lineHeight: "1rem",
    letterSpacing: "0.02em",
  }),
  label: Object.freeze({
    fontSize: "0.75rem",
    lineHeight: "1rem",
    letterSpacing: "0.08em",
  }),
  compact: Object.freeze({
    fontSize: "0.8125rem",
    lineHeight: "1.125rem",
    letterSpacing: "0",
  }),
  body: Object.freeze({
    fontSize: "0.875rem",
    lineHeight: "1.375rem",
    letterSpacing: "0",
  }),
  title: Object.freeze({
    fontSize: "1rem",
    lineHeight: "1.5rem",
    letterSpacing: "-0.01em",
  }),
  heading: Object.freeze({
    fontSize: "1.125rem",
    lineHeight: "1.5rem",
    letterSpacing: "-0.015em",
  }),
  display: Object.freeze({
    fontSize: "1.375rem",
    lineHeight: "1.75rem",
    letterSpacing: "-0.025em",
  }),
});

export type WorkbenchTypographyTokenName =
  keyof typeof workbenchTypographyTokens;

export const workbenchFontTokens = Object.freeze({
  sans: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  mono:
    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
});

export const workbenchTrackingTokens = Object.freeze({
  meta: "0.02em",
  caption: "0.04em",
  label: "0.08em",
  wordmark: "0.18em",
  display: "-0.02em",
});

export const workbenchRadiusTokens = Object.freeze({
  indicator: "0.25rem",
  chip: "0.375rem",
  controlSm: "0.375rem",
  controlMd: "0.5rem",
  control: "0.625rem",
  panel: "0.75rem",
  card: "0.875rem",
  dialog: "1rem",
  round: "999px",
});

export type WorkbenchRadiusTokenName = keyof typeof workbenchRadiusTokens;

export const workbenchBorderTokens = Object.freeze({
  hairline: "1px",
  strong: "2px",
});

export const workbenchElevationTokens = Object.freeze({
  control: "inset 0 1px 0 rgba(244,242,250,0.025)",
  controlAccent: "inset 0 1px 0 rgba(221,215,255,0.06)",
  controlLift: "0 1px 2px rgba(0,0,0,0.42)",
  controlSelected: "inset 0 0 0 1px rgba(169,154,255,0.16)",
  controlWarning: "inset 0 0 0 1px rgba(255,209,102,0.08)",
  cyanInset: "inset 0 0 0 1px rgba(127,208,255,0.28)",
  divider: "inset 0 -1px 0 rgba(244,242,250,0.02)",
  panel:
    "0 16px 42px -30px rgba(0,0,0,0.95), inset 0 1px 0 rgba(244,242,250,0.035)",
  primary:
    "0 8px 20px -14px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.22)",
  controlActive:
    "0 6px 18px -12px rgba(0,0,0,0.9), inset 0 0 0 1px rgba(169,154,255,0.16)",
  controlChecked: "0 4px 12px -8px rgba(0,0,0,0.9)",
  switchChecked: "0 4px 14px -9px rgba(0,0,0,0.9)",
  card: "0 16px 40px -30px rgba(0,0,0,0.95)",
  cardHover: "0 20px 44px -32px rgba(0,0,0,0.95)",
  cardSubtle: "0 10px 28px -26px rgba(0,0,0,0.9)",
  cardAccent: "0 18px 42px -32px rgba(0,0,0,0.95)",
  cardWarning: "0 18px 42px -32px rgba(0,0,0,0.95)",
  floating: "0 14px 34px rgba(0,0,0,0.32)",
  popover: "0 22px 54px -26px rgba(0,0,0,0.98)",
  dialog: "0 28px 90px rgba(0,0,0,0.68)",
  dialogCompact: "0 24px 80px -30px rgba(0,0,0,0.9)",
  node:
    "0 0 0 1px rgba(169,154,255,0.24), 0 18px 44px -30px rgba(0,0,0,0.95)",
  statusOk: "0 0 0 3px rgba(94,225,165,0.12)",
  statusDanger: "0 0 0 3px rgba(255,116,141,0.12)",
  cyanSelection: "0 0 0 2px rgba(127,208,255,0.42)",
});

export const workbenchFocusTokens = Object.freeze({
  ringWidth: "2px",
  ringOffset: "2px",
});

export const workbenchGradientTokens = Object.freeze({
  primary:
    "linear-gradient(135deg, #b29cff 0%, #858fff 48%, #67d7ff 100%)",
  body:
    "linear-gradient(180deg, rgba(17,19,29,0.38), transparent 28%)",
  ambient:
    "linear-gradient(rgba(244,242,250,0.016) 1px, transparent 1px), linear-gradient(90deg, rgba(244,242,250,0.016) 1px, transparent 1px), radial-gradient(760px 500px at 72% -18%, rgba(143,124,255,0.09), transparent 68%)",
  panelSheen:
    "linear-gradient(180deg, rgba(244,242,250,0.035), transparent 34%)",
  dialogChrome:
    "linear-gradient(180deg, rgba(244,242,250,0.035), transparent 36%), rgba(13,15,23,0.96)",
  dialogBody:
    "linear-gradient(180deg, rgba(244,242,250,0.012), transparent 18%), rgba(244,242,250,0.006)",
  selected:
    "linear-gradient(180deg, rgba(169,154,255,0.11), transparent 42%), #12131f",
  graph:
    "radial-gradient(720px 520px at 50% 46%, rgba(143,124,255,0.055), transparent 68%)",
  configPreset:
    "linear-gradient(135deg, rgba(255,209,102,0.075), rgba(167,139,250,0.105))",
  configPresetHeader:
    "linear-gradient(90deg, rgba(255,209,102,0.12), rgba(167,139,250,0.13))",
  configPresetHeaderHover:
    "linear-gradient(90deg, rgba(255,209,102,0.16), rgba(167,139,250,0.17))",
  configNavigation:
    "linear-gradient(90deg, rgba(255,209,102,0.075), rgba(167,139,250,0.095))",
  configNavigationHover:
    "linear-gradient(90deg, rgba(255,209,102,0.11), rgba(167,139,250,0.13))",
  clusterActive:
    "linear-gradient(135deg, rgba(146,113,255,0.88), rgba(111,168,255,0.58))",
  minimap:
    "radial-gradient(circle at 35% 20%, rgba(34,211,238,0.08), transparent 34%), linear-gradient(180deg, rgba(7,9,16,0.96), rgba(5,6,10,0.98))",
  clusterPanel:
    "linear-gradient(180deg, rgba(14,16,24,0.98), rgba(8,10,16,0.98))",
  componentInfo:
    "linear-gradient(180deg, rgba(20,23,33,0.98), rgba(12,14,21,0.98))",
});

export const workbenchOpacityTokens = Object.freeze({
  hidden: "0",
  faint: "0.3",
  low: "0.35",
  restrained: "0.45",
  disabled: "0.5",
  soft: "0.6",
  muted: "0.65",
  quiet: "0.7",
  subdued: "0.75",
  resting: "0.82",
  strong: "0.9",
  visible: "1",
  scrim: "0.8",
});

export const workbenchMotionTokens = Object.freeze({
  durationFast: "100ms",
  duration: "150ms",
  durationSlow: "220ms",
  durationSpin: "1s",
  durationPulse: "2s",
  easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
  easeStandard: "cubic-bezier(0.4, 0, 0.6, 1)",
});

export const workbenchVisualizationTokens = Object.freeze({
  multiRunLineColors: Object.freeze([
    workbenchVisualTokens.chartCyan,
    workbenchVisualTokens.chartVioletSoft,
    workbenchVisualTokens.chartYellow,
    workbenchVisualTokens.chartRose,
    workbenchVisualTokens.chartEmerald,
  ]),
  scalarSeriesColors: Object.freeze([
    workbenchVisualTokens.chartViolet,
    workbenchVisualTokens.chartCyan,
    workbenchVisualTokens.chartAmber,
    workbenchVisualTokens.chartEmerald,
    workbenchVisualTokens.chartPink,
    workbenchVisualTokens.chartVioletSoft,
    workbenchVisualTokens.chartRose,
    workbenchVisualTokens.chartBlue,
    workbenchVisualTokens.chartYellow,
    workbenchVisualTokens.chartTeal,
  ]),
  clusterCategories: Object.freeze({
    initial: workbenchVisualTokens.clusterInitial,
    grown: workbenchVisualTokens.clusterGrown,
    recentAdded: workbenchVisualTokens.clusterRecent,
  }),
});

const colorVariableEntries = Object.entries(workbenchVisualTokens).flatMap(
  ([name, value]) => {
    const cssName = kebabCase(name);
    const channels = colorChannels(value);
    return [
      [`--color-${cssName}`, value],
      [`--color-${cssName}-rgb`, channels.rgb],
      [`--color-${cssName}-alpha`, channels.alpha],
    ];
  },
);

const typographyVariableEntries = Object.entries(
  workbenchTypographyTokens,
).flatMap(([name, value]) => {
  const cssName = kebabCase(name);
  return [
    [`--type-${cssName}-size`, value.fontSize],
    [`--type-${cssName}-leading`, value.lineHeight],
    [`--type-${cssName}-tracking`, value.letterSpacing],
  ];
});

/** Complete runtime custom-property projection of the canonical registry. */
export const workbenchCssVariables = Object.freeze(
  Object.fromEntries([
    ...colorVariableEntries,
    ...typographyVariableEntries,
    ...Object.entries(cssVariables("font-fallback", workbenchFontTokens)),
    ...Object.entries(cssVariables("tracking", workbenchTrackingTokens)),
    ...Object.entries(cssVariables("space", workbenchSpacingTokens)),
    ...Object.entries(cssVariables("control", workbenchControlTokens)),
    ...Object.entries(cssVariables("radius", workbenchRadiusTokens)),
    ...Object.entries(cssVariables("border", workbenchBorderTokens)),
    ...Object.entries(cssVariables("elevation", workbenchElevationTokens)),
    ...Object.entries(cssVariables("focus", workbenchFocusTokens)),
    ...Object.entries(cssVariables("gradient", workbenchGradientTokens)),
    ...Object.entries(cssVariables("opacity", workbenchOpacityTokens)),
    ...Object.entries(cssVariables("motion", workbenchMotionTokens)),
  ]) as Record<`--${string}`, string>,
);

/**
 * Tailwind color adapter. RGB-channel variables preserve `/opacity` modifiers;
 * colors with an intrinsic alpha retain it when no modifier is supplied.
 */
export function workbenchTailwindColor(token: WorkbenchVisualTokenName) {
  const cssName = kebabCase(token);
  return ({ opacityValue }: { opacityValue?: string }) =>
    `rgb(var(--color-${cssName}-rgb) / ${
      opacityValue === undefined
        ? `var(--color-${cssName}-alpha)`
        : `calc(var(--color-${cssName}-alpha) * ${opacityValue})`
    })`;
}

/** Canvas adapter for data-dependent opacity without duplicating RGB values. */
export function workbenchColorWithAlpha(
  token: WorkbenchVisualTokenName,
  alpha: number,
) {
  const channels = colorChannels(workbenchVisualTokens[token]).rgb.replaceAll(
    " ",
    ", ",
  );
  return `rgba(${channels}, ${alpha})`;
}
