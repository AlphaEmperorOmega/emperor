import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import tailwindConfig from "../../tailwind.config";

const MINIMUM_NORMAL_TEXT_CONTRAST = 4.5;
const primarySurfaceNames = ["bg", "bg-2", "panel", "panel-2", "card-a", "card-b"] as const;

type ThemeColor = (typeof primarySurfaceNames)[number] | "ink-faint";

const themeColors = tailwindConfig.theme?.extend?.colors as Record<ThemeColor, string>;
const globalsCss = readFileSync(
  fileURLToPath(new URL("../../app/globals.css", import.meta.url)),
  "utf8",
);

function cssCustomProperty(name: ThemeColor) {
  const match = globalsCss.match(new RegExp(`--${name}:\\s*(#[0-9a-f]{6});`, "i"));
  if (!match) {
    throw new Error(`Missing --${name} CSS custom property`);
  }
  return match[1].toLowerCase();
}

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

describe("normal faint-text theme contrast", () => {
  it("keeps the CSS custom properties and Tailwind colors synchronized", () => {
    for (const colorName of ["ink-faint", ...primarySurfaceNames] as const) {
      expect(cssCustomProperty(colorName)).toBe(themeColors[colorName].toLowerCase());
    }
  });

  it("meets WCAG AA contrast on every primary surface", () => {
    const faintText = themeColors["ink-faint"];

    for (const surfaceName of primarySurfaceNames) {
      const surface = themeColors[surfaceName];
      expect(
        contrastRatio(faintText, surface),
        `ink-faint ${faintText} on ${surfaceName} ${surface}`,
      ).toBeGreaterThanOrEqual(MINIMUM_NORMAL_TEXT_CONTRAST);
    }
  });
});
