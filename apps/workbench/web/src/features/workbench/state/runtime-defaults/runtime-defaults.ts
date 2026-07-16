import { type ConfigField } from "@/lib/api/models";
import {
  ADAPTIVE_OPTION_PAIRS,
  concreteConfigOptionChoice,
  configKeyToken,
  defaultConfigFieldValue,
  fieldValue,
  hasOverride,
  isPresentConfigOptionValue,
  normalizeConfigFieldValue,
  overrideValue,
  type OverrideValues,
} from "@/lib/config";

export type RuntimeDefaultsMetricState =
  | "default"
  | "override"
  | "preset"
  | "override-and-preset";

export type RuntimeDefaultsMetrics = {
  fieldCount: number;
  overrideCount: number;
  presetCount: number;
  state: RuntimeDefaultsMetricState;
};

function fieldMatchesKey(field: ConfigField, key: string) {
  const token = configKeyToken(key);
  return (
    configKeyToken(field.key) === token ||
    configKeyToken(field.configKey) === token
  );
}

function fieldByKey(fields: ConfigField[], key: string) {
  return fields.find((field) => fieldMatchesKey(field, key));
}

function canonicalizeOverrideKeys(
  fields: ConfigField[],
  overrides: OverrideValues,
): OverrideValues {
  const fieldsByKey = new Map<string, ConfigField>();
  for (const field of fields) {
    fieldsByKey.set(configKeyToken(field.key), field);
    fieldsByKey.set(configKeyToken(field.configKey), field);
  }
  return Object.fromEntries(
    Object.entries(overrides).map(([key, value]) => [
      fieldsByKey.get(configKeyToken(key))?.key ?? key,
      value,
    ]),
  );
}

function deleteOverrideByKey(overrides: OverrideValues, key: string) {
  const token = configKeyToken(key);
  for (const overrideKey of Object.keys(overrides)) {
    if (configKeyToken(overrideKey) === token) {
      delete overrides[overrideKey];
    }
  }
}

export function isEnabledRuntimeDefaultValue(value: string) {
  return ["true", "1", "yes", "on"].includes(value.trim().toLowerCase());
}

function normalizeAdaptiveOptionOverrides(
  fields: ConfigField[],
  overrides: OverrideValues,
): OverrideValues {
  const next = { ...overrides };

  for (const { flagKey, optionKey } of ADAPTIVE_OPTION_PAIRS) {
    const flagField = fieldByKey(fields, flagKey);
    const optionField = fieldByKey(fields, optionKey);
    if (!flagField || !optionField) {
      continue;
    }

    const flagValue = fieldValue(flagField, next);
    if (isEnabledRuntimeDefaultValue(flagValue)) {
      if (!isPresentConfigOptionValue(overrideValue(next, optionField.key))) {
        const optionChoice = concreteConfigOptionChoice(optionField);
        if (optionChoice !== undefined) {
          deleteOverrideByKey(next, optionField.key);
          next[optionField.key] = optionChoice;
        }
      }
      continue;
    }

    deleteOverrideByKey(next, optionField.key);
  }

  return next;
}

function overridesEqual(left: OverrideValues, right: OverrideValues) {
  const leftEntries = Object.entries(left);
  const rightEntries = Object.entries(right);
  return (
    leftEntries.length === rightEntries.length &&
    leftEntries.every(([key, value]) => right[key] === value)
  );
}

function withoutOverride(overrides: OverrideValues, key: string) {
  const token = configKeyToken(key);
  return Object.fromEntries(
    Object.entries(overrides).filter(
      ([overrideKey]) => configKeyToken(overrideKey) !== token,
    ),
  );
}

function normalizedOverrides(
  fields: ConfigField[],
  overrides: OverrideValues,
) {
  return normalizeAdaptiveOptionOverrides(
    fields,
    canonicalizeOverrideKeys(fields, overrides),
  );
}

function preserveIdentity(current: OverrideValues, next: OverrideValues) {
  return overridesEqual(current, next) ? current : next;
}

/**
 * Canonical Runtime Defaults editing for lifecycle owners. Drafts retain
 * preset-owned values so they can become active again under another preset;
 * effectivePresetOverrides is the read-side active projection.
 */
export const runtimeDefaultsEditor = {
  normalize(fields: ConfigField[], current: OverrideValues) {
    return preserveIdentity(current, normalizedOverrides(fields, current));
  },

  replace(fields: ConfigField[], overrides: OverrideValues | undefined) {
    return normalizedOverrides(fields, overrides ?? {});
  },

  edit(
    fields: ConfigField[],
    current: OverrideValues,
    key: string,
    value: string,
  ) {
    const field = fieldByKey(fields, key);
    const token = configKeyToken(key);
    const existingKey = Object.keys(current).find(
      (overrideKey) => configKeyToken(overrideKey) === token,
    );
    const withoutExisting = withoutOverride(current, key);
    const isDefaultValue =
      field &&
      normalizeConfigFieldValue(field, value) === defaultConfigFieldValue(field);
    const edited = isDefaultValue
      ? withoutExisting
      : {
          ...withoutExisting,
          [field?.key ?? existingKey ?? key]: value,
        };
    return preserveIdentity(current, normalizedOverrides(fields, edited));
  },

  clear(fields: ConfigField[], current: OverrideValues, key: string) {
    return preserveIdentity(
      current,
      normalizedOverrides(fields, withoutOverride(current, key)),
    );
  },
};

export function inactivePresetOwnedOverrideKeys(
  fields: ConfigField[],
  overrides: OverrideValues,
) {
  const presetOwnedKeys = new Set(
    fields
      .filter((field) => field.locked)
      .map((field) => configKeyToken(field.key)),
  );
  return Object.keys(overrides)
    .filter((key) => presetOwnedKeys.has(configKeyToken(key)))
    .sort((left, right) => left.localeCompare(right));
}

export function effectivePresetOverrides(
  fields: ConfigField[],
  overrides: OverrideValues,
): OverrideValues {
  const inactiveKeys = new Set(
    inactivePresetOwnedOverrideKeys(fields, overrides),
  );
  if (inactiveKeys.size === 0) {
    return { ...overrides };
  }
  return Object.fromEntries(
    Object.entries(overrides).filter(([key]) => !inactiveKeys.has(key)),
  );
}

export function modifiedRuntimeDefaultCount(
  fields: ConfigField[],
  overrides: OverrideValues,
) {
  return fields.filter((field) => hasOverride(overrides, field.key)).length;
}

export function presetOwnedRuntimeDefaultCount(fields: ConfigField[]) {
  return fields.filter((field) => Boolean(field.locked)).length;
}

export function runtimeDefaultsMetrics(
  fields: ConfigField[],
  overrides: OverrideValues,
): RuntimeDefaultsMetrics {
  const overrideCount = modifiedRuntimeDefaultCount(fields, overrides);
  const presetCount = presetOwnedRuntimeDefaultCount(fields);
  const state =
    overrideCount > 0 && presetCount > 0
      ? "override-and-preset"
      : overrideCount > 0
        ? "override"
        : presetCount > 0
          ? "preset"
          : "default";
  return {
    fieldCount: fields.length,
    overrideCount,
    presetCount,
    state,
  };
}
