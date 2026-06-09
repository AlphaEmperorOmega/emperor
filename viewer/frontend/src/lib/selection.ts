import { type ConfigValue } from "@/lib/api";

const DEFAULT_MODEL_TYPE = "models";

function modelTypeLabel(value: string) {
  const normalized = value.replace(/[_-]+/g, " ").trim().toLowerCase();
  return normalized ? normalized[0].toUpperCase() + normalized.slice(1) : value;
}

export function modelTypeForId(modelId: string) {
  const trimmed = modelId.trim();
  const separatorIndex = trimmed.indexOf("/");
  if (separatorIndex <= 0) {
    return DEFAULT_MODEL_TYPE;
  }
  return trimmed.slice(0, separatorIndex).trim() || DEFAULT_MODEL_TYPE;
}

export function modelNameForId(modelId: string) {
  const trimmed = modelId.trim();
  const separatorIndex = trimmed.indexOf("/");
  if (separatorIndex < 0 || separatorIndex === trimmed.length - 1) {
    return trimmed;
  }
  return trimmed.slice(separatorIndex + 1).trim() || trimmed;
}

export function modelTypeOptions(models: readonly string[]) {
  const seen = new Set<string>();
  return models.reduce<Array<{ value: string; label: string }>>((options, model) => {
    const value = modelTypeForId(model);
    if (!seen.has(value)) {
      seen.add(value);
      options.push({ value, label: modelTypeLabel(value) });
    }
    return options;
  }, []);
}

export function modelsForType(models: readonly string[], selectedType: string) {
  if (!selectedType) {
    return [...models];
  }
  return models.filter((model) => modelTypeForId(model) === selectedType);
}

export function uniqueValidValues<T>(
  values: readonly T[],
  validValues: readonly T[],
) {
  const validValueSet = new Set(validValues);
  const seen = new Set<T>();
  return values.filter((value) => {
    if (!validValueSet.has(value) || seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
}

export function selectionValuesEqual<T>(
  left: readonly T[],
  right: readonly T[],
) {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

export function normalizeSelection<T>(
  values: readonly T[],
  validValues: readonly T[],
  fallback: readonly T[] = [],
) {
  const selectedValues = uniqueValidValues(values, validValues);
  if (selectedValues.length > 0) {
    return selectedValues;
  }
  if (validValues.length === 0) {
    return [];
  }
  const fallbackValues = uniqueValidValues(fallback, validValues);
  if (fallbackValues.length > 0) {
    return [fallbackValues[0]];
  }
  return [validValues[0]];
}

export function normalizePrimarySelection<T>(
  values: readonly T[],
  validValues: readonly T[],
  primaryValue?: T,
) {
  const selectedValues = uniqueValidValues(values, validValues);
  const validPrimary =
    primaryValue !== undefined && validValues.includes(primaryValue)
      ? primaryValue
      : undefined;
  if (selectedValues.length === 0) {
    return validPrimary === undefined ? [] : [validPrimary];
  }
  if (validPrimary === undefined || !selectedValues.includes(validPrimary)) {
    return selectedValues;
  }
  return [
    validPrimary,
    ...selectedValues.filter((value) => value !== validPrimary),
  ];
}

export function toggleSetValue<T>(set: ReadonlySet<T>, value: T) {
  const next = new Set(set);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

function configValueKey(value: ConfigValue) {
  return value === null ? "null:null" : `${typeof value}:${String(value)}`;
}

export function configValueEquals(a: ConfigValue, b: ConfigValue) {
  return configValueKey(a) === configValueKey(b);
}

export function valueIsSelected(
  values: readonly ConfigValue[],
  value: ConfigValue,
) {
  return values.some((candidate) => configValueEquals(candidate, value));
}
