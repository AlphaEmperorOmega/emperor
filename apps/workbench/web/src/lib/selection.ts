import type { ConfigValue } from "@/lib/api/schemas";
import type { ModelIdentity } from "@/lib/api/model-catalog";

function modelTypeLabel(value: string) {
  const normalized = value.replace(/[_-]+/g, " ").trim().toLowerCase();
  return normalized ? normalized[0].toUpperCase() + normalized.slice(1) : value;
}

export function modelTypeForId(identity: ModelIdentity) {
  return identity.modelType;
}

export function modelNameForId(identity: ModelIdentity) {
  return identity.model;
}

export function modelIdentityKey(identity: ModelIdentity) {
  return `${identity.modelType}/${identity.model}`;
}

export function modelTypeOptions(models: readonly ModelIdentity[]) {
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

export function modelsForType<T extends ModelIdentity>(
  models: readonly T[],
  selectedType: string,
) {
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
