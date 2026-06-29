import { type ConfigField, type ConfigValue } from "@/lib/api";

export type OverrideValues = Record<string, string>;

export type ActiveOverrideScope = "preset" | "snapshot";

export type ConfigSection = {
  title: string;
  fields: ConfigField[];
  children?: ConfigSection[];
  controlFieldKey?: string;
};

export type ConfigFieldGroup = {
  title: string;
  fields: ConfigField[];
  controlField?: ConfigField;
  isEnabled?: boolean;
  disabledReason?: string;
};

export type ConfigSearchOption = {
  sectionTitle: string;
  field: ConfigField;
  key: string;
  label: string;
  configKey: string;
  flag: string;
  type: string;
  isModified: boolean;
  isLocked: boolean;
};

export type ConfigSearchState = {
  query: string;
  selectedFieldKey?: string | null;
};

export function configKeyToken(key: string) {
  return key.trim().replace(/-/g, "_").toLowerCase();
}

export function canonicalConfigKey(key: string) {
  return key.trim().replace(/-/g, "_").toUpperCase();
}

function configFieldMatchesKey(field: ConfigField, key: string) {
  const token = configKeyToken(key);
  return (
    configKeyToken(field.key) === token ||
    configKeyToken(field.configKey) === token
  );
}

export function overrideValue(overrides: OverrideValues, key: string) {
  if (Object.prototype.hasOwnProperty.call(overrides, key)) {
    return overrides[key];
  }
  const token = configKeyToken(key);
  const entry = Object.entries(overrides).find(
    ([overrideKey]) => configKeyToken(overrideKey) === token,
  );
  return entry?.[1];
}

export function normalizeConfigOverrides(
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

const CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE = new Map([
  ["Gate Options", "gate_flag"],
  ["Gate Stack Options", "gate_stack_independent_flag"],
  ["Halting Options", "halting_flag"],
  ["Halting Stack Options", "halting_stack_independent_flag"],
  ["Memory Options", "memory_flag"],
  ["Memory Stack Options", "memory_stack_independent_flag"],
  ["Recurrent Layer Options", "recurrent_flag"],
  ["Recurrent Gate Options", "recurrent_gate_flag"],
  ["Recurrent Gate Stack Options", "recurrent_gate_stack_independent_flag"],
  ["Recurrent Halting Options", "recurrent_halting_flag"],
  ["Recurrent Halting Stack Options", "recurrent_halting_stack_independent_flag"],
  ["Weight Options", "weight_option_flag"],
  ["Weight Generator Options", "weight_option_flag"],
  ["Weight Generator Stack Options", "weight_generator_stack_independent_flag"],
  ["Bias Options", "bias_option_flag"],
  ["Bias Generator Options", "bias_option_flag"],
  ["Bias Generator Stack Options", "bias_generator_stack_independent_flag"],
  ["Diagonal Options", "diagonal_option_flag"],
  ["Diagonal Generator Options", "diagonal_option_flag"],
  ["Diagonal Generator Stack Options", "diagonal_generator_stack_independent_flag"],
  ["Mask Options", "mask_option_flag"],
  ["Mask Stack Options", "mask_generator_stack_independent_flag"],
]);

const FALLBACK_CONTROLLED_SECTION_FLAG_KEYS = new Set([
  "gate_flag",
  "halting_flag",
  "memory_flag",
  "recurrent_flag",
]);

const GENERAL_CONFIG_SECTION = "General";
const RECURRENT_LAYER_CONFIG_SECTION = "Recurrent Layer Options";
const INPUT_BOUNDARY_PROJECTOR_SECTION = "Input Boundary Projector Options";
const OUTPUT_BOUNDARY_PROJECTOR_SECTION = "Output Boundary Projector Options";

export function displayConfigFieldSection(field: ConfigField) {
  return field.section || GENERAL_CONFIG_SECTION;
}

export function normalizeConfigFieldForDisplay(field: ConfigField): ConfigField {
  const section = displayConfigFieldSection(field);
  return field.section === section ? field : { ...field, section };
}

export function fieldValue(field: ConfigField, overrides: OverrideValues) {
  const value = field.locked
    ? field.lockedValue
    : (overrideValue(overrides, field.key) ?? field.default);
  return value === null || value === undefined ? "" : String(value);
}

export function defaultLabel(field: ConfigField) {
  return field.default === null || field.default === undefined ? "None" : String(field.default);
}

function normalizePrimitiveConfigValue(value: ConfigValue | string) {
  return value === null || value === undefined ? "" : String(value).trim();
}

export function normalizeConfigFieldValue(
  field: ConfigField,
  value: ConfigValue | string,
) {
  const raw = normalizePrimitiveConfigValue(value);
  if (field.nullable && raw === "") {
    return "null";
  }
  if (field.type === "bool") {
    const lower = raw.toLowerCase();
    if (lower === "true" || lower === "false") {
      return lower;
    }
  }
  if (field.type === "int") {
    const numberValue = Number(raw);
    if (Number.isInteger(numberValue)) {
      return String(numberValue);
    }
  }
  if (field.type === "float") {
    const numberValue = Number(raw);
    if (Number.isFinite(numberValue)) {
      return String(numberValue);
    }
  }
  return raw;
}

export function defaultConfigFieldValue(field: ConfigField) {
  return normalizeConfigFieldValue(field, field.default);
}

export function overrideValueForConfigField(
  field: ConfigField,
  value: ConfigValue | string,
) {
  const normalizedValue = normalizeConfigFieldValue(field, value);
  return field.nullable && normalizedValue === "null" ? "" : normalizedValue;
}

export function isDefaultConfigFieldValue(
  field: ConfigField,
  value: ConfigValue | string,
) {
  return normalizeConfigFieldValue(field, value) === defaultConfigFieldValue(field);
}

export function hasOverride(overrides: OverrideValues, key: string) {
  return overrideValue(overrides, key) !== undefined;
}

export const ADAPTIVE_OPTION_PAIRS = [
  { flagKey: "weight_option_flag", optionKey: "weight_option" },
  { flagKey: "bias_option_flag", optionKey: "bias_option" },
  { flagKey: "diagonal_option_flag", optionKey: "diagonal_option" },
  { flagKey: "mask_option_flag", optionKey: "row_mask_option" },
] as const;

function configFieldByKey(fields: ConfigField[], key: string) {
  return fields.find((field) => configFieldMatchesKey(field, key));
}

export function concreteConfigOptionChoice(field: ConfigField) {
  for (const choice of field.choices) {
    if (choice === null || choice === undefined) {
      continue;
    }
    const value = String(choice).trim();
    const normalized = value.toLowerCase();
    if (value && normalized !== "none" && normalized !== "null") {
      return value;
    }
  }
  return undefined;
}

export function isPresentConfigOptionValue(value: string | undefined) {
  const normalized = (value ?? "").trim().toLowerCase();
  return normalized !== "" && normalized !== "null" && normalized !== "none";
}

function deleteOverrideByKey(overrides: OverrideValues, key: string) {
  const token = configKeyToken(key);
  for (const overrideKey of Object.keys(overrides)) {
    if (configKeyToken(overrideKey) === token) {
      delete overrides[overrideKey];
    }
  }
}

export function normalizeAdaptiveOptionOverrides(
  fields: ConfigField[],
  overrides: OverrideValues,
): OverrideValues {
  const next = { ...overrides };

  for (const { flagKey, optionKey } of ADAPTIVE_OPTION_PAIRS) {
    const flagField = configFieldByKey(fields, flagKey);
    const optionField = configFieldByKey(fields, optionKey);
    if (!flagField || !optionField) {
      continue;
    }

    const flagValue = fieldValue(flagField, next);
    if (isEnabledConfigValue(flagValue)) {
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

function adaptiveFlagKeyForOptionField(fieldKey: string) {
  const fieldToken = configKeyToken(fieldKey);
  return ADAPTIVE_OPTION_PAIRS.find(
    ({ optionKey }) => configKeyToken(optionKey) === fieldToken,
  )?.flagKey;
}

export function isBoundaryProjectorControlField(field: ConfigField) {
  const key = configKeyToken(field.key);
  return (
    key === "input_layer_weight_option" ||
    key === "input_layer_bias_option" ||
    key === "input_layer_diagonal_option" ||
    key === "input_layer_row_mask_option" ||
    key === "output_layer_weight_option" ||
    key === "output_layer_bias_option" ||
    key === "output_layer_diagonal_option" ||
    key === "output_layer_row_mask_option"
  );
}

export function isAdaptiveOptionNoneSuppressed(
  field: ConfigField,
  overrides: OverrideValues,
) {
  if (isBoundaryProjectorControlField(field)) {
    return isPresentConfigOptionValue(fieldValue(field, overrides));
  }

  const flagKey = adaptiveFlagKeyForOptionField(field.key);
  if (!flagKey) {
    return false;
  }
  return isEnabledConfigValue(overrideValue(overrides, flagKey) ?? "");
}

export function configFieldSelectOptions(
  field: ConfigField,
  overrides: OverrideValues,
) {
  const includeNone = field.nullable && !isAdaptiveOptionNoneSuppressed(field, overrides);
  return [
    ...(includeNone ? [{ value: "", label: "None" }] : []),
    ...field.choices.map((choice) => ({
      value: String(choice),
      label: String(choice),
    })),
  ];
}

export function lockedOverrideKeys(
  fields: ConfigField[],
  overrides: OverrideValues,
) {
  const lockedKeys = new Set(
    fields.filter((field) => field.locked).map((field) => configKeyToken(field.key)),
  );
  return Object.keys(overrides)
    .filter((key) => lockedKeys.has(configKeyToken(key)))
    .sort((left, right) => left.localeCompare(right));
}

export function effectivePresetOverrides(
  fields: ConfigField[],
  overrides: OverrideValues,
): OverrideValues {
  const inactiveKeys = new Set(lockedOverrideKeys(fields, overrides));
  if (inactiveKeys.size === 0) {
    return { ...overrides };
  }
  return Object.fromEntries(
    Object.entries(overrides).filter(([key]) => !inactiveKeys.has(key)),
  );
}

export function overrideDigest(overrides: OverrideValues) {
  return Object.entries(overrides)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}=${value}`)
    .join("\u0000");
}

export function activeOverrideScopeLabel(scope: ActiveOverrideScope) {
  return scope === "snapshot" ? "Snapshot draft" : "Preset overrides";
}

export function modifiedCount(fields: ConfigField[], overrides: OverrideValues) {
  return fields.filter((field) => hasOverride(overrides, field.key)).length;
}

export function presetOwnedCount(fields: ConfigField[]) {
  return fields.filter((field) => Boolean(field.locked)).length;
}

export function hasPresetOwnedFields(fields: ConfigField[]) {
  return presetOwnedCount(fields) > 0;
}

function isEnabledConfigValue(value: string) {
  return ["true", "1", "yes", "on"].includes(value.trim().toLowerCase());
}

export function controlledSectionFlagField(section: ConfigSection) {
  const controlFieldKey =
    section.controlFieldKey ?? CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE.get(section.title);
  if (controlFieldKey) {
    return section.fields.find((field) => configFieldMatchesKey(field, controlFieldKey));
  }

  return section.fields.find((field) =>
    FALLBACK_CONTROLLED_SECTION_FLAG_KEYS.has(configKeyToken(field.key)),
  );
}

export function controlledSectionDisabledReason(
  section: ConfigSection,
  controlField: ConfigField,
) {
  return `Enable ${controlField.label} before editing ${section.title}.`;
}

export function isControlledSectionEnabled(
  section: ConfigSection,
  overrides: OverrideValues,
) {
  const controlField = controlledSectionFlagField(section);
  return controlField ? isEnabledConfigValue(fieldValue(controlField, overrides)) : true;
}

export function controlledSectionState(
  section: ConfigSection,
  overrides: OverrideValues,
) {
  const controlField = controlledSectionFlagField(section);
  const isControlled = Boolean(controlField);
  const isEnabled = controlField
    ? isEnabledConfigValue(fieldValue(controlField, overrides))
    : true;

  return {
    controlField,
    isControlled,
    isEnabled,
    disabledReason:
      controlField && !isEnabled
        ? controlledSectionDisabledReason(section, controlField)
        : undefined,
  };
}

export function disabledConfigFieldReasons(
  sections: ConfigSection[],
  overrides: OverrideValues,
) {
  const disabledReasons = new Map<string, string>();

  function collect(section: ConfigSection, inheritedReason?: string) {
    const state = controlledSectionState(section, overrides);
    const ownDisabledReason =
      !state.isEnabled && state.disabledReason ? state.disabledReason : undefined;
    const sectionDisabledReason =
      inheritedReason ?? ownDisabledReason;

    if (sectionDisabledReason) {
      for (const field of section.fields) {
        const isEditableControl =
          !inheritedReason &&
          state.controlField !== undefined &&
          configFieldMatchesKey(field, state.controlField.key);
        if (!isEditableControl) {
          disabledReasons.set(field.key, sectionDisabledReason);
        }
      }
    }

    if (!sectionDisabledReason) {
      const boundaryGroups = boundaryProjectorFieldGroups(
        section.title,
        section.fields,
        overrides,
      );
      for (const group of boundaryGroups ?? []) {
        if (!group.disabledReason || !group.controlField) {
          continue;
        }
        for (const field of group.fields) {
          if (!configFieldMatchesKey(field, group.controlField.key)) {
            disabledReasons.set(field.key, group.disabledReason);
          }
        }
      }
    }

    for (const child of section.children ?? []) {
      collect(child, sectionDisabledReason);
    }
  }

  for (const section of deriveNestedConfigSections(sections)) {
    collect(section);
  }

  return disabledReasons;
}

export function fieldsByKey(sections: ConfigSection[]) {
  const fields = new Map<string, ConfigField>();
  for (const section of sections) {
    for (const field of section.fields) {
      fields.set(field.key, field);
    }
  }
  return fields;
}

export function flattenConfigSearchOptions(
  sections: ConfigSection[],
  overrides: OverrideValues,
) {
  const options: ConfigSearchOption[] = [];
  for (const section of sections) {
    for (const field of section.fields) {
      options.push({
        sectionTitle: section.title,
        field,
        key: field.key,
        label: field.label,
        configKey: field.configKey,
        flag: field.flag,
        type: field.type,
        isModified: hasOverride(overrides, field.key),
        isLocked: Boolean(field.locked),
      });
    }
  }
  return options;
}

function normalizedSearchText(value: string) {
  return value.trim().toLowerCase();
}

export function configSearchOptionMatchesQuery(
  option: Pick<
    ConfigSearchOption,
    "sectionTitle" | "key" | "label" | "configKey" | "flag"
  >,
  query: string,
) {
  const normalizedQuery = normalizedSearchText(query);
  if (!normalizedQuery) {
    return true;
  }

  return [
    option.label,
    option.key,
    option.configKey,
    option.flag,
    option.sectionTitle,
  ].some((value) => value.toLowerCase().includes(normalizedQuery));
}

export function filterConfigSectionsForSearch(
  sections: ConfigSection[],
  { query, selectedFieldKey }: ConfigSearchState,
) {
  const normalizedQuery = normalizedSearchText(query);
  const selectedKey = selectedFieldKey?.trim();
  if (!normalizedQuery && !selectedKey) {
    return sections;
  }

  const matchedSections = sections.reduce<ConfigSection[]>((visibleSections, section) => {
    const fields = section.fields.filter((field) => {
      if (selectedKey) {
        return configKeyToken(field.key) === configKeyToken(selectedKey);
      }

      return configSearchOptionMatchesQuery(
        {
          sectionTitle: section.title,
          key: field.key,
          label: field.label,
          configKey: field.configKey,
          flag: field.flag,
        },
        normalizedQuery,
      );
    });

    if (fields.length > 0) {
      visibleSections.push({ ...section, fields });
    }

    return visibleSections;
  }, []);

  return withAncestorConfigSections(sections, matchedSections);
}

function fieldsWithPrefix(fields: ConfigField[], prefix: string) {
  return fields.filter((field) => configKeyToken(field.key).startsWith(prefix));
}

const STACK_SCOPE_FIELD_SUFFIXES = ["hidden_dim", "layer_norm_position"];

function stackScopedFields(fields: ConfigField[], prefix: string) {
  const stackPrefix = `${prefix}stack_`;
  return fields.filter(
    (field) => {
      const key = configKeyToken(field.key);
      return (
        key.startsWith(stackPrefix) ||
        STACK_SCOPE_FIELD_SUFFIXES.some((suffix) => key === `${prefix}${suffix}`)
      );
    },
  );
}

function sectionWithFields(
  title: string,
  fields: ConfigField[],
  options: Pick<ConfigSection, "children" | "controlFieldKey"> = {},
): ConfigSection | undefined {
  if (fields.length === 0) {
    return undefined;
  }
  return { title, fields, ...options };
}

function boundaryProjectorPrefix(sectionTitle: string) {
  if (sectionTitle === INPUT_BOUNDARY_PROJECTOR_SECTION) {
    return "input_layer_";
  }
  if (sectionTitle === OUTPUT_BOUNDARY_PROJECTOR_SECTION) {
    return "output_layer_";
  }
  return undefined;
}

function boundaryProjectorGroupTitle(prefix: string, fieldKey: string) {
  const key = configKeyToken(fieldKey);
  if (key.startsWith(`${prefix}weight_`)) {
    return "Weight";
  }
  if (key.startsWith(`${prefix}bias_`)) {
    return "Bias";
  }
  if (key.startsWith(`${prefix}diagonal_`)) {
    return "Diagonal";
  }
  if (
    key === `${prefix}row_mask_option` ||
    key.startsWith(`${prefix}mask_`)
  ) {
    return "Mask";
  }
  return undefined;
}

function boundaryProjectorControlFieldKey(prefix: string, groupTitle: string) {
  if (groupTitle === "Weight") {
    return `${prefix}weight_option`;
  }
  if (groupTitle === "Bias") {
    return `${prefix}bias_option`;
  }
  if (groupTitle === "Diagonal") {
    return `${prefix}diagonal_option`;
  }
  if (groupTitle === "Mask") {
    return `${prefix}row_mask_option`;
  }
  return undefined;
}

export function boundaryProjectorFieldGroups(
  sectionTitle: string,
  fields: ConfigField[],
  overrides: OverrideValues = {},
): ConfigFieldGroup[] | undefined {
  const prefix = boundaryProjectorPrefix(sectionTitle);
  if (!prefix) {
    return undefined;
  }

  const groupsByTitle = new Map(
    [
      "Weight",
      "Bias",
      "Diagonal",
      "Mask",
    ].map((title) => [title, [] as ConfigField[]]),
  );

  for (const field of fields) {
    const groupTitle = boundaryProjectorGroupTitle(prefix, field.key);
    if (groupTitle) {
      groupsByTitle.get(groupTitle)?.push(field);
    }
  }

  const groups = Array.from(groupsByTitle, ([title, groupFields]) => {
    const controlFieldKey = boundaryProjectorControlFieldKey(prefix, title);
    const controlField = groupFields.find((field) =>
      configFieldMatchesKey(field, controlFieldKey ?? ""),
    );
    const isEnabled = controlField
      ? isPresentConfigOptionValue(fieldValue(controlField, overrides))
      : true;
    return {
      title,
      fields: groupFields,
      controlField,
      isEnabled,
      disabledReason:
        controlField && !isEnabled
          ? `Select ${controlField.label} before editing ${title.toLowerCase()} boundary settings.`
          : undefined,
    };
  }).filter((group) => group.fields.length > 0);
  const groupedFieldCount = groups.reduce(
    (count, group) => count + group.fields.length,
    0,
  );

  return groupedFieldCount === fields.length ? groups : undefined;
}

function titleFromFieldPrefix(prefix: string) {
  return prefix
    .replace(/_$/, "")
    .split("_")
    .filter(Boolean)
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function titleWithoutOptionsSuffix(title: string) {
  return title.replace(/\s+Options$/i, "");
}

const STACK_CHILD_TITLE_BY_PREFIX = new Map([
  ["gate_", "Gate Stack Options"],
  ["halting_", "Halting Stack Options"],
  ["memory_", "Memory Stack Options"],
  ["recurrent_gate_", "Recurrent Gate Stack Options"],
  ["recurrent_halting_", "Recurrent Halting Stack Options"],
  ["weight_generator_", "Weight Generator Stack Options"],
  ["bias_generator_", "Bias Generator Stack Options"],
  ["diagonal_generator_", "Diagonal Generator Stack Options"],
  ["mask_generator_", "Mask Stack Options"],
]);

function sectionTitleImpliesStackPrefix(section: ConfigSection, prefix: string) {
  const prefixTitle = titleFromFieldPrefix(prefix).toLowerCase();
  const sectionBaseTitle = titleWithoutOptionsSuffix(section.title).toLowerCase();

  return sectionBaseTitle.endsWith("stack") && sectionBaseTitle.includes(prefixTitle);
}

const SECTION_OWNED_STACK_PREFIXES_BY_TITLE = new Map([
  ["Layer Stack Options", new Set(["stack_"])],
  ["Layer Stack Submodule Options", new Set(["submodule_"])],
  ["Adaptive Submodule Stack Options", new Set(["adaptive_submodule_"])],
  ["Gate Stack Options", new Set(["gate_"])],
  ["Halting Stack Options", new Set(["halting_"])],
  ["Memory Stack Options", new Set(["memory_"])],
  ["Recurrent Gate Stack Options", new Set(["recurrent_gate_"])],
  ["Recurrent Halting Stack Options", new Set(["recurrent_halting_"])],
  ["Weight Generator Stack Options", new Set(["weight_generator_"])],
  ["Bias Generator Stack Options", new Set(["bias_generator_"])],
  ["Diagonal Generator Stack Options", new Set(["diagonal_generator_"])],
  ["Mask Stack Options", new Set(["mask_generator_"])],
]);

function sectionTitleOwnsStackPrefix(
  section: ConfigSection,
  prefix: string | undefined,
) {
  if (!prefix) {
    return false;
  }

  return SECTION_OWNED_STACK_PREFIXES_BY_TITLE.get(section.title)?.has(prefix) ?? false;
}

function stackScopedPrefixFromKey(fieldKey: string) {
  const key = configKeyToken(fieldKey);
  if (/^(.+_)stack_/.test(key)) {
    return undefined;
  }

  for (const suffix of STACK_SCOPE_FIELD_SUFFIXES) {
    if (!key.endsWith(`_${suffix}`)) {
      continue;
    }
    return key.slice(0, -suffix.length);
  }
  return undefined;
}

function stackGroupPrefixes(section: ConfigSection) {
  const prefixes: string[] = [];
  const seen = new Set<string>();

  function addPrefix(prefix: string | undefined) {
    if (!prefix || seen.has(prefix)) {
      return;
    }
    prefixes.push(prefix);
    seen.add(prefix);
  }

  for (const field of section.fields) {
    const prefix = /^(.+_)stack_/.exec(configKeyToken(field.key))?.[1];
    if (!sectionTitleOwnsStackPrefix(section, prefix)) {
      addPrefix(prefix);
    }
  }

  for (const field of section.fields) {
    const prefix = stackScopedPrefixFromKey(field.key);
    if (
      prefix &&
      sectionTitleImpliesStackPrefix(section, prefix) &&
      !sectionTitleOwnsStackPrefix(section, prefix)
    ) {
      addPrefix(prefix);
    }
  }

  return prefixes;
}

function stackChildTitle(section: ConfigSection, prefix: string) {
  const explicitTitle = STACK_CHILD_TITLE_BY_PREFIX.get(prefix);
  if (explicitTitle) {
    return explicitTitle;
  }

  const prefixTitle = titleFromFieldPrefix(prefix);
  const sectionBaseTitle = titleWithoutOptionsSuffix(section.title);
  const lowerSectionBaseTitle = sectionBaseTitle.toLowerCase();
  const lowerPrefixTitle = prefixTitle.toLowerCase();

  if (
    lowerSectionBaseTitle.endsWith("stack") &&
    lowerSectionBaseTitle.includes(lowerPrefixTitle)
  ) {
    return sectionBaseTitle;
  }

  return `${prefixTitle} Stack Options`;
}

function optionalFieldKey(fields: ConfigField[], key: string) {
  return fields.find((field) => configFieldMatchesKey(field, key))?.key;
}

function deriveStackChildren(section: ConfigSection) {
  return stackGroupPrefixes(section)
    .map((prefix) =>
      sectionWithFields(
        stackChildTitle(section, prefix),
        stackScopedFields(section.fields, prefix),
        {
          controlFieldKey: optionalFieldKey(
            section.fields,
            `${prefix}stack_independent_flag`,
          ),
        },
      ),
    )
    .filter((child): child is ConfigSection => Boolean(child));
}

function withDerivedStackChildren(section: ConfigSection) {
  const stackChildren = deriveStackChildren(section);
  if (stackChildren.length === 0) {
    return section;
  }
  return {
    ...section,
    children: [...(section.children ?? []), ...stackChildren],
  };
}

function sectionTitleFromFields(
  fields: ConfigField[],
  parentTitle: string,
  fallbackTitle: string,
) {
  return fields.find((field) => field.section && field.section !== parentTitle)?.section
    ?? fallbackTitle;
}

function deriveRecurrentChildren(section: ConfigSection) {
  const recurrentGateFields = fieldsWithPrefix(section.fields, "recurrent_gate_");
  const recurrentHaltingFields = fieldsWithPrefix(
    section.fields,
    "recurrent_halting_",
  );

  return [
    sectionWithFields(
      sectionTitleFromFields(
        recurrentGateFields,
        section.title,
        "Recurrent Gate Options",
      ),
      recurrentGateFields,
      {
        controlFieldKey: "recurrent_gate_flag",
      },
    ),
    sectionWithFields(
      sectionTitleFromFields(
        recurrentHaltingFields,
        section.title,
        "Recurrent Halting Options",
      ),
      recurrentHaltingFields,
      {
        controlFieldKey: "recurrent_halting_flag",
      },
    ),
  ]
    .filter((child): child is ConfigSection => Boolean(child))
    .map((child) => withDerivedStackChildren(child));
}

const CHILD_SECTION_TITLES_BY_TITLE = new Map([
  ["Gate Options", ["Gate Stack Options"]],
  ["Halting Options", ["Halting Stack Options"]],
  ["Memory Options", ["Memory Stack Options"]],
  [
    RECURRENT_LAYER_CONFIG_SECTION,
    ["Recurrent Gate Options", "Recurrent Halting Options"],
  ],
  ["Recurrent Gate Options", ["Recurrent Gate Stack Options"]],
  ["Recurrent Halting Options", ["Recurrent Halting Stack Options"]],
  ["Weight Generator Options", ["Weight Generator Stack Options"]],
  ["Bias Generator Options", ["Bias Generator Stack Options"]],
  ["Diagonal Generator Options", ["Diagonal Generator Stack Options"]],
  ["Mask Options", ["Mask Stack Options"]],
]);

function parentSectionTitlesByChildTitle() {
  const parentsByChild = new Map<string, string[]>();
  for (const [parentTitle, childTitles] of CHILD_SECTION_TITLES_BY_TITLE) {
    for (const childTitle of childTitles) {
      const parents = parentsByChild.get(childTitle) ?? [];
      parents.push(parentTitle);
      parentsByChild.set(childTitle, parents);
    }
  }
  return parentsByChild;
}

function withAncestorConfigSections(
  sourceSections: ConfigSection[],
  matchedSections: ConfigSection[],
) {
  if (matchedSections.length === 0) {
    return matchedSections;
  }

  const sourceSectionsByTitle = new Map(
    sourceSections.map((section) => [section.title, section]),
  );
  const parentsByChild = parentSectionTitlesByChildTitle();
  const visibleSectionsByTitle = new Map(
    matchedSections.map((section) => [section.title, section]),
  );

  function includeAncestors(sectionTitle: string) {
    for (const parentTitle of parentsByChild.get(sectionTitle) ?? []) {
      if (!sourceSectionsByTitle.has(parentTitle)) {
        continue;
      }
      if (!visibleSectionsByTitle.has(parentTitle)) {
        visibleSectionsByTitle.set(parentTitle, {
          ...sourceSectionsByTitle.get(parentTitle)!,
          fields: [],
        });
      }
      includeAncestors(parentTitle);
    }
  }

  for (const section of matchedSections) {
    includeAncestors(section.title);
  }

  return sourceSections
    .filter((section) => visibleSectionsByTitle.has(section.title))
    .map((section) => visibleSectionsByTitle.get(section.title)!);
}

function withSectionControlField(section: ConfigSection) {
  const controlFieldKey = CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE.get(section.title);
  return controlFieldKey ? { ...section, controlFieldKey } : section;
}

function consumedChildSectionTitles(sectionsByTitle: Map<string, ConfigSection>) {
  const consumed = new Set<string>();
  for (const [parentTitle, childTitles] of CHILD_SECTION_TITLES_BY_TITLE) {
    if (!sectionsByTitle.has(parentTitle)) {
      continue;
    }
    for (const childTitle of childTitles) {
      if (sectionsByTitle.has(childTitle)) {
        consumed.add(childTitle);
      }
    }
  }
  return consumed;
}

function nestedConfigSection(
  section: ConfigSection,
  sectionsByTitle: Map<string, ConfigSection>,
  visiting = new Set<string>(),
): ConfigSection {
  if (visiting.has(section.title)) {
    return withSectionControlField(section);
  }

  const nextVisiting = new Set(visiting);
  nextVisiting.add(section.title);
  const explicitChildren = (CHILD_SECTION_TITLES_BY_TITLE.get(section.title) ?? [])
    .map((title) => sectionsByTitle.get(title))
    .filter((child): child is ConfigSection => Boolean(child))
    .map((child) => nestedConfigSection(child, sectionsByTitle, nextVisiting));

  let result = withSectionControlField(section);
  if (
    section.title === RECURRENT_LAYER_CONFIG_SECTION &&
    explicitChildren.length === 0
  ) {
    const recurrentChildren = deriveRecurrentChildren(result);
    if (recurrentChildren.length > 0) {
      result = { ...result, children: recurrentChildren };
    }
  }

  if (!boundaryProjectorPrefix(result.title)) {
    result = withDerivedStackChildren(result);
  }
  if (explicitChildren.length > 0) {
    result = {
      ...result,
      children: [...(result.children ?? []), ...explicitChildren],
    };
  }
  return result;
}

function deriveNestedConfigSectionsFromFields(sections: ConfigSection[]) {
  const sectionsByTitle = new Map(sections.map((section) => [section.title, section]));
  const consumedChildTitles = consumedChildSectionTitles(sectionsByTitle);
  return sections
    .filter((section) => !consumedChildTitles.has(section.title))
    .map((section) => nestedConfigSection(section, sectionsByTitle));
}

function controlFieldForSection(section: ConfigSection | undefined) {
  if (!section?.controlFieldKey) {
    return undefined;
  }
  return section.fields.find((field) =>
    configFieldMatchesKey(field, section.controlFieldKey ?? ""),
  );
}

function withRequiredControlFields(
  section: ConfigSection,
  sourceSection: ConfigSection | undefined,
): ConfigSection {
  const controlField = controlFieldForSection(section) ?? controlFieldForSection(sourceSection);
  const hasControlField =
    controlField && section.fields.some((field) => field.key === controlField.key);
  const sourceChildrenByTitle = new Map(
    (sourceSection?.children ?? []).map((child) => [child.title, child]),
  );
  const children = (section.children ?? []).map((child) =>
    withRequiredControlFields(child, sourceChildrenByTitle.get(child.title)),
  );
  const fieldsWithControl =
    controlField && !hasControlField ? [controlField, ...section.fields] : section.fields;
  const fieldKeys = new Set(fieldsWithControl.map((field) => field.key));
  const descendantFields = children
    .flatMap((child) => child.fields)
    .filter((field) => !fieldKeys.has(field.key));

  return {
    ...section,
    controlFieldKey: section.controlFieldKey ?? sourceSection?.controlFieldKey,
    fields: [...fieldsWithControl, ...descendantFields],
    children,
  };
}

export function deriveNestedConfigSections(
  sections: ConfigSection[],
  sourceSections?: ConfigSection[],
) {
  const derivedSections = deriveNestedConfigSectionsFromFields(sections);
  if (!sourceSections) {
    return derivedSections;
  }

  const sourceSectionsByTitle = new Map(
    deriveNestedConfigSectionsFromFields(sourceSections).map((section) => [
      section.title,
      section,
    ]),
  );

  return derivedSections.map((section) =>
    withRequiredControlFields(section, sourceSectionsByTitle.get(section.title)),
  );
}

export function sectionElementId(
  index: number,
  title: string,
  prefix = "full-config-section",
) {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${prefix}-${index}-${slug || "section"}`;
}

export function sectionCountLabel(count: number, noun: "field" | "override" | "preset") {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function sectionTitlesFromSignature(signature: string) {
  return signature ? signature.split("\u0000") : [];
}
