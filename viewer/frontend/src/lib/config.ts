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

const CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE = new Map([
  ["Gate Stack Options", "gate_flag"],
  ["Halting Options", "halting_flag"],
  ["Memory Options", "memory_flag"],
  ["Recurrent Layer Options", "recurrent_flag"],
  ["Recurrent Gate Stack Options", "recurrent_gate_flag"],
  ["Recurrent Halting Options", "recurrent_halting_flag"],
  ["Weight Options", "weight_option_flag"],
  ["Bias Options", "bias_option_flag"],
  ["Diagonal Options", "diagonal_option_flag"],
  ["Mask Options", "mask_option_flag"],
  ["Input Boundary Projector Options", "input_layer_adaptive_flag"],
  ["Output Boundary Projector Options", "output_layer_adaptive_flag"],
  ["Weight Generator Stack Options", "weight_generator_stack_independent_flag"],
  ["Bias Generator Stack Options", "bias_generator_stack_independent_flag"],
  ["Diagonal Generator Stack Options", "diagonal_generator_stack_independent_flag"],
  ["Mask Generator Stack Options", "mask_generator_stack_independent_flag"],
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
const RECURRENT_LAYER_FIELD_PREFIXES = [
  "recurrent_gate_",
  "recurrent_halting_",
];

export function displayConfigFieldSection(field: ConfigField) {
  if (RECURRENT_LAYER_FIELD_PREFIXES.some((prefix) => field.key.startsWith(prefix))) {
    return RECURRENT_LAYER_CONFIG_SECTION;
  }
  return field.section || GENERAL_CONFIG_SECTION;
}

export function normalizeConfigFieldForDisplay(field: ConfigField): ConfigField {
  const section = displayConfigFieldSection(field);
  return field.section === section ? field : { ...field, section };
}

export function fieldValue(field: ConfigField, overrides: OverrideValues) {
  const value = field.locked ? field.lockedValue : (overrides[field.key] ?? field.default);
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
  return Object.prototype.hasOwnProperty.call(overrides, key);
}

export function lockedOverrideKeys(
  fields: ConfigField[],
  overrides: OverrideValues,
) {
  const lockedKeys = new Set(
    fields.filter((field) => field.locked).map((field) => field.key),
  );
  return Object.keys(overrides)
    .filter((key) => lockedKeys.has(key))
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
    return section.fields.find((field) => field.key === controlFieldKey);
  }

  return section.fields.find((field) =>
    FALLBACK_CONTROLLED_SECTION_FLAG_KEYS.has(field.key),
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
        const isEditableControl = !inheritedReason && field.key === state.controlField?.key;
        if (!isEditableControl) {
          disabledReasons.set(field.key, sectionDisabledReason);
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

  return sections.reduce<ConfigSection[]>((visibleSections, section) => {
    const fields = section.fields.filter((field) => {
      if (selectedKey) {
        return field.key === selectedKey;
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
}

function fieldsWithPrefix(fields: ConfigField[], prefix: string) {
  return fields.filter((field) => field.key.startsWith(prefix));
}

const STACK_SCOPE_FIELD_SUFFIXES = ["hidden_dim", "layer_norm_position"];
const STACK_SCOPE_FLAG_SUFFIXES = ["bias_flag"];

function stackScopedFields(fields: ConfigField[], prefix: string) {
  const stackPrefix = `${prefix}stack_`;
  return fields.filter(
    (field) =>
      field.key.startsWith(stackPrefix) ||
      STACK_SCOPE_FIELD_SUFFIXES.some((suffix) => field.key === `${prefix}${suffix}`) ||
      STACK_SCOPE_FLAG_SUFFIXES.some((suffix) => field.key === `${prefix}${suffix}`),
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
  if (fieldKey.startsWith(`${prefix}weight_`)) {
    return "Weight";
  }
  if (fieldKey.startsWith(`${prefix}bias_`)) {
    return "Bias";
  }
  if (fieldKey.startsWith(`${prefix}diagonal_`)) {
    return "Diagonal";
  }
  if (
    fieldKey === `${prefix}row_mask_option` ||
    fieldKey.startsWith(`${prefix}mask_`)
  ) {
    return "Mask";
  }
  if (fieldKey.startsWith(`${prefix}adaptive_generator_stack_`)) {
    return "Adaptive Generator Stack";
  }
  return undefined;
}

export function boundaryProjectorFieldGroups(
  sectionTitle: string,
  fields: ConfigField[],
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
      "Adaptive Generator Stack",
    ].map((title) => [title, [] as ConfigField[]]),
  );

  for (const field of fields) {
    const groupTitle = boundaryProjectorGroupTitle(prefix, field.key);
    if (groupTitle) {
      groupsByTitle.get(groupTitle)?.push(field);
    }
  }

  const groups = Array.from(groupsByTitle, ([title, groupFields]) => ({
    title,
    fields: groupFields,
  })).filter((group) => group.fields.length > 0);
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
  ["gate_", "Gate Model Stack"],
  ["recurrent_gate_", "Recurrent Gate Model Stack"],
]);

function sectionTitleImpliesStackPrefix(section: ConfigSection, prefix: string) {
  const prefixTitle = titleFromFieldPrefix(prefix).toLowerCase();
  const sectionBaseTitle = titleWithoutOptionsSuffix(section.title).toLowerCase();

  return sectionBaseTitle.endsWith("stack") && sectionBaseTitle.includes(prefixTitle);
}

function stackScopedPrefixFromKey(fieldKey: string) {
  const suffixes = [...STACK_SCOPE_FIELD_SUFFIXES, ...STACK_SCOPE_FLAG_SUFFIXES];
  for (const suffix of suffixes) {
    if (!fieldKey.endsWith(`_${suffix}`)) {
      continue;
    }
    return fieldKey.slice(0, -suffix.length);
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
    addPrefix(/^(.+_)stack_/.exec(field.key)?.[1]);
  }

  for (const field of section.fields) {
    const prefix = stackScopedPrefixFromKey(field.key);
    if (prefix && sectionTitleImpliesStackPrefix(section, prefix)) {
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
  return fields.some((field) => field.key === key) ? key : undefined;
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

function deriveRecurrentChildren(section: ConfigSection) {
  const recurrentGateFields = fieldsWithPrefix(section.fields, "recurrent_gate_");
  const recurrentHaltingFields = fieldsWithPrefix(
    section.fields,
    "recurrent_halting_",
  );

  return [
    sectionWithFields("Recurrent Gate Stack Options", recurrentGateFields, {
      controlFieldKey: "recurrent_gate_flag",
    }),
    sectionWithFields("Recurrent Halting Options", recurrentHaltingFields, {
      controlFieldKey: "recurrent_halting_flag",
    }),
  ]
    .filter((child): child is ConfigSection => Boolean(child))
    .map((child) => withDerivedStackChildren(child));
}

function deriveNestedConfigSectionsFromFields(sections: ConfigSection[]) {
  return sections.map((section) => {
    if (section.title === "Recurrent Layer Options") {
      return {
        ...section,
        children: deriveRecurrentChildren(section),
        controlFieldKey: "recurrent_flag",
      };
    }

    const controlFieldKey = CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE.get(section.title);
    if (boundaryProjectorPrefix(section.title)) {
      return controlFieldKey ? { ...section, controlFieldKey } : section;
    }

    return withDerivedStackChildren(
      controlFieldKey ? { ...section, controlFieldKey } : section,
    );
  });
}

function controlFieldForSection(section: ConfigSection | undefined) {
  if (!section?.controlFieldKey) {
    return undefined;
  }
  return section.fields.find((field) => field.key === section.controlFieldKey);
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
