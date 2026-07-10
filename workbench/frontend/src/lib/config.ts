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
  stackHint?: {
    label: string;
    title: string;
    sourceTitle: string;
  };
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

export type InheritedStackSectionHint = {
  label: string;
  title: string;
  sourceTitle: string;
  isCustom: boolean;
};

export type InheritedConfigFieldHint = {
  label: string;
  title: string;
  sourceTitle: string;
  sourceField: ConfigField;
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
  ["Attention Mode", "expert_attention_flag"],
  ["Gate Options", "gate_flag"],
  ["Gate Stack Options", "gate_stack_independent_flag"],
  ["Halting Options", "halting_flag"],
  ["Halting Stack Options", "halting_stack_independent_flag"],
  ["Memory Options", "memory_flag"],
  ["Memory Stack Options", "memory_stack_independent_flag"],
  ["Router Gate Options", "router_gate_flag"],
  ["Router Gate Stack Options", "router_gate_stack_independent_flag"],
  ["Router Halting Options", "router_halting_flag"],
  ["Router Halting Stack Options", "router_halting_stack_independent_flag"],
  ["Router Memory Options", "router_memory_flag"],
  ["Router Memory Stack Options", "router_memory_stack_independent_flag"],
  ["Router Recurrent Layer Options", "router_recurrent_flag"],
  ["Router Recurrent Gate Options", "router_recurrent_gate_flag"],
  [
    "Router Recurrent Gate Stack Options",
    "router_recurrent_gate_stack_independent_flag",
  ],
  ["Router Recurrent Halting Options", "router_recurrent_halting_flag"],
  [
    "Router Recurrent Halting Stack Options",
    "router_recurrent_halting_stack_independent_flag",
  ],
  ["Expert Gate Options", "expert_gate_flag"],
  ["Expert Gate Stack Options", "expert_gate_stack_independent_flag"],
  ["Expert Halting Options", "expert_halting_flag"],
  ["Expert Halting Stack Options", "expert_halting_stack_independent_flag"],
  ["Expert Memory Options", "expert_memory_flag"],
  ["Expert Memory Stack Options", "expert_memory_stack_independent_flag"],
  ["Expert Recurrent Layer Options", "expert_recurrent_flag"],
  ["Expert Recurrent Gate Options", "expert_recurrent_gate_flag"],
  [
    "Expert Recurrent Gate Stack Options",
    "expert_recurrent_gate_stack_independent_flag",
  ],
  ["Expert Recurrent Halting Options", "expert_recurrent_halting_flag"],
  [
    "Expert Recurrent Halting Stack Options",
    "expert_recurrent_halting_stack_independent_flag",
  ],
  ["Feed-Forward Gate Options", "ff_gate_flag"],
  ["Feed-Forward Gate Stack Options", "ff_gate_stack_independent_flag"],
  ["Feed-Forward Halting Options", "ff_halting_flag"],
  ["Feed-Forward Halting Stack Options", "ff_halting_stack_independent_flag"],
  ["Feed-Forward Memory Options", "ff_memory_flag"],
  ["Feed-Forward Memory Stack Options", "ff_memory_stack_independent_flag"],
  ["Feed-Forward Recurrent Layer Options", "ff_recurrent_flag"],
  ["Feed-Forward Recurrent Gate Options", "ff_recurrent_gate_flag"],
  [
    "Feed-Forward Recurrent Gate Stack Options",
    "ff_recurrent_gate_stack_independent_flag",
  ],
  ["Feed-Forward Recurrent Halting Options", "ff_recurrent_halting_flag"],
  [
    "Feed-Forward Recurrent Halting Stack Options",
    "ff_recurrent_halting_stack_independent_flag",
  ],
  ["Attention Projection Gate Options", "attn_gate_flag"],
  [
    "Attention Projection Gate Stack Options",
    "attn_gate_stack_independent_flag",
  ],
  ["Attention Projection Halting Options", "attn_halting_flag"],
  [
    "Attention Projection Halting Stack Options",
    "attn_halting_stack_independent_flag",
  ],
  ["Attention Projection Memory Options", "attn_memory_flag"],
  [
    "Attention Projection Memory Stack Options",
    "attn_memory_stack_independent_flag",
  ],
  ["Attention Projection Recurrent Layer Options", "attn_recurrent_flag"],
  [
    "Attention Projection Recurrent Gate Options",
    "attn_recurrent_gate_flag",
  ],
  [
    "Attention Projection Recurrent Gate Stack Options",
    "attn_recurrent_gate_stack_independent_flag",
  ],
  [
    "Attention Projection Recurrent Halting Options",
    "attn_recurrent_halting_flag",
  ],
  [
    "Attention Projection Recurrent Halting Stack Options",
    "attn_recurrent_halting_stack_independent_flag",
  ],
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
  ["Router Weight Generator Options", "router_weight_option_flag"],
  [
    "Router Weight Generator Stack Options",
    "router_weight_generator_stack_independent_flag",
  ],
  ["Router Bias Generator Options", "router_bias_option_flag"],
  [
    "Router Bias Generator Stack Options",
    "router_bias_generator_stack_independent_flag",
  ],
  ["Router Diagonal Generator Options", "router_diagonal_option_flag"],
  [
    "Router Diagonal Generator Stack Options",
    "router_diagonal_generator_stack_independent_flag",
  ],
  ["Router Mask Options", "router_mask_option_flag"],
  [
    "Router Mask Stack Options",
    "router_mask_generator_stack_independent_flag",
  ],
]);

const FALLBACK_CONTROLLED_SECTION_FLAG_KEYS = new Set([
  "gate_flag",
  "halting_flag",
  "memory_flag",
  "recurrent_flag",
]);

const INHERITED_STACK_SECTIONS_BY_TITLE = new Map([
  [
    "Gate Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Halting Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Memory Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Router Gate Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Router Halting Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Router Memory Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Router Recurrent Gate Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Router Recurrent Halting Stack Options",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "Expert Gate Stack Options",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "Expert Halting Stack Options",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "Expert Memory Stack Options",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "Expert Recurrent Gate Stack Options",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "Expert Recurrent Halting Stack Options",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "Feed-Forward Gate Stack Options",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "Feed-Forward Halting Stack Options",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "Feed-Forward Memory Stack Options",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "Feed-Forward Recurrent Gate Stack Options",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "Feed-Forward Recurrent Halting Stack Options",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "Attention Projection Gate Stack Options",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "Attention Projection Halting Stack Options",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "Attention Projection Memory Stack Options",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "Attention Projection Recurrent Gate Stack Options",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "Attention Projection Recurrent Halting Stack Options",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "Recurrent Gate Stack Options",
    {
      sourceTitle: "Gate Stack Options",
      inheritedLabel: "Inherits Gate Stack",
    },
  ],
  [
    "Recurrent Halting Stack Options",
    {
      sourceTitle: "Halting Stack Options",
      inheritedLabel: "Inherits Halting Stack",
    },
  ],
  [
    "Weight Generator Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "Bias Generator Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "Diagonal Generator Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "Mask Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "Router Weight Generator Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "Router Bias Generator Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "Router Diagonal Generator Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "Router Mask Stack Options",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
]);

const GENERAL_CONFIG_SECTION = "General";
const GLOBAL_CONFIG_SECTION = "Global";
const LAYER_STACK_OPTIONS_SECTION = "Layer Stack Options";
const LAYER_STACK_SUBMODULE_OPTIONS_SECTION = "Layer Stack Submodule Options";
const INPUT_BOUNDARY_MODEL_SECTION = "Input Boundary Model Options";
const OUTPUT_BOUNDARY_MODEL_SECTION = "Output Boundary Model Options";
const ADAPTIVE_GENERATOR_STACK_SECTION = "Adaptive Generator Stack Options";
const MIXTURE_OF_EXPERTS_SECTION = "Mixture Of Experts Model Options";
const ATTENTION_MODE_SECTION = "Attention Mode";

const DISPLAY_SECTION_PATH_OVERRIDES_BY_FIELD_KEY = new Map<string, string[]>([
  [
    "expert_attention_flag",
    [MIXTURE_OF_EXPERTS_SECTION, ATTENTION_MODE_SECTION],
  ],
  [
    "expert_attention_use_kv_expert_models_flag",
    [MIXTURE_OF_EXPERTS_SECTION, ATTENTION_MODE_SECTION],
  ],
]);

const DISPLAY_CONFIG_FIELD_LABELS_BY_KEY = new Map<string, string>([
  ["expert_attention_flag", "Use MixtureOfAttentionHeads"],
  ["expert_attention_use_kv_expert_models_flag", "Expert K/V projections"],
]);

const DISPLAY_CONFIG_SECTIONS = new Map<
  string,
  { title: string; description: string }
>([
  [
    LAYER_STACK_OPTIONS_SECTION,
    {
      title: "Layer Hidden Stack Options",
      description: "Hidden stack inside each layer",
    },
  ],
  [
    LAYER_STACK_SUBMODULE_OPTIONS_SECTION,
    {
      title: "Shared Submodule Stack Defaults",
      description:
        "Defaults for gate, halting, memory, and recurrent stacks unless overridden.",
    },
  ],
  [
    ATTENTION_MODE_SECTION,
    {
      title: "Attention Mode",
      description:
        "Off uses SelfAttention. On uses MixtureOfAttentionHeads for expert-routed attention projections.",
    },
  ],
]);

export function displayConfigFieldSection(field: ConfigField) {
  return field.section || GENERAL_CONFIG_SECTION;
}

export function displayConfigSectionTitle(title: string) {
  return DISPLAY_CONFIG_SECTIONS.get(title)?.title ?? title;
}

export function displayConfigSectionDescription(title: string) {
  return DISPLAY_CONFIG_SECTIONS.get(title)?.description;
}

function displaySectionPathOverride(field: ConfigField) {
  return DISPLAY_SECTION_PATH_OVERRIDES_BY_FIELD_KEY.get(configKeyToken(field.key));
}

function sentenceCaseLabel(label: string) {
  const trimmed = label.trim().replace(/\s+/g, " ");
  return trimmed ? `${trimmed[0]!.toUpperCase()}${trimmed.slice(1)}` : trimmed;
}

function sectionTitleLabelPrefixes(sectionTitle: string) {
  const prefix = sectionTitle
    .trim()
    .replace(/\s+options$/i, "")
    .toLowerCase();
  const abbreviatedPrefix = prefix
    .replace(/\bfeed-forward\b/g, "ff")
    .replace(/\battention projection\b/g, "attn");

  return [
    prefix,
    abbreviatedPrefix,
    prefix.replace(/\blayer\b\s*/g, "").trim(),
    abbreviatedPrefix.replace(/\blayer\b\s*/g, "").trim(),
  ].filter((candidate, index, candidates) =>
    candidate.length > 0 && candidates.indexOf(candidate) === index
  );
}

function stackFieldLabelPrefix(fieldKey: string) {
  const key = configKeyToken(fieldKey);
  if (key.startsWith("stack_")) {
    return "stack";
  }
  const stackIndex = key.indexOf("_stack_");
  if (stackIndex === -1) {
    return undefined;
  }
  return key
    .slice(0, stackIndex + "_stack".length)
    .replace(/_/g, " ");
}

function stackSectionLabelPrefix(sectionTitle: string) {
  if (sectionTitle === LAYER_STACK_OPTIONS_SECTION) {
    return "stack";
  }
  if (sectionTitle === LAYER_STACK_SUBMODULE_OPTIONS_SECTION) {
    return "submodule stack";
  }
  return undefined;
}

export function displayConfigFieldLabel(
  field: ConfigField,
  sectionTitle: string,
  groupTitle?: string,
) {
  const explicitLabel = DISPLAY_CONFIG_FIELD_LABELS_BY_KEY.get(
    configKeyToken(field.key),
  );
  if (explicitLabel) {
    return explicitLabel;
  }

  const controlFieldKey = CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE.get(sectionTitle);
  if (
    controlFieldKey &&
    configFieldMatchesKey(field, controlFieldKey)
  ) {
    return configKeyToken(field.key).endsWith("_stack_independent_flag")
      ? "Use custom stack"
      : "Enabled";
  }

  const label = field.label.trim().replace(/\s+/g, " ");
  const normalizedLabel = label.toLowerCase();
  const prefixCandidates = [
    stackFieldLabelPrefix(field.key),
    stackSectionLabelPrefix(sectionTitle),
    ...sectionTitleLabelPrefixes(sectionTitle),
    groupTitle ? groupTitle.toLowerCase() : undefined,
  ]
    .filter((prefix): prefix is string => Boolean(prefix))
    .sort((left, right) => right.length - left.length);

  for (const prefix of prefixCandidates) {
    if (normalizedLabel.startsWith(`${prefix} `)) {
      return sentenceCaseLabel(label.slice(prefix.length + 1));
    }
  }

  return sentenceCaseLabel(label);
}

export function normalizeConfigFieldForDisplay(field: ConfigField): ConfigField {
  const section = displayConfigFieldSection(field);
  return field.section === section ? field : { ...field, section };
}

function normalizedFieldSectionPath(field: ConfigField) {
  const sectionPath = field.sectionPath.filter(
    (section): section is string => typeof section === "string" && section.length > 0,
  );
  if (sectionPath.length === 0) {
    throw new Error(`Config field ${field.key} is missing sectionPath metadata.`);
  }
  return sectionPath;
}

export function configSectionFields(section: ConfigSection): ConfigField[] {
  return [
    ...section.fields,
    ...(section.children ?? []).flatMap((child) => configSectionFields(child)),
  ];
}

export function configSectionsFields(sections: ConfigSection[]): ConfigField[] {
  return sections.flatMap((section) => configSectionFields(section));
}

export function configSectionCount(sections: ConfigSection[]): number {
  return sections.reduce(
    (count, section) =>
      count + 1 + configSectionCount(section.children ?? []),
    0,
  );
}

export function groupConfigFieldsBySectionPath(fields: ConfigField[]): ConfigSection[] {
  type DraftSection = {
    title: string;
    fields: ConfigField[];
    children: DraftSection[];
  };

  const roots: DraftSection[] = [];

  function childFor(parent: DraftSection[] | DraftSection, title: string) {
    const children = Array.isArray(parent) ? parent : parent.children;
    let child = children.find((candidate) => candidate.title === title);
    if (!child) {
      child = { title, fields: [], children: [] };
      children.push(child);
    }
    return child;
  }

  for (const rawField of fields) {
    const normalizedField = normalizeConfigFieldForDisplay(rawField);
    const overriddenSectionPath = displaySectionPathOverride(normalizedField);
    const field = overriddenSectionPath
      ? {
          ...normalizedField,
          section: overriddenSectionPath[overriddenSectionPath.length - 1]!,
          sectionPath: overriddenSectionPath,
        }
      : normalizedField;
    const sectionPath = normalizedFieldSectionPath(field);
    let node = childFor(roots, sectionPath[0]!);
    for (const sectionTitle of sectionPath.slice(1)) {
      node = childFor(node, sectionTitle);
    }
    node.fields.push(field);
  }

  function finalize(section: DraftSection): ConfigSection {
    return {
      title: section.title,
      fields: section.fields,
      ...(section.children.length > 0
        ? { children: section.children.map(finalize) }
        : {}),
    };
  }

  return roots.map(finalize);
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
  { flagKey: "router_weight_option_flag", optionKey: "router_weight_option" },
  { flagKey: "router_bias_option_flag", optionKey: "router_bias_option" },
  { flagKey: "router_diagonal_option_flag", optionKey: "router_diagonal_option" },
  { flagKey: "router_mask_option_flag", optionKey: "router_row_mask_option" },
] as const;

function configFieldByKey(fields: ConfigField[], key: string) {
  return fields.find((field) => configFieldMatchesKey(field, key));
}

export function inheritedHiddenModelFieldHint(
  section: Pick<ConfigSection, "title" | "fields">,
  allFields: ConfigField[],
): InheritedConfigFieldHint | undefined {
  if (section.title !== LAYER_STACK_OPTIONS_SECTION) {
    return undefined;
  }
  if (configFieldByKey(section.fields, "hidden_dim")) {
    return undefined;
  }
  const sourceField = configFieldByKey(allFields, "hidden_dim");
  if (!sourceField) {
    return undefined;
  }
  return {
    label: sourceField.label,
    title: `${sourceField.label} comes from ${GLOBAL_CONFIG_SECTION} / ${sourceField.configKey}.`,
    sourceTitle: GLOBAL_CONFIG_SECTION,
    sourceField,
  };
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

export function isBoundaryModelControlField(field: ConfigField) {
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
  if (isBoundaryModelControlField(field)) {
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

export function controlledSectionFlagField(
  section: Pick<ConfigSection, "title" | "fields" | "controlFieldKey">,
) {
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

export function inheritedStackSectionHint(
  section: Pick<ConfigSection, "title" | "fields" | "controlFieldKey">,
  overrides: OverrideValues,
): InheritedStackSectionHint | undefined {
  const inheritance = INHERITED_STACK_SECTIONS_BY_TITLE.get(section.title);
  if (!inheritance) {
    return undefined;
  }

  const controlField = controlledSectionFlagField(section);
  if (!controlField) {
    return undefined;
  }

  const isCustom = isEnabledConfigValue(fieldValue(controlField, overrides));
  const title = isCustom
    ? `Uses ${section.title} values while ${controlField.label} is on. Disable it to inherit ${inheritance.sourceTitle}.`
    : `Uses ${inheritance.sourceTitle} while ${controlField.label} is off. Enable it to use ${section.title} values.`;

  return {
    label: isCustom ? "Custom Stack" : inheritance.inheritedLabel,
    title,
    sourceTitle: inheritance.sourceTitle,
    isCustom,
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
    const shouldPropagateOwnDisabledReason =
      !state.controlField ||
      !configKeyToken(state.controlField.key).endsWith("_stack_independent_flag");

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
      const boundaryGroups = boundaryModelFieldGroups(
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
      collect(
        child,
        inheritedReason ??
          (shouldPropagateOwnDisabledReason ? ownDisabledReason : undefined),
      );
    }
  }

  for (const section of deriveNestedConfigSections(sections)) {
    collect(section);
  }

  return disabledReasons;
}

export function fieldsByKey(sections: ConfigSection[]) {
  const fields = new Map<string, ConfigField>();
  for (const field of configSectionsFields(sections)) {
    fields.set(field.key, field);
  }
  return fields;
}

export function flattenConfigSearchOptions(
  sections: ConfigSection[],
  overrides: OverrideValues,
) {
  const options: ConfigSearchOption[] = [];
  function collect(section: ConfigSection) {
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
    for (const child of section.children ?? []) {
      collect(child);
    }
  }
  for (const section of sections) {
    collect(section);
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

  function filterSection(section: ConfigSection): ConfigSection | undefined {
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
    const children = (section.children ?? [])
      .map(filterSection)
      .filter((child): child is ConfigSection => Boolean(child));
    if (fields.length === 0 && children.length === 0) {
      return undefined;
    }
    return {
      ...section,
      fields,
      ...(children.length > 0 ? { children } : { children: undefined }),
    };
  }

  return sections
    .map(filterSection)
    .filter((section): section is ConfigSection => Boolean(section));
}

function boundaryModelPrefix(sectionTitle: string) {
  if (sectionTitle === INPUT_BOUNDARY_MODEL_SECTION) {
    return "input_layer_";
  }
  if (sectionTitle === OUTPUT_BOUNDARY_MODEL_SECTION) {
    return "output_layer_";
  }
  return undefined;
}

function boundaryModelGroupTitle(prefix: string, fieldKey: string) {
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

function boundaryModelControlFieldKey(prefix: string, groupTitle: string) {
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

function boundaryModelStackHint(): NonNullable<ConfigFieldGroup["stackHint"]> {
  return {
    label: "Uses Adaptive Stack",
    title: `Boundary generator stack settings come from ${ADAPTIVE_GENERATOR_STACK_SECTION}.`,
    sourceTitle: ADAPTIVE_GENERATOR_STACK_SECTION,
  };
}

export function boundaryModelFieldGroups(
  sectionTitle: string,
  fields: ConfigField[],
  overrides: OverrideValues = {},
): ConfigFieldGroup[] | undefined {
  const prefix = boundaryModelPrefix(sectionTitle);
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
    const groupTitle = boundaryModelGroupTitle(prefix, field.key);
    if (groupTitle) {
      groupsByTitle.get(groupTitle)?.push(field);
    }
  }

  const groups = Array.from(groupsByTitle, ([title, groupFields]) => {
    const controlFieldKey = boundaryModelControlFieldKey(prefix, title);
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
      stackHint: boundaryModelStackHint(),
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

function withSectionControlField(section: ConfigSection): ConfigSection {
  const controlFieldKey = CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE.get(section.title);
  const children = section.children?.map(withSectionControlField);
  return {
    ...section,
    ...(controlFieldKey ? { controlFieldKey } : {}),
    ...(children ? { children } : {}),
  };
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

  return {
    ...section,
    controlFieldKey: section.controlFieldKey ?? sourceSection?.controlFieldKey,
    fields: fieldsWithControl,
    children,
  };
}

export function deriveNestedConfigSections(
  sections: ConfigSection[],
  sourceSections?: ConfigSection[],
) {
  const derivedSections = groupConfigFieldsBySectionPath(
    configSectionsFields(sections),
  ).map(withSectionControlField);
  if (!sourceSections) {
    return derivedSections;
  }

  const sourceSectionsByTitle = new Map(
    groupConfigFieldsBySectionPath(configSectionsFields(sourceSections))
      .map(withSectionControlField)
      .map((section) => [
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
