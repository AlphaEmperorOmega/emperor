import { type ConfigField } from "@/lib/api";

export type OverrideValues = Record<string, string>;

export type ConfigSection = {
  title: string;
  fields: ConfigField[];
  children?: ConfigSection[];
  controlFieldKey?: string;
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
  ["Recurrent Layer Options", "recurrent_flag"],
  ["Recurrent Gate Stack Options", "recurrent_gate_flag"],
  ["Recurrent Halting Options", "recurrent_halting_flag"],
]);

const FALLBACK_CONTROLLED_SECTION_FLAG_KEYS = new Set([
  "gate_flag",
  "halting_flag",
  "recurrent_flag",
]);

export function fieldValue(field: ConfigField, overrides: OverrideValues) {
  const value = field.locked ? field.lockedValue : (overrides[field.key] ?? field.default);
  return value === null || value === undefined ? "" : String(value);
}

export function defaultLabel(field: ConfigField) {
  return field.default === null || field.default === undefined ? "None" : String(field.default);
}

export function hasOverride(overrides: OverrideValues, key: string) {
  return Object.prototype.hasOwnProperty.call(overrides, key);
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

function stackScopedFields(fields: ConfigField[], prefix: string) {
  const stackPrefix = `${prefix}stack_`;
  return fields.filter(
    (field) =>
      field.key.startsWith(stackPrefix) ||
      STACK_SCOPE_FIELD_SUFFIXES.some((suffix) => field.key === `${prefix}${suffix}`),
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

function deriveHaltingChildren(section: ConfigSection) {
  const haltingStackFields = stackScopedFields(section.fields, "halting_");
  return [
    sectionWithFields("Halting Stack Options", haltingStackFields),
  ].filter((child): child is ConfigSection => Boolean(child));
}

function deriveRecurrentChildren(section: ConfigSection) {
  const recurrentGateFields = fieldsWithPrefix(section.fields, "recurrent_gate_");
  const recurrentHaltingFields = fieldsWithPrefix(
    section.fields,
    "recurrent_halting_",
  );
  const recurrentHaltingStackFields = stackScopedFields(
    recurrentHaltingFields,
    "recurrent_halting_",
  );
  const recurrentHaltingChildren = [
    sectionWithFields("Recurrent Halting Stack Options", recurrentHaltingStackFields),
  ].filter((child): child is ConfigSection => Boolean(child));

  return [
    sectionWithFields("Recurrent Gate Stack Options", recurrentGateFields, {
      controlFieldKey: "recurrent_gate_flag",
    }),
    sectionWithFields("Recurrent Halting Options", recurrentHaltingFields, {
      children: recurrentHaltingChildren,
      controlFieldKey: "recurrent_halting_flag",
    }),
  ].filter((child): child is ConfigSection => Boolean(child));
}

function deriveNestedConfigSectionsFromFields(sections: ConfigSection[]) {
  return sections.map((section) => {
    if (section.title === "Halting Options") {
      return {
        ...section,
        children: deriveHaltingChildren(section),
        controlFieldKey: "halting_flag",
      };
    }

    if (section.title === "Recurrent Layer Options") {
      return {
        ...section,
        children: deriveRecurrentChildren(section),
        controlFieldKey: "recurrent_flag",
      };
    }

    const controlFieldKey = CONTROLLED_SECTION_FLAG_KEYS_BY_TITLE.get(section.title);
    return controlFieldKey ? { ...section, controlFieldKey } : section;
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
