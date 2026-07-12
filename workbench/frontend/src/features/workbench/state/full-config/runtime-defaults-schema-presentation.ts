import { type ConfigField } from "@/lib/api";
import {
  type ConfigSearchState,
  type ConfigSection,
  type InheritedStackSectionHint,
  type OverrideValues,
  boundaryModelFieldGroups,
  configFieldSelectOptions,
  configSearchOptionMatchesQuery,
  configSectionFields,
  configSectionsFields,
  concreteConfigOptionChoice,
  controlledSectionState,
  deriveNestedConfigSections,
  disabledConfigFieldReasons,
  displayConfigFieldLabel,
  displayConfigSectionDescription,
  displayConfigSectionTitle,
  fieldValue,
  filterConfigSectionsForSearch,
  flattenConfigSearchOptions,
  hasOverride,
  inheritedHiddenModelFieldHint,
  inheritedStackSectionHint,
  modifiedCount,
  presetOwnedCount,
  sectionElementId,
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

export type RuntimeDefaultsFieldPresentation = {
  schema: ConfigField;
  key: string;
  label: string;
  value: string;
  selectOptions: Array<{ value: string; label: string }>;
  isModified: boolean;
  isPresetOwned: boolean;
  isEnabledValue: boolean;
  modeLabel?: string;
  disabledReason?: string;
};

export type RuntimeDefaultsInheritedFieldPresentation = {
  label: string;
  title: string;
  sourceTitle: string;
  field: RuntimeDefaultsFieldPresentation;
};

export type RuntimeDefaultsFieldGroupPresentation = {
  id: string;
  title: string;
  fields: RuntimeDefaultsFieldPresentation[];
  controlField?: RuntimeDefaultsFieldPresentation;
  isEnabled: boolean;
  isSwitchDisabled: boolean;
  firstConcreteOption?: string;
  disabledReason?: string;
  stackHint?: {
    label: string;
    title: string;
    sourceTitle: string;
  };
  metrics: RuntimeDefaultsMetrics;
};

export type RuntimeDefaultsSectionPresentation = {
  id: string;
  title: string;
  displayTitle: string;
  displayDescription?: string;
  fields: RuntimeDefaultsFieldPresentation[];
  bodyFields: RuntimeDefaultsFieldPresentation[];
  fieldGroups?: RuntimeDefaultsFieldGroupPresentation[];
  children: RuntimeDefaultsSectionPresentation[];
  controlField?: RuntimeDefaultsFieldPresentation;
  disabledReason?: string;
  isDisabled: boolean;
  directMetrics: RuntimeDefaultsMetrics;
  treeMetrics: RuntimeDefaultsMetrics;
  stackInheritanceHint?: InheritedStackSectionHint;
  inheritedField?: RuntimeDefaultsInheritedFieldPresentation;
  childTitleSignature: string;
  groupTitleSignature: string;
  enabledGroupTitleSignature: string;
};

export type RuntimeDefaultsSearchOptionPresentation = {
  sectionTitle: string;
  rootSectionTitle: string;
  field: RuntimeDefaultsFieldPresentation;
  key: string;
  label: string;
  configKey: string;
  flag: string;
  type: string;
};

export type RuntimeDefaultsSchemaPresentation = {
  schemaFields: ConfigField[];
  fieldCount: number;
  presetOwnedFieldCount: number;
  sections: RuntimeDefaultsSectionPresentation[];
  defaultOpenSectionTitles: string[];
  searchOpenKey: string;
  isSearchActive: boolean;
  search: {
    options: RuntimeDefaultsSearchOptionPresentation[];
    matchesQuery: (
      option: RuntimeDefaultsSearchOptionPresentation,
      query: string,
    ) => boolean;
  };
};

function isEnabledConfigValue(value: string) {
  return ["true", "1", "yes", "on"].includes(value.trim().toLowerCase());
}

function attentionModeValueLabel(field: ConfigField, value: string) {
  if (field.key.trim().replace(/-/g, "_").toLowerCase() !== "expert_attention_flag") {
    return undefined;
  }
  return isEnabledConfigValue(value) ? "MixtureOfAttentionHeads" : "SelfAttention";
}

function metricsFor(fields: ConfigField[], overrides: OverrideValues): RuntimeDefaultsMetrics {
  const overrideCount = modifiedCount(fields, overrides);
  const presetCount = presetOwnedCount(fields);
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

function presentField(
  field: ConfigField,
  overrides: OverrideValues,
  label: string,
  disabledReason?: string,
): RuntimeDefaultsFieldPresentation {
  const value = fieldValue(field, overrides);
  return {
    schema: field,
    key: field.key,
    label,
    value,
    selectOptions: configFieldSelectOptions(field, overrides),
    isModified: hasOverride(overrides, field.key),
    isPresetOwned: field.locked === true,
    isEnabledValue: isEnabledConfigValue(value),
    modeLabel: attentionModeValueLabel(field, value),
    disabledReason,
  };
}

function sectionMap(sections: ConfigSection[]) {
  const byTitle = new Map<string, ConfigSection>();
  function collect(section: ConfigSection) {
    byTitle.set(section.title, section);
    for (const child of section.children ?? []) {
      collect(child);
    }
  }
  for (const section of sections) {
    collect(section);
  }
  return byTitle;
}

function rootSectionTitles(sections: ConfigSection[]) {
  const bySectionTitle = new Map<string, string>();
  function collect(section: ConfigSection, rootTitle: string) {
    bySectionTitle.set(section.title, rootTitle);
    for (const child of section.children ?? []) {
      collect(child, rootTitle);
    }
  }
  for (const section of sections) {
    collect(section, section.title);
  }
  return bySectionTitle;
}

function descendantFieldKeys(sections: ConfigSection[]) {
  return new Set(sections.flatMap((section) => configSectionFields(section).map((field) => field.key)));
}

function presentSection({
  section,
  sourceSectionsByTitle,
  allFields,
  disabledFieldReasonByKey,
  overrides,
  showInheritedFields,
  inheritedDisabledReason,
  index,
  idPrefix,
}: {
  section: ConfigSection;
  sourceSectionsByTitle: Map<string, ConfigSection>;
  allFields: ConfigField[];
  disabledFieldReasonByKey: Map<string, string>;
  overrides: OverrideValues;
  showInheritedFields: boolean;
  inheritedDisabledReason?: string;
  index: number;
  idPrefix?: string;
}): RuntimeDefaultsSectionPresentation {
  const id = sectionElementId(index, section.title, idPrefix);
  const sourceSection = sourceSectionsByTitle.get(section.title) ?? section;
  const controlState = controlledSectionState(sourceSection, overrides);
  const disabledReason = inheritedDisabledReason ?? controlState.disabledReason;
  const controlField = controlState.controlField
    ? presentField(
        controlState.controlField,
        overrides,
        displayConfigFieldLabel(controlState.controlField, section.title),
        disabledFieldReasonByKey.get(controlState.controlField.key),
      )
    : undefined;
  const childFieldKeys = descendantFieldKeys(section.children ?? []);
  const bodySchemas = section.fields.filter(
    (field) => field.key !== controlField?.key && !childFieldKeys.has(field.key),
  );
  const rawGroups = boundaryModelFieldGroups(section.title, bodySchemas, overrides);
  const fieldGroups = rawGroups?.map((group, groupIndex) => {
    const groupControlField = group.controlField
      ? presentField(
          group.controlField,
          overrides,
          displayConfigFieldLabel(group.controlField, section.title, group.title),
          disabledFieldReasonByKey.get(group.controlField.key),
        )
      : undefined;
    const firstConcreteOption = group.controlField
      ? concreteConfigOptionChoice(group.controlField)
      : undefined;
    const isEnabled = group.isEnabled === true;
    return {
      id: sectionElementId(groupIndex, group.title, `${id}-boundary-group`),
      title: group.title,
      fields: group.fields.map((field) =>
        presentField(
          field,
          overrides,
          displayConfigFieldLabel(field, section.title, group.title),
          disabledFieldReasonByKey.get(field.key),
        ),
      ),
      controlField: groupControlField,
      isEnabled,
      isSwitchDisabled:
        !group.controlField ||
        group.controlField.locked === true ||
        (!isEnabled && firstConcreteOption === undefined),
      firstConcreteOption,
      disabledReason: group.disabledReason,
      stackHint: group.stackHint,
      metrics: metricsFor(group.fields, overrides),
    } satisfies RuntimeDefaultsFieldGroupPresentation;
  });
  const bodyFields = bodySchemas.map((field) =>
    presentField(
      field,
      overrides,
      displayConfigFieldLabel(field, section.title),
      disabledFieldReasonByKey.get(field.key),
    ),
  );
  const children = (section.children ?? []).map((child, childIndex) =>
    presentSection({
      section: child,
      sourceSectionsByTitle,
      allFields,
      disabledFieldReasonByKey,
      overrides,
      showInheritedFields,
      inheritedDisabledReason: disabledReason,
      index: childIndex,
      idPrefix: `${id}-nested-section`,
    }),
  );
  const inheritedHint = showInheritedFields
    ? inheritedHiddenModelFieldHint(
        { title: section.title, fields: section.fields },
        allFields,
      )
    : undefined;
  const inheritedField = inheritedHint
    ? {
        label: inheritedHint.label,
        title: inheritedHint.title,
        sourceTitle: inheritedHint.sourceTitle,
        field: presentField(
          inheritedHint.sourceField,
          overrides,
          inheritedHint.label,
          disabledFieldReasonByKey.get(inheritedHint.sourceField.key),
        ),
      }
    : undefined;
  const displayDescription =
    section.title === "Attention Mode" && !controlField
      ? undefined
      : displayConfigSectionDescription(section.title);

  return {
    id,
    title: section.title,
    displayTitle: displayConfigSectionTitle(section.title),
    displayDescription,
    fields: section.fields.map((field) =>
      presentField(
        field,
        overrides,
        displayConfigFieldLabel(field, section.title),
        disabledFieldReasonByKey.get(field.key),
      ),
    ),
    bodyFields,
    fieldGroups,
    children,
    controlField,
    disabledReason,
    isDisabled: disabledReason !== undefined,
    directMetrics: metricsFor(section.fields, overrides),
    treeMetrics: metricsFor(configSectionFields(section), overrides),
    stackInheritanceHint: inheritedStackSectionHint(
      {
        title: section.title,
        fields: section.fields,
        controlFieldKey: controlField?.key,
      },
      overrides,
    ),
    inheritedField,
    childTitleSignature: children.map((child) => child.title).join("\u0000"),
    groupTitleSignature: (fieldGroups ?? []).map((group) => group.title).join("\u0000"),
    enabledGroupTitleSignature: (fieldGroups ?? [])
      .filter((group) => group.isEnabled)
      .map((group) => group.title)
      .join("\u0000"),
  };
}

function matchesSearchQuery(
  option: RuntimeDefaultsSearchOptionPresentation,
  query: string,
) {
  return configSearchOptionMatchesQuery(option, query);
}

export function presentRuntimeDefaultsSchema({
  sections,
  overrides,
  search,
}: {
  sections: ConfigSection[];
  overrides: OverrideValues;
  search: ConfigSearchState;
}): RuntimeDefaultsSchemaPresentation {
  const schemaFields = configSectionsFields(sections);
  const sourceSections = deriveNestedConfigSections(sections);
  const sourceSectionsByTitle = sectionMap(sourceSections);
  const rootTitleBySectionTitle = rootSectionTitles(sourceSections);
  const disabledFieldReasonByKey = disabledConfigFieldReasons(sections, overrides);
  const selectedFieldKey = search.selectedFieldKey ?? null;
  const isSearchActive = search.query.trim().length > 0 || selectedFieldKey !== null;
  const filteredSections = filterConfigSectionsForSearch(sections, {
    query: search.query,
    selectedFieldKey,
  });
  const visibleSections = deriveNestedConfigSections(filteredSections, sections);
  const sectionPresentations = visibleSections.map((section, index) =>
    presentSection({
      section,
      sourceSectionsByTitle,
      allFields: schemaFields,
      disabledFieldReasonByKey,
      overrides,
      showInheritedFields: !isSearchActive,
      index,
    }),
  );
  const defaultOpenSectionTitles = isSearchActive
    ? sectionPresentations.map((section) => section.title)
    : sectionPresentations
        .filter((section, index) => index === 0 || section.treeMetrics.overrideCount > 0)
        .map((section) => section.title);
  const searchOptions = flattenConfigSearchOptions(sections).map((option) => ({
    sectionTitle: option.sectionTitle,
    rootSectionTitle:
      rootTitleBySectionTitle.get(option.sectionTitle) ?? option.sectionTitle,
    field: presentField(
      option.field,
      overrides,
      option.label,
      disabledFieldReasonByKey.get(option.key),
    ),
    key: option.key,
    label: option.label,
    configKey: option.configKey,
    flag: option.flag,
    type: option.type,
  }));

  return {
    schemaFields,
    fieldCount: schemaFields.length,
    presetOwnedFieldCount: presetOwnedCount(schemaFields),
    sections: sectionPresentations,
    defaultOpenSectionTitles,
    searchOpenKey: isSearchActive
      ? `${selectedFieldKey ?? ""}\u0000${search.query.trim()}`
      : "all",
    isSearchActive,
    search: {
      options: searchOptions,
      matchesQuery: matchesSearchQuery,
    },
  };
}
