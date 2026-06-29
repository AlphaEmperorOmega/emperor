import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { type ConfigField } from "@/lib/api";
import {
  commandField,
  configFieldGridFor,
  configFieldRowFor,
  expectFullConfigSectionGrid,
  expectResponsiveConfigFieldGrid,
  fullConfigSearchPopup,
  fullConfigSearchResultRow,
  fullConfigSectionFor,
  fullConfigSectionGridFor,
  fullConfigSectionNavRowFor,
  installFetchMock,
  openFullConfig,
  openTrainingCommand,
  renderViewer,
  resetViewerAppTestState,
  schemaResponse,
  scrollIntoViewMock,
  selectSearchableDropdownOption,
  selectTargetOption,
  typeConfigFieldValue,
  waitForOpenFullConfigButton,
  waitForTargetValue,
} from "./support";

function accordionPanelFor(accordion: HTMLElement) {
  const panelId = accordion.getAttribute("aria-controls");
  const panel = panelId ? document.getElementById(panelId) : null;

  if (!(panel instanceof HTMLElement)) {
    throw new Error("Expected accordion panel to render");
  }

  return panel;
}

function directFieldGridFor(accordion: HTMLElement) {
  const panel = accordionPanelFor(accordion);
  const grid = Array.from(panel.children).find(
    (child): child is HTMLElement =>
      child instanceof HTMLElement && child.classList.contains("md:grid-cols-2"),
  );

  if (!(grid instanceof HTMLElement)) {
    throw new Error("Expected accordion body to contain a direct field grid");
  }

  return grid;
}

function configFieldGroupFor(accordion: HTMLElement, title: string) {
  const panel = accordionPanelFor(accordion);
  const group = panel.querySelector(`[data-config-field-group="${title}"]`);

  if (!(group instanceof HTMLElement)) {
    throw new Error(`Expected ${title} config field group to render`);
  }

  return group;
}

function expectHeaderControlBeforeMetric(
  section: HTMLElement,
  controlLabel: string,
  metricLabel: string,
) {
  const switchControl = within(section).getByRole("switch", {
    name: controlLabel,
  });
  const headerControl = switchControl.closest("[data-config-section-header-control]");
  const metric = within(section).getByLabelText(metricLabel);

  if (!(headerControl instanceof HTMLElement)) {
    throw new Error(`Expected ${controlLabel} header control to render`);
  }

  expect(switchControl).toBeInTheDocument();
  expect(headerControl).toContainElement(switchControl);
  expect(within(headerControl).queryByText(controlLabel)).not.toBeInTheDocument();
  expect(within(headerControl).queryByText("Enabled")).not.toBeInTheDocument();
  expect(within(headerControl).queryByText("Off")).not.toBeInTheDocument();
  expect(
    Boolean(switchControl.compareDocumentPosition(metric) & Node.DOCUMENT_POSITION_FOLLOWING),
  ).toBe(true);
}

function expectNoHeaderControlInAccordionBody(
  accordion: HTMLElement,
  controlLabel: string,
) {
  const panel = accordionPanelFor(accordion);

  expect(within(panel).queryByRole("switch", { name: controlLabel }))
    .not.toBeInTheDocument();
  expect(within(panel).queryByText(controlLabel)).not.toBeInTheDocument();
}

function makeScrollable(element: HTMLElement) {
  Object.defineProperty(element, "clientHeight", {
    configurable: true,
    value: 180,
  });
  Object.defineProperty(element, "scrollHeight", {
    configurable: true,
    value: 900,
  });
  element.scrollTop = 760;
}

function expectBooleanSegmentedControl(container: HTMLElement, name: string | RegExp) {
  const control = within(container).getByRole("radiogroup", { name });
  const radios = within(control).getAllByRole("radio");
  const on = within(control).getByRole("radio", { name: "On" });
  const off = within(control).getByRole("radio", { name: "Off" });

  expect(radios).toEqual([on, off]);
  expect(within(control).queryByRole("radio", { name: "None" })).not.toBeInTheDocument();
  expect(within(container).queryByRole("switch", { name })).not.toBeInTheDocument();
  expect(within(container).queryByRole("combobox", { name })).not.toBeInTheDocument();

  return { control, on, off };
}

type ConfigFieldFixture = Partial<ConfigField> &
  Pick<ConfigField, "key" | "section" | "type" | "default">;

function configFixtureField({
  key,
  configKey,
  flag,
  label,
  section,
  type,
  default: defaultValue,
  nullable,
  choices,
  locked,
  lockedValue,
  lockedReason,
}: ConfigFieldFixture): ConfigField {
  return {
    key,
    configKey: configKey ?? key.toUpperCase(),
    flag: flag ?? `--${key.replace(/_/g, "-")}`,
    label: label ?? key.replace(/_/g, " "),
    section,
    type,
    default: defaultValue,
    nullable: nullable ?? defaultValue === null,
    choices: choices ?? [],
    locked,
    lockedValue,
    lockedReason,
  };
}

const canonicalStackFixtureFields = [
  configFixtureField({
    key: "stack_hidden_dim",
    section: "Layer Stack Options",
    type: "int",
    default: 256,
  }),
  configFixtureField({
    key: "stack_num_layers",
    section: "Layer Stack Options",
    type: "int",
    default: 5,
  }),
  configFixtureField({
    key: "stack_activation",
    section: "Layer Stack Options",
    type: "enum",
    default: "GELU",
    choices: ["DISABLED", "RELU", "GELU", "SIGMOID", "TANH"],
  }),
  configFixtureField({
    key: "stack_layer_norm_position",
    section: "Layer Stack Options",
    type: "enum",
    default: "BEFORE",
    choices: ["DISABLED", "DEFAULT", "BEFORE", "AFTER"],
  }),
  configFixtureField({
    key: "stack_apply_output_pipeline_flag",
    section: "Layer Stack Options",
    type: "bool",
    default: false,
    choices: [true, false],
  }),
  configFixtureField({
    key: "stack_bias_flag",
    section: "Layer Stack Options",
    type: "bool",
    default: true,
    choices: [true, false],
  }),
];

function stackFixtureField(
  overrides: Partial<ConfigField> & Pick<ConfigField, "key" | "section" | "default">,
): ConfigField {
  const suffix = overrides.key.split("_stack_", 2)[1];
  const canonical = canonicalStackFixtureFields.find(
    (field) => field.key === `stack_${suffix}`,
  );

  if (!canonical) {
    throw new Error(`Missing canonical stack field fixture for ${overrides.key}`);
  }

  return configFixtureField({
    ...overrides,
    type: canonical.type,
    choices: canonical.choices,
  });
}

function nestedControlledSchemaResponse() {
  return {
    ...schemaResponse,
    fields: [
      ...schemaResponse.fields,
      {
        key: "halting_flag",
        configKey: "HALTING_FLAG",
        flag: "--halting-flag",
        label: "halting flag",
        section: "Halting Options",
        type: "bool",
        default: false,
        nullable: false,
        choices: [true, false],
      },
      {
        key: "halting_threshold",
        configKey: "HALTING_THRESHOLD",
        flag: "--halting-threshold",
        label: "halting threshold",
        section: "Halting Options",
        type: "float",
        default: 0.99,
        nullable: false,
        choices: [],
      },
      stackFixtureField({
        key: "halting_stack_num_layers",
        section: "Halting Stack Options",
        default: 2,
        nullable: false,
      }),
      stackFixtureField({
        key: "halting_stack_hidden_dim",
        section: "Halting Stack Options",
        default: 64,
        nullable: false,
      }),
      stackFixtureField({
        key: "halting_stack_layer_norm_position",
        section: "Halting Stack Options",
        default: "BEFORE",
        nullable: false,
      }),
      {
        key: "memory_flag",
        configKey: "MEMORY_FLAG",
        flag: "--memory-flag",
        label: "memory flag",
        section: "Memory Options",
        type: "bool",
        default: false,
        nullable: false,
        choices: [true, false],
      },
      {
        key: "memory_option",
        configKey: "MEMORY_OPTION",
        flag: "--memory-option",
        label: "memory option",
        section: "Memory Options",
        type: "class",
        default: "GatedResidualDynamicMemoryConfig",
        nullable: false,
        choices: [
          "GatedResidualDynamicMemoryConfig",
          "WeightedDynamicMemoryConfig",
        ],
      },
      {
        key: "memory_position_option",
        configKey: "MEMORY_POSITION_OPTION",
        flag: "--memory-position-option",
        label: "memory position option",
        section: "Memory Options",
        type: "enum",
        default: "AFTER_AFFINE",
        nullable: false,
        choices: ["BEFORE_AFFINE", "AFTER_AFFINE"],
      },
      {
        key: "memory_test_time_training_learning_rate",
        configKey: "MEMORY_TEST_TIME_TRAINING_LEARNING_RATE",
        flag: "--memory-test-time-training-learning-rate",
        label: "memory test time training learning rate",
        section: "Memory Options",
        type: "float",
        default: null,
        nullable: true,
        choices: [],
      },
      stackFixtureField({
        key: "memory_stack_hidden_dim",
        section: "Memory Stack Options",
        default: 128,
        nullable: false,
      }),
      stackFixtureField({
        key: "memory_stack_layer_norm_position",
        section: "Memory Stack Options",
        default: "BEFORE",
        nullable: false,
      }),
      stackFixtureField({
        key: "memory_stack_num_layers",
        section: "Memory Stack Options",
        default: 2,
        nullable: false,
      }),
      stackFixtureField({
        key: "memory_stack_activation",
        section: "Memory Stack Options",
        default: "GELU",
        nullable: false,
      }),
      {
        key: "memory_stack_dropout_probability",
        configKey: "MEMORY_STACK_DROPOUT_PROBABILITY",
        flag: "--memory-stack-dropout-probability",
        label: "memory stack dropout probability",
        section: "Memory Stack Options",
        type: "float",
        default: 0,
        nullable: false,
        choices: [],
      },
      {
        key: "recurrent_flag",
        configKey: "RECURRENT_FLAG",
        flag: "--recurrent-flag",
        label: "recurrent flag",
        section: "Recurrent Layer Options",
        type: "bool",
        default: false,
        nullable: false,
        choices: [true, false],
      },
      {
        key: "recurrent_max_steps",
        configKey: "RECURRENT_MAX_STEPS",
        flag: "--recurrent-max-steps",
        label: "recurrent max steps",
        section: "Recurrent Layer Options",
        type: "int",
        default: 4,
        nullable: false,
        choices: [],
      },
      {
        key: "recurrent_gate_flag",
        configKey: "RECURRENT_GATE_FLAG",
        flag: "--recurrent-gate-flag",
        label: "recurrent gate flag",
        section: "Recurrent Gate Options",
        type: "bool",
        default: false,
        nullable: false,
        choices: [true, false],
      },
      stackFixtureField({
        key: "recurrent_gate_stack_hidden_dim",
        section: "Recurrent Gate Stack Options",
        default: 128,
        nullable: false,
      }),
      {
        key: "recurrent_halting_flag",
        configKey: "RECURRENT_HALTING_FLAG",
        flag: "--recurrent-halting-flag",
        label: "recurrent halting flag",
        section: "Recurrent Halting Options",
        type: "bool",
        default: false,
        nullable: false,
        choices: [true, false],
      },
      {
        key: "recurrent_halting_threshold",
        configKey: "RECURRENT_HALTING_THRESHOLD",
        flag: "--recurrent-halting-threshold",
        label: "recurrent halting threshold",
        section: "Recurrent Halting Options",
        type: "float",
        default: 0.95,
        nullable: false,
        choices: [],
      },
      stackFixtureField({
        key: "recurrent_halting_stack_num_layers",
        section: "Recurrent Halting Stack Options",
        default: 2,
        nullable: false,
      }),
      stackFixtureField({
        key: "recurrent_halting_stack_hidden_dim",
        section: "Recurrent Halting Stack Options",
        default: 64,
        nullable: false,
      }),
      stackFixtureField({
        key: "recurrent_halting_stack_layer_norm_position",
        section: "Recurrent Halting Stack Options",
        default: "BEFORE",
        nullable: false,
      }),
    ],
  };
}

function gateOptionSchemaResponse() {
  return {
    ...schemaResponse,
    fields: [
      ...schemaResponse.fields,
      {
        key: "gate_option",
        configKey: "GATE_OPTION",
        flag: "--gate-option",
        label: "gate option",
        section: "Gate Options",
        type: "enum",
        default: "MULTIPLIER",
        nullable: true,
        choices: ["MULTIPLIER", "ADDITION"],
      },
      {
        key: "gate_activation",
        configKey: "GATE_ACTIVATION",
        flag: "--gate-activation",
        label: "gate activation",
        section: "Gate Options",
        type: "enum",
        default: "SIGMOID",
        nullable: true,
        choices: ["None", "SIGMOID", "TANH"],
      },
      stackFixtureField({
        key: "gate_stack_hidden_dim",
        section: "Gate Stack Options",
        default: 128,
        nullable: false,
      }),
      stackFixtureField({
        key: "gate_stack_layer_norm_position",
        section: "Gate Stack Options",
        default: "BEFORE",
        nullable: false,
      }),
      stackFixtureField({
        key: "gate_stack_bias_flag",
        section: "Gate Stack Options",
        default: true,
        nullable: false,
      }),
      stackFixtureField({
        key: "gate_stack_num_layers",
        section: "Gate Stack Options",
        default: 2,
        nullable: false,
      }),
      {
        key: "recurrent_gate_flag",
        configKey: "RECURRENT_GATE_FLAG",
        flag: "--recurrent-gate-flag",
        label: "recurrent gate flag",
        section: "Recurrent Gate Options",
        type: "bool",
        default: false,
        nullable: false,
        choices: [true, false],
      },
      {
        key: "recurrent_gate_option",
        configKey: "RECURRENT_GATE_OPTION",
        flag: "--recurrent-gate-option",
        label: "recurrent gate option",
        section: "Recurrent Gate Options",
        type: "enum",
        default: "MULTIPLIER",
        nullable: true,
        choices: ["MULTIPLIER", "ADDITION"],
      },
      {
        key: "recurrent_gate_activation",
        configKey: "RECURRENT_GATE_ACTIVATION",
        flag: "--recurrent-gate-activation",
        label: "recurrent gate activation",
        section: "Recurrent Gate Options",
        type: "enum",
        default: "SIGMOID",
        nullable: true,
        choices: ["None", "SIGMOID", "TANH"],
      },
      stackFixtureField({
        key: "recurrent_gate_stack_hidden_dim",
        section: "Recurrent Gate Stack Options",
        default: 128,
        nullable: false,
      }),
      stackFixtureField({
        key: "recurrent_gate_stack_layer_norm_position",
        section: "Recurrent Gate Stack Options",
        default: "BEFORE",
        nullable: false,
      }),
      stackFixtureField({
        key: "recurrent_gate_stack_bias_flag",
        section: "Recurrent Gate Stack Options",
        default: true,
        nullable: false,
      }),
      stackFixtureField({
        key: "recurrent_gate_stack_num_layers",
        section: "Recurrent Gate Stack Options",
        default: 2,
        nullable: false,
      }),
    ],
  };
}

function adaptiveComponentSchemaResponse() {
  const adaptiveField = (
    key: string,
    section: string,
    type: "bool" | "class" | "int",
    defaultValue: boolean | string | number | null,
    choices: Array<boolean | string> = [],
  ) => ({
    key,
    configKey: key.toUpperCase(),
    flag: `--${key.replace(/_/g, "-")}`,
    label: key.replace(/_/g, " "),
    section,
    type,
    default: defaultValue,
    nullable: defaultValue === null,
    choices,
  });

  return {
    ...schemaResponse,
    fields: [
      ...schemaResponse.fields,
      adaptiveField("weight_option_flag", "Weight Generator Options", "bool", false, [
        true,
        false,
      ]),
      adaptiveField("weight_option", "Weight Generator Options", "class", null, [
        "DualModelDynamicWeightConfig",
      ]),
      adaptiveField(
        "weight_generator_stack_independent_flag",
        "Weight Generator Stack Options",
        "bool",
        false,
        [true, false],
      ),
      stackFixtureField({
        key: "weight_generator_stack_hidden_dim",
        section: "Weight Generator Stack Options",
        default: null,
      }),
      adaptiveField("bias_option_flag", "Bias Generator Options", "bool", false, [
        true,
        false,
      ]),
      adaptiveField("bias_option", "Bias Generator Options", "class", null, [
        "AdditiveDynamicBiasConfig",
      ]),
      adaptiveField("diagonal_option_flag", "Diagonal Generator Options", "bool", false, [
        true,
        false,
      ]),
      adaptiveField("diagonal_option", "Diagonal Generator Options", "class", null, [
        "CombinedDynamicDiagonalConfig",
      ]),
      adaptiveField("mask_option_flag", "Mask Options", "bool", false, [
        true,
        false,
      ]),
      adaptiveField("row_mask_option", "Mask Options", "class", null, [
        "WeightInformedScoreAxisMaskConfig",
      ]),
      adaptiveField(
        "mask_generator_stack_independent_flag",
        "Mask Stack Options",
        "bool",
        false,
        [true, false],
      ),
      stackFixtureField({
        key: "mask_generator_stack_hidden_dim",
        section: "Mask Stack Options",
        default: null,
      }),
      stackFixtureField({
        key: "adaptive_submodule_stack_hidden_dim",
        section: "Adaptive Submodule Stack Options",
        default: 256,
      }),
      stackFixtureField({
        key: "adaptive_submodule_stack_num_layers",
        section: "Adaptive Submodule Stack Options",
        default: 2,
      }),
    ],
  };
}

function boundaryProjectorSchemaResponse() {
  const boundaryField = (
    key: string,
    section: string,
    type: "bool" | "class" | "float" | "int",
    defaultValue: boolean | string | number | null,
    choices: Array<boolean | string> = [],
  ) => ({
    key,
    configKey: key.toUpperCase(),
    flag: `--${key.replace(/_/g, "-")}`,
    label: key.replace(/_/g, " "),
    section,
    type,
    default: defaultValue,
    nullable: defaultValue === null,
    choices,
  });

  return {
    ...schemaResponse,
    fields: [
      ...schemaResponse.fields,
      boundaryField(
        "input_layer_weight_option",
        "Input Boundary Projector Options",
        "class",
        null,
        ["DualModelDynamicWeightConfig"],
      ),
      boundaryField(
        "input_layer_weight_decay_warmup_batches",
        "Input Boundary Projector Options",
        "int",
        0,
      ),
      boundaryField(
        "input_layer_weight_decay_rate",
        "Input Boundary Projector Options",
        "float",
        0,
      ),
      boundaryField(
        "input_layer_bias_option",
        "Input Boundary Projector Options",
        "class",
        null,
        ["AdditiveDynamicBiasConfig"],
      ),
      boundaryField(
        "input_layer_diagonal_option",
        "Input Boundary Projector Options",
        "class",
        null,
        ["CombinedDynamicDiagonalConfig"],
      ),
      boundaryField(
        "input_layer_row_mask_option",
        "Input Boundary Projector Options",
        "class",
        null,
        ["WeightInformedScoreAxisMaskConfig"],
      ),
      boundaryField(
        "output_layer_weight_option",
        "Output Boundary Projector Options",
        "class",
        null,
        ["DualModelDynamicWeightConfig"],
      ),
      boundaryField(
        "output_layer_bias_option",
        "Output Boundary Projector Options",
        "class",
        null,
        ["AdditiveDynamicBiasConfig"],
      ),
      boundaryField(
        "output_layer_diagonal_option",
        "Output Boundary Projector Options",
        "class",
        null,
        ["CombinedDynamicDiagonalConfig"],
      ),
      boundaryField(
        "output_layer_row_mask_option",
        "Output Boundary Projector Options",
        "class",
        null,
        ["WeightInformedScoreAxisMaskConfig"],
      ),
    ],
  };
}

describe("ViewerApp Full Config", () => {
  beforeEach(resetViewerAppTestState);

  it("keeps full config controls out of the left sidebar", async () => {
    installFetchMock();
    renderViewer();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /open full config/i }),
    ).toBeEnabled();
    expect(screen.queryByText("Sections")).not.toBeInTheDocument();
    expect(screen.queryByText("Fields")).not.toBeInTheDocument();
    expect(screen.queryByText("Changed")).not.toBeInTheDocument();
    expect(screen.queryByText("Layer Stack Options")).not.toBeInTheDocument();
    expect(screen.queryByText("Gate Stack Options")).not.toBeInTheDocument();
    expect(screen.queryByText("Modified Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
  });

  it("hides the open full config action while selecting experiment targets", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForOpenFullConfigButton(user);

    await user.click(
      screen.getByRole("radio", { name: /^experiments$/i }),
    );

    await waitFor(() => {
      expect(screen.getByRole("radio", { name: /^experiments$/i }))
        .toHaveAttribute("aria-checked", "true");
      expect(screen.queryByRole("button", { name: /open full config/i }))
        .not.toBeInTheDocument();
    });
  });

  it("keeps the global snapshot library out of the model sidebar", async () => {
    installFetchMock({
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [],
      },
      configSnapshotLibraryResponse: {
        snapshots: [
          {
            id: "bert-snapshot",
            modelType: "transformer_encoder",
            model: "bert_linear",
            preset: "bert-baseline",
            name: "Bert tuned",
            overrides: { stack_hidden_dim: "128" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderViewer();

    await waitForTargetValue("preset", "baseline");
    expect(screen.getByRole("button", { name: /open full config/i })).toBeEnabled();
    expect(screen.queryByRole("heading", { name: "Snapshots" }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("Bert tuned")).not.toBeInTheDocument();
  });

  it("opens snapshot draft config from the preset sidebar action", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    const createButton = await screen.findByRole("button", {
      name: "Create Snapshot",
    });
    await waitFor(() => expect(createButton).toBeEnabled());
    await user.click(createButton);

    const dialog = await screen.findByRole("dialog", {
      name: /full configuration/i,
    });
    expect(
      within(dialog).getByRole("button", { name: "Save as Snapshot" }),
    ).toBeInTheDocument();
  });

  it("duplicates the selected snapshot as a new snapshot", async () => {
    const { configSnapshotCreateRequests, configSnapshotUpdateRequests } =
      installFetchMock({
        configSnapshotsResponse: {
          modelType: "linears",
          model: "linear",
          snapshots: [
            {
              id: "snapshot-wide",
              modelType: "linears",
              model: "linear",
              preset: "baseline",
              name: "Wide",
              overrides: { stack_hidden_dim: "128" },
              createdAt: "2026-06-01T00:00:00.000Z",
              updatedAt: "2026-06-01T00:00:00.000Z",
            },
          ],
        },
      });
    renderViewer();
    const user = userEvent.setup();

    const snapshotsButton = await screen.findByRole("radio", {
      name: /snapshots/i,
    });
    await waitFor(() => expect(snapshotsButton).toBeEnabled());
    await user.click(snapshotsButton);
    expect(await screen.findByRole("combobox", { name: /^snapshot$/i }))
      .toHaveTextContent("Wide");

    await user.click(screen.getByRole("button", { name: "Duplicate" }));
    const dialog = await screen.findByRole("dialog", {
      name: /full configuration/i,
    });
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "192");
    await user.click(
      within(dialog).getByRole("button", { name: "Save as Snapshot" }),
    );
    const saveDialog = await screen.findByRole("dialog", {
      name: "Save as Snapshot",
    });
    const saveButton = within(saveDialog).getByRole("button", {
      name: "Save Snapshot",
    });
    expect(saveButton).toBeDisabled();
    await user.type(
      within(saveDialog).getByLabelText(/^name$/i),
      "Wide copy",
    );
    await waitFor(() => expect(saveButton).toBeEnabled());
    await user.click(saveButton);

    await waitFor(() => expect(configSnapshotCreateRequests).toHaveLength(1));
    expect(configSnapshotCreateRequests[0]).toMatchObject({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      name: "Wide copy",
      overrides: { stack_hidden_dim: "192" },
    });
    expect(configSnapshotUpdateRequests).toHaveLength(0);
  });

  it("edits the selected snapshot with an update request", async () => {
    const { configSnapshotCreateRequests, configSnapshotUpdateRequests } =
      installFetchMock({
        configSnapshotsResponse: {
          modelType: "linears",
          model: "linear",
          snapshots: [
            {
              id: "snapshot-wide",
              modelType: "linears",
              model: "linear",
              preset: "baseline",
              name: "Wide",
              overrides: { stack_hidden_dim: "128" },
              createdAt: "2026-06-01T00:00:00.000Z",
              updatedAt: "2026-06-01T00:00:00.000Z",
            },
          ],
        },
      });
    renderViewer();
    const user = userEvent.setup();

    const snapshotsButton = await screen.findByRole("radio", {
      name: /snapshots/i,
    });
    await waitFor(() => expect(snapshotsButton).toBeEnabled());
    await user.click(snapshotsButton);
    await user.click(await screen.findByRole("button", { name: "Edit" }));

    const dialog = await screen.findByRole("dialog", {
      name: /full configuration/i,
    });
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "192");
    await user.click(
      within(dialog).getByRole("button", { name: "Save Snapshot Changes" }),
    );
    const saveDialog = await screen.findByRole("dialog", {
      name: "Save Snapshot Changes",
    });
    expect(within(saveDialog).getByLabelText(/^name$/i)).toHaveValue("Wide");
    await user.click(
      within(saveDialog).getByRole("button", {
        name: "Save Snapshot Changes",
      }),
    );

    await waitFor(() => expect(configSnapshotUpdateRequests).toHaveLength(1));
    expect(configSnapshotUpdateRequests[0]).toEqual({
      id: "snapshot-wide",
      body: {
        name: "Wide",
        overrides: { stack_hidden_dim: "192" },
      },
    });
    expect(configSnapshotCreateRequests).toHaveLength(0);
  });

  it("opens the full config popup with section accordions expanded by default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const dialogHeader = dialog.querySelector("header");
    const dialogBody = dialog.querySelector(".full-config-dialog-body");
    const dialogFooter = dialog.querySelector("footer");
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate options section, 1 field, 0 overrides/i,
    });
    const layerNavToggle = within(sectionNav).getByRole("button", {
      name: /close layer stack options/i,
    });
    const gateNavToggle = within(sectionNav).getByRole("button", {
      name: /open gate options/i,
    });
    const layerSection = layerAccordion.closest("section");
    const gateSection = gateAccordion.closest("section");
    const sectionGrid = fullConfigSectionGridFor(layerAccordion);
    const closeButton = within(dialog).getByRole("button", {
      name: /close full config/i,
    });

    if (
      !(dialogHeader instanceof HTMLElement) ||
      !(dialogBody instanceof HTMLElement) ||
      !(dialogFooter instanceof HTMLElement) ||
      !(layerSection instanceof HTMLElement) ||
      !(gateSection instanceof HTMLElement)
    ) {
      throw new Error("Expected full config dialog chrome to render");
    }

    expect(dialog).toHaveClass(
      "full-config-dialog-shell",
      "rounded-[10px]",
      "border",
      "border-line",
      "bg-panel",
    );
    expect(dialog).not.toHaveClass("edge", "rounded-card", "bg-white/[0.018]");
    expect(dialogHeader).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-header",
      "border-line-soft",
    );
    expect(dialogHeader).not.toHaveClass("bg-panel/85");
    expect(dialogHeader).not.toHaveClass("border-line");
    expect(dialogBody).toHaveClass("full-config-dialog-body");
    expect(dialogBody).not.toHaveClass("bg-bg-2/80");
    expect(dialogFooter).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-footer",
      "border-line-soft",
    );
    expect(dialogFooter).not.toHaveClass("bg-panel/85");
    expect(dialogFooter).not.toHaveClass("border-line");
    expect(closeButton).toHaveClass("border-line-soft", "bg-white/[0.025]");
    expect(closeButton).not.toHaveClass("bg-white/[0.035]");
    expect(sectionGrid).toBe(fullConfigSectionGridFor(gateAccordion));
    expectFullConfigSectionGrid(sectionGrid);
    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(layerAccordion).toHaveAttribute("aria-controls");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toHaveAttribute("aria-controls");
    expect(gateAccordion).toBeDisabled();
    expect(layerAccordion.closest("h3")).toHaveClass("h-full");
    expect(layerAccordion).toHaveClass("h-full");
    expect(gateAccordion.closest("h3")).toHaveClass("h-full");
    expect(gateAccordion).toHaveClass("h-full");
    expect(layerAccordion).toHaveClass("overflow-hidden", "bg-white/[0.055]");
    expect(gateAccordion).toHaveClass("overflow-hidden", "bg-white/[0.025]");
    expect(layerSection).toHaveClass(
      "grid",
      "content-start",
      "gap-1.5",
      "relative",
      "overflow-visible",
      "rounded-[10px]",
      "border",
      "border-line",
      "bg-white/[0.018]",
      "px-0",
      "py-0",
      "shadow-[0_16px_40px_-30px_rgba(0,0,0,0.95)]",
      "focus-within:z-30",
    );
    expect(layerSection).not.toHaveClass("overflow-hidden");
    expect(layerSection).not.toHaveClass("rounded-[12px]", "bg-panel/80");
    expect(gateSection).toHaveClass(
      "grid",
      "content-start",
      "gap-1.5",
      "relative",
      "overflow-visible",
      "rounded-[10px]",
      "border",
      "border-line-soft",
      "bg-white/[0.012]",
      "px-0",
      "py-0",
      "shadow-[0_10px_28px_-26px_rgba(0,0,0,0.9)]",
      "focus-within:z-30",
    );
    expect(gateSection).not.toHaveClass("overflow-hidden");
    expect(gateSection).not.toHaveClass("rounded-[12px]", "bg-panel/70");
    expect(layerNavToggle).toHaveAttribute("aria-expanded", "true");
    expect(layerNavToggle).toHaveAttribute("aria-controls");
    expect(gateNavToggle).toHaveAttribute("aria-expanded", "false");
    expect(gateNavToggle).toHaveAttribute("aria-controls");
    expect(gateNavToggle).toBeDisabled();
    expect(layerAccordion).not.toHaveTextContent(/3 fields|0 overrides/i);
    expect(within(layerSection).getByLabelText("3 fields")).not.toHaveAttribute("tabindex");
    expect(within(layerSection).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
    expect(within(gateSection).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(within(gateSection).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
    expect(within(dialog).getByLabelText("4 fields")).toHaveTextContent("4");
    expect(within(dialog).getByLabelText("4 fields")).not.toHaveTextContent("4 fields");
    const layerJump = within(sectionNav).getByRole("button", {
      name: /jump to layer stack options/i,
    });
    const gateJump = within(sectionNav).getByRole("button", {
      name: /jump to gate options/i,
    });
    const layerNavRow = layerJump.parentElement?.parentElement;
    expect(layerNavRow).toHaveClass("group/section-row");
    expect(layerNavRow).toHaveClass("focus-within:ring-2", "hover:bg-violet/10");
    expect(layerJump).not.toHaveClass("peer/title", "group/title");
    expect(layerJump).not.toHaveClass("pr-[7.75rem]");
    expect(gateJump).not.toHaveClass("pr-[7.75rem]");
    expect(within(sectionNav).getByLabelText("3 fields")).toHaveTextContent("3");
    expect(within(sectionNav).getByLabelText("3 fields")).not.toHaveAttribute("tabindex");
    within(sectionNav)
      .getAllByLabelText("0 overrides")
      .forEach((metric) => expect(metric).not.toHaveAttribute("tabindex"));
    expect(layerJump).not.toHaveTextContent(/3 fields|0 overrides/i);
    expect(within(sectionNav).getByLabelText("1 field")).toHaveTextContent("1");
    expect(within(sectionNav).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(gateJump).not.toHaveTextContent(/1 field|0 overrides/i);
    const hiddenDimControl = within(dialog).getByLabelText(/hidden dim/i);
    const gateSwitch = within(dialog).getByRole("switch", { name: /gate flag/i });

    expect(hiddenDimControl).toBeInTheDocument();
    expectResponsiveConfigFieldGrid(configFieldGridFor(hiddenDimControl));
    expect(hiddenDimControl).toHaveClass("h-10", "px-3", "py-2");
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(gateSwitch).toBeInTheDocument();
    expect(gateAccordion).not.toContainElement(gateSwitch);
    expect(gateSwitch.parentElement).toHaveClass("inline-flex", "items-center", "px-2.5");
    expect(gateSwitch.parentElement).not.toHaveTextContent(/gate flag\s*Off/i);
    expectHeaderControlBeforeMetric(gateSection, "gate flag", "1 field");
    expect(within(dialog).queryByText("--stack-hidden-dim")).not.toBeInTheDocument();
    expect(within(dialog).queryByText("--gate-flag")).not.toBeInTheDocument();
  });

  it("keeps layer stack-prefixed fields in the parent accordion", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "stack_dropout_probability",
            configKey: "STACK_DROPOUT_PROBABILITY",
            flag: "--stack-dropout-probability",
            label: "stack dropout probability",
            section: "Layer Stack Options",
            type: "float",
            default: 0.2,
            nullable: false,
            choices: [],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const layerAccordion = within(dialog).getByRole("button", {
      name: /^layer stack options section, 4 fields, 0 overrides/i,
    });

    if (layerAccordion.getAttribute("aria-expanded") !== "true") {
      await user.click(layerAccordion);
    }

    const layerSection = fullConfigSectionFor(layerAccordion);
    expect(
      within(layerSection).queryByRole("button", {
        name: /^layer stack section,/i,
      }),
    ).not.toBeInTheDocument();

    const layerDirectGrid = directFieldGridFor(layerAccordion);
    expect(within(layerDirectGrid).getByLabelText(/stack hidden dim/i))
      .toBeInTheDocument();
    expect(within(layerDirectGrid).getByLabelText(/stack num layers/i))
      .toBeInTheDocument();
    expect(within(layerDirectGrid).getByLabelText(/stack activation/i))
      .toBeInTheDocument();
  });

  it("shows shared layer stack submodule defaults as a top-level accordion", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "submodule_stack_hidden_dim",
            configKey: "SUBMODULE_STACK_HIDDEN_DIM",
            flag: "--submodule-hidden-dim",
            label: "submodule hidden dim",
            section: "Layer Stack Submodule Options",
            type: "int",
            default: 256,
            nullable: false,
            choices: [],
          },
          {
            key: "submodule_stack_activation",
            configKey: "SUBMODULE_STACK_ACTIVATION",
            flag: "--submodule-stack-activation",
            label: "submodule stack activation",
            section: "Layer Stack Submodule Options",
            type: "enum",
            default: "GELU",
            nullable: false,
            choices: ["GELU", "MISH"],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const submoduleAccordion = within(dialog).getByRole("button", {
      name: /layer stack submodule options section, 2 fields, 0 overrides/i,
    });

    expect(submoduleAccordion).toHaveAttribute("aria-expanded", "false");
    expect(
      within(sectionNav).getByRole("button", {
        name: /jump to layer stack submodule options/i,
      }),
    ).toBeInTheDocument();
    await user.click(submoduleAccordion);
    expect(
      within(dialog).queryByRole("button", {
        name: /submodule stack options section/i,
      }),
    ).not.toBeInTheDocument();
    const submoduleDirectGrid = directFieldGridFor(submoduleAccordion);
    expect(within(submoduleDirectGrid).getByLabelText(/submodule hidden dim/i))
      .toBeInTheDocument();
    expect(within(submoduleDirectGrid).getByLabelText(/submodule stack activation/i))
      .toBeInTheDocument();
  });

  it("locks controlled config accordions until their header flag is enabled", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          stackFixtureField({
            key: "gate_stack_hidden_dim",
            section: "Gate Stack Options",
            default: 256,
            nullable: false,
          }),
          {
            key: "halting_flag",
            configKey: "HALTING_FLAG",
            flag: "--halting-flag",
            label: "halting flag",
            section: "Halting Options",
            type: "bool",
            default: false,
            nullable: false,
            choices: [true, false],
          },
          {
            key: "halting_threshold",
            configKey: "HALTING_THRESHOLD",
            flag: "--halting-threshold",
            label: "halting threshold",
            section: "Halting Options",
            type: "float",
            default: 0.99,
            nullable: false,
            choices: [],
          },
          {
            key: "recurrent_flag",
            configKey: "RECURRENT_FLAG",
            flag: "--recurrent-flag",
            label: "recurrent flag",
            section: "Recurrent Layer Options",
            type: "bool",
            default: false,
            nullable: false,
            choices: [true, false],
          },
          {
            key: "recurrent_max_steps",
            configKey: "RECURRENT_MAX_STEPS",
            flag: "--recurrent-max-steps",
            label: "recurrent max steps",
            section: "Recurrent Layer Options",
            type: "int",
            default: 4,
            nullable: false,
            choices: [],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate options section, 2 fields, 0 overrides/i,
    });
    const haltingAccordion = within(dialog).getByRole("button", {
      name: /halting options section, 2 fields, 0 overrides/i,
    });
    const recurrentAccordion = within(dialog).getByRole("button", {
      name: /recurrent layer options section, 2 fields, 0 overrides/i,
    });
    const gateSection = fullConfigSectionFor(gateAccordion);
    const haltingSection = fullConfigSectionFor(haltingAccordion);
    const recurrentSection = fullConfigSectionFor(recurrentAccordion);

    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(haltingAccordion).toHaveAttribute("aria-expanded", "false");
    expect(recurrentAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toBeDisabled();
    expect(haltingAccordion).toBeDisabled();
    expect(recurrentAccordion).toBeDisabled();
    expectHeaderControlBeforeMetric(gateSection, "gate flag", "2 fields");
    expectHeaderControlBeforeMetric(haltingSection, "halting flag", "2 fields");
    expectHeaderControlBeforeMetric(recurrentSection, "recurrent flag", "2 fields");
    expect(within(gateSection).getByLabelText("0 overrides")).toBeInTheDocument();
    expect(within(haltingSection).getByLabelText("0 overrides")).toBeInTheDocument();
    expect(within(recurrentSection).getByLabelText("0 overrides")).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/gate stack hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/halting threshold/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/recurrent max steps/i)).not.toBeInTheDocument();

    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });
    await user.type(search, "gate stack hidden");
    const searchPopup = fullConfigSearchPopup(dialog);
    const gateHiddenSearchRow = fullConfigSearchResultRow(
      searchPopup,
      /gate stack hidden dim/i,
    );
    expect(
      within(gateHiddenSearchRow).getByRole("textbox", {
        name: /current value/i,
      }),
    ).toBeDisabled();
    expect(gateHiddenSearchRow).toHaveTextContent(
      /enable gate flag before editing gate options/i,
    );

    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));
    await user.click(within(dialog).getByRole("switch", { name: /gate flag/i }));
    await user.click(within(dialog).getByRole("switch", { name: /halting flag/i }));
    await user.click(within(dialog).getByRole("switch", { name: /recurrent flag/i }));

    const enabledGateAccordion = within(dialog).getByRole("button", {
      name: /gate options section, 2 fields, 1 override/i,
    });
    const enabledHaltingAccordion = within(dialog).getByRole("button", {
      name: /halting options section, 2 fields, 1 override/i,
    });
    const enabledRecurrentAccordion = within(dialog).getByRole("button", {
      name: /recurrent layer options section, 2 fields, 1 override/i,
    });
    const enabledGateSection = fullConfigSectionFor(enabledGateAccordion);
    const enabledHaltingSection = fullConfigSectionFor(enabledHaltingAccordion);
    const enabledRecurrentSection = fullConfigSectionFor(enabledRecurrentAccordion);

    expect(enabledGateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(enabledHaltingAccordion).toHaveAttribute("aria-expanded", "true");
    expect(enabledRecurrentAccordion).toHaveAttribute("aria-expanded", "true");
    expectHeaderControlBeforeMetric(enabledGateSection, "gate flag", "2 fields");
    expectHeaderControlBeforeMetric(enabledHaltingSection, "halting flag", "2 fields");
    expectHeaderControlBeforeMetric(enabledRecurrentSection, "recurrent flag", "2 fields");
    expect(within(enabledGateSection).getByLabelText("1 override")).toBeInTheDocument();
    expect(within(enabledHaltingSection).getByLabelText("1 override")).toBeInTheDocument();
    expect(within(enabledRecurrentSection).getByLabelText("1 override")).toBeInTheDocument();
    expectNoHeaderControlInAccordionBody(enabledGateAccordion, "gate flag");
    expectNoHeaderControlInAccordionBody(enabledHaltingAccordion, "halting flag");
    expectNoHeaderControlInAccordionBody(enabledRecurrentAccordion, "recurrent flag");
    expect(within(dialog).getByLabelText(/gate stack hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/halting threshold/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/recurrent max steps/i)).toBeInTheDocument();
  });

  it("uses adaptive option flags as section header controls", async () => {
    installFetchMock({ schemaResponse: adaptiveComponentSchemaResponse() });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const weightAccordion = within(dialog).getByRole("button", {
      name: /^weight generator options section, 4 fields, 0 overrides/i,
    });
    const biasAccordion = within(dialog).getByRole("button", {
      name: /^bias generator options section, 2 fields, 0 overrides/i,
    });
    const diagonalAccordion = within(dialog).getByRole("button", {
      name: /^diagonal generator options section, 2 fields, 0 overrides/i,
    });
    const maskAccordion = within(dialog).getByRole("button", {
      name: /^mask options section, 4 fields, 0 overrides/i,
    });
    const weightSection = fullConfigSectionFor(weightAccordion);
    const biasSection = fullConfigSectionFor(biasAccordion);
    const diagonalSection = fullConfigSectionFor(diagonalAccordion);
    const maskSection = fullConfigSectionFor(maskAccordion);

    expect(weightAccordion).toBeDisabled();
    expect(biasAccordion).toBeDisabled();
    expect(diagonalAccordion).toBeDisabled();
    expect(maskAccordion).toBeDisabled();
    expectHeaderControlBeforeMetric(
      weightSection,
      "weight option flag",
      "4 fields",
    );
    expectHeaderControlBeforeMetric(biasSection, "bias option flag", "2 fields");
    expectHeaderControlBeforeMetric(
      diagonalSection,
      "diagonal option flag",
      "2 fields",
    );
    expectHeaderControlBeforeMetric(
      maskSection,
      "mask option flag",
      "4 fields",
    );
    expect(within(dialog).queryByLabelText(/^weight option$/i))
      .not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/^row mask option$/i))
      .not.toBeInTheDocument();

    await user.click(
      within(dialog).getByRole("switch", { name: /^weight option flag$/i }),
    );
    await user.click(
      within(dialog).getByRole("switch", { name: /^mask option flag$/i }),
    );

    const enabledWeightAccordion = within(dialog).getByRole("button", {
      name: /^weight generator options section, 4 fields, 2 overrides/i,
    });
    const enabledMaskAccordion = within(dialog).getByRole("button", {
      name: /^mask options section, 4 fields, 2 overrides/i,
    });
    const weightDirectGrid = directFieldGridFor(enabledWeightAccordion);
    const maskDirectGrid = directFieldGridFor(enabledMaskAccordion);
    const weightGeneratorAccordion = within(
      fullConfigSectionFor(enabledWeightAccordion),
    ).getByRole("button", {
      name: /^weight generator stack options section, 2 fields, 0 overrides/i,
    });
    const maskGeneratorAccordion = within(
      fullConfigSectionFor(enabledMaskAccordion),
    ).getByRole("button", {
      name: /^mask stack options section, 2 fields, 0 overrides/i,
    });

    expect(within(weightDirectGrid).getByLabelText(/^weight option$/i))
      .toBeInTheDocument();
    expect(within(maskDirectGrid).getByLabelText(/^row mask option$/i))
      .toBeInTheDocument();
    expect(weightGeneratorAccordion).toBeDisabled();
    expect(maskGeneratorAccordion).toBeDisabled();
    expectHeaderControlBeforeMetric(
      fullConfigSectionFor(weightGeneratorAccordion),
      "weight generator stack independent flag",
      "2 fields",
    );
    expectHeaderControlBeforeMetric(
      fullConfigSectionFor(maskGeneratorAccordion),
      "mask generator stack independent flag",
      "2 fields",
    );

    const adaptiveStackOptionsAccordion = within(dialog).getByRole("button", {
      name: /^adaptive submodule stack options section, 2 fields, 0 overrides/i,
    });
    await user.click(adaptiveStackOptionsAccordion);
    const adaptiveStackDirectGrid = directFieldGridFor(adaptiveStackOptionsAccordion);

    expect(
      within(adaptiveStackDirectGrid).getByLabelText(
        /adaptive submodule stack hidden dim/i,
      ),
    ).toBeInTheDocument();

    await user.click(
      within(fullConfigSectionFor(enabledMaskAccordion)).getByRole("switch", {
        name: /^mask generator stack independent flag$/i,
      }),
    );
    expect(
      within(accordionPanelFor(maskGeneratorAccordion)).getByLabelText(
        /mask generator stack hidden dim/i,
      ),
    ).toBeInTheDocument();

    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });
    await user.type(search, "weight generator stack hidden");
    const hiddenDimRow = fullConfigSearchResultRow(
      fullConfigSearchPopup(dialog),
      /weight generator stack hidden dim/i,
    );
    expect(
      within(hiddenDimRow).getByRole("textbox", {
        name: /current value/i,
      }),
    ).toBeDisabled();
    expect(hiddenDimRow).toHaveTextContent(
      /enable weight generator stack independent flag before editing weight generator stack options/i,
    );
    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));

    await user.click(
      within(dialog).getByRole("switch", {
        name: /^weight generator stack independent flag$/i,
      }),
    );
    const enabledWeightGeneratorAccordion = within(dialog).getByRole("button", {
      name: /^weight generator stack options section, 2 fields, 1 override/i,
    });

    expect(
      within(accordionPanelFor(enabledWeightGeneratorAccordion)).getByLabelText(
        /weight generator stack hidden dim/i,
      ),
    ).toBeInTheDocument();
  });

  it("uses nested option accordions for boundary projector sections", async () => {
    installFetchMock({ schemaResponse: boundaryProjectorSchemaResponse() });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const inputAccordion = within(dialog).getByRole("button", {
      name: /^input boundary projector options section, 6 fields, 0 overrides/i,
    });
    const outputAccordion = within(dialog).getByRole("button", {
      name: /^output boundary projector options section, 4 fields, 0 overrides/i,
    });

    expect(inputAccordion).toBeEnabled();
    expect(outputAccordion).toBeEnabled();

    await user.click(inputAccordion);
    await user.click(outputAccordion);

    const inputPanel = accordionPanelFor(inputAccordion);
    const outputPanel = accordionPanelFor(outputAccordion);

    const disabledWeightGroup = within(inputPanel).getByRole("button", {
      name: /^weight boundary projector group, 3 fields, 0 overrides/i,
    });
    const inputWeightGroup = configFieldGroupFor(inputAccordion, "Weight");
    const inputWeightSwitch = within(inputWeightGroup).getByRole("switch", {
      name: /^input layer weight option$/i,
    });
    const outputWeightSwitch = within(
      configFieldGroupFor(outputAccordion, "Weight"),
    ).getByRole("switch", {
      name: /^output layer weight option$/i,
    });

    expect(disabledWeightGroup).toBeDisabled();
    expect(inputWeightGroup).toHaveClass("overflow-visible");
    expect(inputWeightSwitch).toHaveAttribute("aria-checked", "false");
    expect(outputWeightSwitch).toHaveAttribute("aria-checked", "false");
    expect(
      within(inputPanel).queryByRole("combobox", {
        name: /^input layer weight option$/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      within(inputPanel).queryByRole("textbox", {
        name: /^input layer weight decay warmup batches$/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      within(inputPanel).queryByRole("textbox", {
        name: /^input layer weight decay rate$/i,
      }),
    ).not.toBeInTheDocument();

    await user.click(inputWeightSwitch);

    const enabledWeightGroup = within(inputPanel).getByRole("button", {
      name: /^weight boundary projector group, 3 fields, 1 override/i,
    });
    const enabledWeightPanel = accordionPanelFor(enabledWeightGroup);
    const inputWeightOption = within(enabledWeightPanel).getByRole("combobox", {
      name: /^input layer weight option$/i,
    });
    const warmupBatchesInput = within(enabledWeightPanel).getByRole("textbox", {
      name: /^input layer weight decay warmup batches$/i,
    });
    const decayRateInput = within(enabledWeightPanel).getByRole("textbox", {
      name: /^input layer weight decay rate$/i,
    });

    expect(enabledWeightGroup).toBeEnabled();
    expect(enabledWeightGroup).toHaveAttribute("aria-expanded", "true");
    expect(inputWeightSwitch).toHaveAttribute("aria-checked", "true");
    expect(inputWeightOption).toHaveTextContent("DualModelDynamicWeightConfig");
    expect(warmupBatchesInput).toHaveAttribute("placeholder", "int");
    expect(decayRateInput).toHaveAttribute("placeholder", "float");

    await user.click(inputWeightOption);

    expect(
      screen.getByRole("option", { name: /^DualModelDynamicWeightConfig$/i }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /^None$/i })).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("option", { name: /^DualModelDynamicWeightConfig$/i }),
    );
    await user.click(inputWeightSwitch);

    expect(
      within(inputPanel).getByRole("button", {
        name: /^weight boundary projector group, 3 fields, 0 overrides/i,
      }),
    ).toBeDisabled();
    expect(
      within(inputPanel).queryByRole("combobox", {
        name: /^input layer weight option$/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      within(outputPanel).getByRole("button", {
        name: /^weight boundary projector group, 1 field, 0 overrides/i,
      }),
    ).toBeDisabled();
  });

  it("renders gate option enum selects in controlled gate sections", async () => {
    installFetchMock({ schemaResponse: gateOptionSchemaResponse() });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /^gate options section, 7 fields, 0 overrides/i,
    });
    const recurrentGateAccordion = within(dialog).getByRole("button", {
      name: /^recurrent gate options section, 7 fields, 0 overrides/i,
    });

    expect(
      within(sectionNav).getByRole("button", {
        name: /jump to recurrent gate options/i,
      }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", {
        name: /jump to recurrent gate stack options/i,
      }),
    ).toBeInTheDocument();
    expect(gateAccordion).toBeDisabled();
    expect(recurrentGateAccordion).toBeDisabled();
    expect(within(dialog).queryByLabelText(/^gate option$/i))
      .not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/^gate activation$/i))
      .not.toBeInTheDocument();
    expect(
      within(fullConfigSectionFor(gateAccordion)).queryByRole("button", {
        name: /gate stack options section/i,
      }),
    )
      .not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/^recurrent gate option$/i))
      .not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/^recurrent gate activation$/i))
      .not.toBeInTheDocument();

    await user.click(within(dialog).getByRole("switch", { name: /^gate flag$/i }));

    const enabledGateAccordion = within(dialog).getByRole("button", {
      name: /^gate options section, 7 fields, 1 override/i,
    });
    if (enabledGateAccordion.getAttribute("aria-expanded") !== "true") {
      await user.click(enabledGateAccordion);
    }
    const gateDirectGrid = directFieldGridFor(enabledGateAccordion);
    const gateModelStackAccordion = within(
      fullConfigSectionFor(enabledGateAccordion),
    ).getByRole("button", {
      name: /gate stack options section, 4 fields, 0 overrides/i,
    });
    const gateOption = within(gateDirectGrid).getByLabelText(/gate option/i);
    expect(gateOption).toHaveTextContent("MULTIPLIER");
    const gateActivation = within(gateDirectGrid).getByLabelText(/gate activation/i);
    expect(gateActivation).toHaveTextContent("SIGMOID");
    expect(within(gateDirectGrid).queryByLabelText(/gate stack hidden dim/i))
      .not.toBeInTheDocument();
    expect(
      within(accordionPanelFor(gateModelStackAccordion)).getByLabelText(
        /gate stack hidden dim/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(gateModelStackAccordion)).getByLabelText(
        /gate stack layer norm position/i,
      ),
    ).toBeInTheDocument();
    expectBooleanSegmentedControl(
      accordionPanelFor(gateModelStackAccordion),
      /gate stack bias flag/i,
    );
    expect(
      within(accordionPanelFor(gateModelStackAccordion)).getByLabelText(
        /gate stack num layers/i,
      ),
    ).toBeInTheDocument();

    const enabledGateSection = fullConfigSectionFor(enabledGateAccordion);
    expect(enabledGateSection).toHaveClass(
      "relative",
      "overflow-visible",
      "focus-within:z-30",
    );
    expect(enabledGateSection).not.toHaveClass("overflow-hidden");

    const gateOptionRoot = gateOption.parentElement;
    if (!(gateOptionRoot instanceof HTMLElement)) {
      throw new Error("Expected gate option dropdown root to render");
    }
    await user.click(gateOption);
    expect(gateOption).toHaveAttribute("aria-expanded", "true");
    const gateOptions = await within(gateOptionRoot).findByRole("listbox", {
      name: /gate option options/i,
    });
    expect(within(gateOptions).getByRole("option", { name: "ADDITION" }))
      .toBeInTheDocument();
    await user.click(within(gateOptions).getByRole("option", { name: "ADDITION" }));
    await waitFor(() => {
      expect(within(gateOptionRoot).queryByRole("listbox")).not.toBeInTheDocument();
    });
    expect(gateOption).toHaveTextContent("ADDITION");

    await user.click(
      within(dialog).getByRole("switch", { name: /^recurrent gate flag$/i }),
    );

    const enabledRecurrentGateAccordion = within(dialog).getByRole("button", {
      name: /^recurrent gate options section, 7 fields, 1 override/i,
    });
    if (enabledRecurrentGateAccordion.getAttribute("aria-expanded") !== "true") {
      await user.click(enabledRecurrentGateAccordion);
    }
    const recurrentGateDirectGrid = directFieldGridFor(
      enabledRecurrentGateAccordion,
    );
    const recurrentGateModelStackAccordion = within(dialog).getByRole("button", {
      name: /recurrent gate stack options section, 4 fields, 0 overrides/i,
    });
    const recurrentGateOption = within(recurrentGateDirectGrid).getByLabelText(
      /recurrent gate option/i,
    );
    const recurrentGateActivation = within(
      recurrentGateDirectGrid,
    ).getByLabelText(/recurrent gate activation/i);
    expect(recurrentGateOption).toHaveTextContent("MULTIPLIER");
    expect(recurrentGateActivation).toHaveTextContent("SIGMOID");
    expect(
      within(recurrentGateDirectGrid).queryByLabelText(
        /recurrent gate stack hidden dim/i,
      ),
    ).not.toBeInTheDocument();
    expect(
      within(accordionPanelFor(recurrentGateModelStackAccordion)).getByLabelText(
        /recurrent gate stack hidden dim/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(recurrentGateModelStackAccordion)).getByLabelText(
        /recurrent gate stack layer norm position/i,
      ),
    ).toBeInTheDocument();
    expectBooleanSegmentedControl(
      accordionPanelFor(recurrentGateModelStackAccordion),
      /recurrent gate stack bias flag/i,
    );
    expect(
      within(accordionPanelFor(recurrentGateModelStackAccordion)).getByLabelText(
        /recurrent gate stack num layers/i,
      ),
    ).toBeInTheDocument();
  });

  it("groups halting, memory, and recurrent stack prefixes into nested config accordions", async () => {
    installFetchMock({ schemaResponse: nestedControlledSchemaResponse() });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const haltingAccordion = within(dialog).getByRole("button", {
      name: /halting options section, 5 fields, 0 overrides/i,
    });
    const memoryAccordion = within(dialog).getByRole("button", {
      name: /memory options section, 9 fields, 0 overrides/i,
    });
    const recurrentAccordion = within(dialog).getByRole("button", {
      name: /recurrent layer options section, 9 fields, 0 overrides/i,
    });
    const haltingSection = fullConfigSectionFor(haltingAccordion);
    const memorySection = fullConfigSectionFor(memoryAccordion);
    const recurrentSection = fullConfigSectionFor(recurrentAccordion);

    expect(
      within(sectionNav).getByRole("button", {
        name: /jump to halting stack options/i,
      }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", {
        name: /jump to memory stack options/i,
      }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", {
        name: /jump to recurrent gate stack options/i,
      }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", {
        name: /jump to recurrent halting options/i,
      }),
    ).toBeInTheDocument();

    await user.click(within(dialog).getByRole("switch", { name: /halting flag/i }));
    await user.click(within(dialog).getByRole("switch", { name: /memory flag/i }));
    await user.click(within(dialog).getByRole("switch", { name: /recurrent flag/i }));

    const enabledHaltingAccordion = within(dialog).getByRole("button", {
      name: /halting options section, 5 fields, 1 override/i,
    });
    const enabledMemoryAccordion = within(dialog).getByRole("button", {
      name: /memory options section, 9 fields, 1 override/i,
    });
    const enabledRecurrentAccordion = within(dialog).getByRole("button", {
      name: /recurrent layer options section, 9 fields, 1 override/i,
    });
    const haltingDirectGrid = directFieldGridFor(enabledHaltingAccordion);
    const memoryDirectGrid = directFieldGridFor(enabledMemoryAccordion);
    const recurrentDirectGrid = directFieldGridFor(enabledRecurrentAccordion);
    const haltingStackAccordion = within(haltingSection).getByRole("button", {
      name: /halting stack options section, 3 fields, 0 overrides/i,
    });
    const memoryStackAccordion = within(memorySection).getByRole("button", {
      name: /memory stack options section, 5 fields, 0 overrides/i,
    });
    const recurrentGateAccordion = within(recurrentSection).getByRole("button", {
      name: /recurrent gate options section, 2 fields, 0 overrides/i,
    });
    const recurrentHaltingAccordion = within(recurrentSection).getByRole("button", {
      name: /recurrent halting options section, 5 fields, 0 overrides/i,
    });

    expect(haltingAccordion).toHaveAttribute("aria-expanded", "true");
    expect(memoryAccordion).toHaveAttribute("aria-expanded", "true");
    expect(recurrentAccordion).toHaveAttribute("aria-expanded", "true");
    expectHeaderControlBeforeMetric(haltingSection, "halting flag", "5 fields");
    expectHeaderControlBeforeMetric(memorySection, "memory flag", "9 fields");
    expectHeaderControlBeforeMetric(recurrentSection, "recurrent flag", "9 fields");
    expect(within(haltingDirectGrid).getByLabelText(/halting threshold/i))
      .toBeInTheDocument();
    expect(within(haltingDirectGrid).queryByLabelText(/halting stack hidden dim/i))
      .not.toBeInTheDocument();
    expect(within(haltingDirectGrid).queryByLabelText(/halting stack layer norm position/i))
      .not.toBeInTheDocument();
    expect(within(memoryDirectGrid).getByLabelText(/memory option/i))
      .toBeInTheDocument();
    expect(within(memoryDirectGrid).getByLabelText(/memory position option/i))
      .toBeInTheDocument();
    expect(
      within(memoryDirectGrid).getByLabelText(
        /memory test time training learning rate/i,
      ),
    ).toBeInTheDocument();
    expect(within(memoryDirectGrid).queryByLabelText(/memory stack hidden dim/i))
      .not.toBeInTheDocument();
    expect(within(memoryDirectGrid).queryByLabelText(/memory stack num layers/i))
      .not.toBeInTheDocument();
    expect(within(memoryDirectGrid).queryByLabelText(/memory stack activation/i))
      .not.toBeInTheDocument();
    expect(within(recurrentDirectGrid).getByLabelText(/recurrent max steps/i))
      .toBeInTheDocument();
    expect(within(recurrentDirectGrid).queryByLabelText(/recurrent gate stack hidden dim/i))
      .not.toBeInTheDocument();
    expect(within(recurrentDirectGrid).queryByLabelText(/recurrent halting threshold/i))
      .not.toBeInTheDocument();
    expect(fullConfigSectionGridFor(haltingStackAccordion)).not.toBe(
      fullConfigSectionGridFor(enabledHaltingAccordion),
    );
    expect(fullConfigSectionGridFor(memoryStackAccordion)).not.toBe(
      fullConfigSectionGridFor(enabledMemoryAccordion),
    );
    expect(fullConfigSectionGridFor(recurrentGateAccordion)).not.toBe(
      fullConfigSectionGridFor(enabledRecurrentAccordion),
    );

    expect(
      within(accordionPanelFor(haltingStackAccordion)).getByLabelText(
        /halting stack hidden dim/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(haltingStackAccordion)).getByLabelText(
        /halting stack layer norm position/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(memoryStackAccordion)).getByLabelText(
        /memory stack hidden dim/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(memoryStackAccordion)).getByLabelText(
        /memory stack layer norm position/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(memoryStackAccordion)).getByLabelText(
        /memory stack num layers/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(memoryStackAccordion)).getByLabelText(
        /memory stack activation/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(memoryStackAccordion)).getByLabelText(
        /memory stack dropout probability/i,
      ),
    ).toBeInTheDocument();
    expect(recurrentGateAccordion).toBeDisabled();
    expect(recurrentHaltingAccordion).toBeDisabled();
    expect(within(recurrentSection).getByRole("switch", {
      name: /recurrent gate flag/i,
    })).toBeInTheDocument();
    expect(within(recurrentSection).getByRole("switch", {
      name: /recurrent halting flag/i,
    })).toBeInTheDocument();

    await user.click(
      within(recurrentSection).getByRole("switch", {
        name: /recurrent gate flag/i,
      }),
    );
    await user.click(
      within(recurrentSection).getByRole("switch", {
        name: /recurrent halting flag/i,
      }),
    );

    const enabledRecurrentGateAccordion = within(recurrentSection).getByRole("button", {
      name: /recurrent gate options section, 2 fields, 1 override/i,
    });
    const enabledRecurrentHaltingAccordion = within(recurrentSection).getByRole(
      "button",
      {
        name: /recurrent halting options section, 5 fields, 1 override/i,
      },
    );
    const recurrentHaltingStackAccordion = within(recurrentSection).getByRole(
      "button",
      {
        name: /recurrent halting stack options section, 3 fields, 0 overrides/i,
      },
    );
    const recurrentGateModelStackAccordion = within(recurrentSection).getByRole(
      "button",
      {
        name: /recurrent gate stack options section, 1 field, 0 overrides/i,
      },
    );
    const recurrentHaltingDirectGrid = directFieldGridFor(
      enabledRecurrentHaltingAccordion,
    );

    expect(enabledRecurrentGateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(enabledRecurrentHaltingAccordion).toHaveAttribute("aria-expanded", "true");
    expect(recurrentGateModelStackAccordion).toHaveAttribute("aria-expanded", "true");
    expect(
      within(accordionPanelFor(recurrentGateModelStackAccordion)).getByLabelText(
        /recurrent gate stack hidden dim/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(recurrentHaltingDirectGrid).getByLabelText(
        /recurrent halting threshold/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(recurrentHaltingDirectGrid).queryByLabelText(
        /recurrent halting stack num layers/i,
      ),
    ).not.toBeInTheDocument();
    expect(
      within(recurrentHaltingDirectGrid).queryByLabelText(
        /recurrent halting stack hidden dim/i,
      ),
    ).not.toBeInTheDocument();
    expect(
      within(recurrentHaltingDirectGrid).queryByLabelText(
        /recurrent halting stack layer norm position/i,
      ),
    ).not.toBeInTheDocument();
    expect(
      within(accordionPanelFor(recurrentHaltingStackAccordion)).getByLabelText(
        /recurrent halting stack num layers/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(recurrentHaltingStackAccordion)).getByLabelText(
        /recurrent halting stack hidden dim/i,
      ),
    ).toBeInTheDocument();
    expect(
      within(accordionPanelFor(recurrentHaltingStackAccordion)).getByLabelText(
        /recurrent halting stack layer norm position/i,
      ),
    ).toBeInTheDocument();
    expectNoHeaderControlInAccordionBody(
      enabledRecurrentGateAccordion,
      "recurrent gate flag",
    );
    expectNoHeaderControlInAccordionBody(
      enabledRecurrentHaltingAccordion,
      "recurrent halting flag",
    );
  });

  it("renders stack booleans as two-option segmented controls", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          canonicalStackFixtureFields.find((field) => field.key === "stack_bias_flag")!,
          canonicalStackFixtureFields.find(
            (field) => field.key === "stack_apply_output_pipeline_flag",
          )!,
          stackFixtureField({
            key: "gate_stack_bias_flag",
            section: "Gate Stack Options",
            default: true,
            nullable: true,
          }),
          stackFixtureField({
            key: "gate_stack_apply_output_pipeline_flag",
            section: "Gate Stack Options",
            default: null,
            nullable: true,
          }),
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const layerAccordion = within(dialog).getByRole("button", {
      name: /^layer stack options section,/i,
    });
    if (layerAccordion.getAttribute("aria-expanded") !== "true") {
      await user.click(layerAccordion);
    }
    const layerPanel = accordionPanelFor(layerAccordion);
    expectBooleanSegmentedControl(layerPanel, /stack apply output pipeline flag/i);
    expectBooleanSegmentedControl(layerPanel, /stack bias flag/i);

    await user.click(within(dialog).getByRole("switch", { name: /gate flag/i }));

    const gateAccordion = within(dialog).getByRole("button", {
      name: /^gate options section, 3 fields, 1 override/i,
    });
    const gateModelStackAccordion = within(fullConfigSectionFor(gateAccordion))
      .getByRole("button", {
        name: /^gate stack options section, 2 fields, 0 overrides/i,
      });
    if (gateModelStackAccordion.getAttribute("aria-expanded") !== "true") {
      await user.click(gateModelStackAccordion);
    }
    const gateModelStackPanel = accordionPanelFor(gateModelStackAccordion);
    const gateBias = expectBooleanSegmentedControl(
      gateModelStackPanel,
      /gate stack bias flag/i,
    );
    const gatePipeline = expectBooleanSegmentedControl(
      gateModelStackPanel,
      /gate stack apply output pipeline flag/i,
    );

    expect(gateBias.on).toHaveAttribute("aria-checked", "true");
    expect(gateBias.off).toHaveAttribute("aria-checked", "false");
    expect(gatePipeline.on).toHaveAttribute("aria-checked", "false");
    expect(gatePipeline.off).toHaveAttribute("aria-checked", "true");

    await user.click(gatePipeline.on);
    expect(gatePipeline.on).toHaveAttribute("aria-checked", "true");
    let commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --gate-flag true --gate-stack-apply-output-pipeline-flag true",
    );
    await user.click(
      within(commandDialog).getByRole("button", {
        name: /close training command/i,
      }),
    );

    await user.click(gatePipeline.off);
    expect(gatePipeline.off).toHaveAttribute("aria-checked", "true");
    commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --gate-flag true --gate-stack-apply-output-pipeline-flag false",
    );
  });

  it("uses nested flag reasons for search results and auto-opens nested matches", async () => {
    installFetchMock({ schemaResponse: nestedControlledSchemaResponse() });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "recurrent gate stack hidden");
    let searchPopup = fullConfigSearchPopup(dialog);
    let recurrentGateRow = fullConfigSearchResultRow(
      searchPopup,
      /recurrent gate stack hidden dim/i,
    );

    expect(
      within(recurrentGateRow).getByRole("textbox", {
        name: /current value/i,
      }),
    ).toBeDisabled();
    expect(recurrentGateRow).toHaveTextContent(
      /enable recurrent flag before editing recurrent layer options/i,
    );

    await user.click(within(dialog).getByRole("switch", { name: /recurrent flag/i }));
    await user.click(search);

    searchPopup = fullConfigSearchPopup(dialog);
    recurrentGateRow = fullConfigSearchResultRow(
      searchPopup,
      /recurrent gate stack hidden dim/i,
    );
    expect(
      within(recurrentGateRow).getByRole("textbox", {
        name: /current value/i,
      }),
    ).toBeDisabled();
    expect(recurrentGateRow).toHaveTextContent(
      /enable recurrent gate flag before editing recurrent gate options/i,
    );

    const recurrentSection = fullConfigSectionFor(
      within(dialog).getByRole("button", {
        name: /recurrent layer options section, 3 fields, 1 override/i,
      }),
    );
    const recurrentGateAccordion = within(recurrentSection).getByRole("button", {
      name: /recurrent gate options section, 2 fields, 0 overrides/i,
    });

    expect(recurrentGateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(recurrentGateAccordion).toBeDisabled();
    expect(within(recurrentSection).getByRole("switch", {
      name: /recurrent gate flag/i,
    })).toBeInTheDocument();

    await user.click(
      within(recurrentSection).getByRole("switch", {
        name: /recurrent gate flag/i,
      }),
    );
    await user.click(
      within(dialog).getByRole("button", {
        name: /recurrent layer options section, 3 fields, 2 overrides/i,
      }),
    );
    await user.click(search);
    recurrentGateRow = fullConfigSearchResultRow(
      fullConfigSearchPopup(dialog),
      /recurrent gate stack hidden dim/i,
    );
    await user.click(
      within(recurrentGateRow).getByRole("button", {
        name: /recurrent gate stack hidden dim/i,
      }),
    );

    expect(search).toHaveValue("recurrent gate stack hidden dim");
    expect(
      within(dialog).getByRole("button", {
        name: /recurrent layer options section, 3 fields, 2 overrides/i,
      }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(
      within(recurrentSection).getByRole("button", {
        name: /recurrent gate options section, 2 fields, 1 override/i,
      }),
    ).toHaveAttribute("aria-expanded", "true");
    const recurrentGateModelStackAccordion = within(recurrentSection).getByRole(
      "button",
      {
        name: /recurrent gate stack options section, 1 field, 0 overrides/i,
      },
    );
    expect(recurrentGateModelStackAccordion).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(
      within(accordionPanelFor(recurrentGateModelStackAccordion)).getByLabelText(
        /recurrent gate stack hidden dim/i,
      ),
    ).toBeInTheDocument();
  });

  it("shows an accessible full config field search", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    expect(search).toHaveAttribute("aria-expanded", "false");
    expect(search).toHaveAttribute("aria-controls");
    expect(search).toHaveAttribute("aria-haspopup", "dialog");
    expect(search).not.toHaveAttribute("aria-activedescendant");

    await user.type(search, "hidden");

    expect(search).toHaveAttribute("aria-expanded", "true");
    expect(search).not.toHaveAttribute("aria-activedescendant");
    const searchPopup = fullConfigSearchPopup(dialog);
    const hiddenDimRow = fullConfigSearchResultRow(searchPopup, /hidden dim/i);

    expect(hiddenDimRow).not.toHaveTextContent(/default\s*256/i);
    expect(hiddenDimRow).not.toHaveTextContent(/current\s*256/i);
    expect(within(hiddenDimRow).getByRole("button", { name: /hidden dim/i }))
      .toBeInTheDocument();
    expect(
      within(hiddenDimRow).getByRole("textbox", { name: /current value/i }),
    ).toHaveValue("256");
  });

  it("lazy-loads full config field search results instead of showing a hidden-count footer", async () => {
    installFetchMock({ schemaResponse: gateOptionSchemaResponse() });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "gate");

    const searchPopup = fullConfigSearchPopup(dialog);
    expect(
      within(searchPopup).getAllByRole("group", {
        name: /config search result/i,
      }),
    ).toHaveLength(8);
    expect(
      within(searchPopup).queryByRole("button", {
        name: /recurrent gate stack hidden dim/i,
      }),
    ).not.toBeInTheDocument();
    expect(within(searchPopup).queryByText(/more matches/i)).not.toBeInTheDocument();

    makeScrollable(searchPopup);
    fireEvent.scroll(searchPopup);

    expect(
      within(searchPopup).getByRole("status", {
        name: /loading more config matches/i,
      }),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(
        fullConfigSearchResultRow(searchPopup, /recurrent gate stack hidden dim/i),
      ).toBeInTheDocument();
    });
    expect(within(searchPopup).queryByText(/more matches/i)).not.toBeInTheDocument();
  });

  it("filters full config cards and sidebar sections while typing", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");

    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", {
        name: /layer stack options section, 1 field, 0 overrides/i,
      }),
    ).toBeInTheDocument();
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });
    const sectionGrid = fullConfigSectionGridFor(layerAccordion);
    expectFullConfigSectionGrid(sectionGrid);
    expect(sectionGrid.children).toHaveLength(1);
    expect(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to gate options/i }),
    ).not.toBeInTheDocument();
    expect(within(sectionNav).getByLabelText("1 field")).toHaveTextContent("1");
  });

  it("finds full config fields by flag and key", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "--gate-flag");

    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByText("--gate-flag")).not.toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "gate_flag");

    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", {
        name: /gate options section, 1 field, 0 overrides/i,
      }),
    ).toBeInTheDocument();
  });

  it("does not match full config fields by current or default value text", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "256");
    const searchPopup = fullConfigSearchPopup(dialog);

    expect(within(searchPopup).getByText("No matching fields")).toBeInTheDocument();
    expect(
      within(searchPopup).queryByRole("group", { name: /hidden dim/i }),
    ).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(sectionNav).getByText("No matching sections")).toBeInTheDocument();
  });

  it("selects a dropdown field and filters to exactly that field", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "activation");
    const searchPopup = fullConfigSearchPopup(dialog);
    const stackActivationRow = fullConfigSearchResultRow(searchPopup, /stack activation/i);
    await user.click(
      within(stackActivationRow).getByRole("button", { name: /stack activation/i }),
    );

    expect(search).toHaveValue("stack activation");
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();
    expect(
      within(dialog).queryByRole("dialog", { name: /matching config fields/i }),
    ).not.toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("edits a numeric field from full config search results", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const searchPopup = fullConfigSearchPopup(dialog);
    const hiddenDimRow = fullConfigSearchResultRow(searchPopup, /hidden dim/i);
    const hiddenDimSearchInput = within(hiddenDimRow).getByRole("textbox", {
      name: /current value/i,
    });

    expect(hiddenDimRow).not.toHaveTextContent(/default\s*256/i);
    expect(hiddenDimRow).not.toHaveTextContent(/current\s*256/i);

    await user.clear(hiddenDimSearchInput);
    await user.type(hiddenDimSearchInput, "128");

    expect(fullConfigSearchPopup(dialog)).toBeInTheDocument();
    expect(hiddenDimRow).toHaveTextContent(/current\s*128/i);
    expect(hiddenDimRow).not.toHaveTextContent(/default\s*256/i);
    expect(within(hiddenDimRow).getByText("override")).toHaveClass("text-violet");
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue("128");

    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: { stack_hidden_dim: "128" },
      });
    });

    await user.click(search);
    const reopenedHiddenDimRow = fullConfigSearchResultRow(
      fullConfigSearchPopup(dialog),
      /hidden dim/i,
    );
    const reopenedHiddenDimSearchInput = within(reopenedHiddenDimRow).getByRole(
      "textbox",
      {
        name: /current value/i,
      },
    );

    await user.clear(reopenedHiddenDimSearchInput);
    await user.type(reopenedHiddenDimSearchInput, "256");

    expect(fullConfigSearchPopup(dialog)).toBeInTheDocument();
    await waitFor(() => {
      expect(within(reopenedHiddenDimRow).queryByText("override")).not.toBeInTheDocument();
    });
    expect(reopenedHiddenDimRow).not.toHaveTextContent(/current\s*256/i);
    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue("256");
    expect(within(dialog).getAllByLabelText("0 overrides").length).toBeGreaterThan(0);

    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      });
    });
  });

  it("edits enum and bool fields from full config search results", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "activation");
    let searchPopup = fullConfigSearchPopup(dialog);
    const stackActivationRow = fullConfigSearchResultRow(searchPopup, /stack activation/i);
    const stackActivationSelect = within(stackActivationRow).getByRole("combobox", {
      name: /current value/i,
    });

    await selectSearchableDropdownOption(
      user,
      stackActivationSelect,
      "RELU",
      "RELU",
    );

    expect(stackActivationRow).toHaveTextContent(/current\s*RELU/i);
    expect(within(dialog).getByLabelText(/stack activation/i))
      .toHaveTextContent("RELU");

    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));
    await user.type(search, "gate");
    searchPopup = fullConfigSearchPopup(dialog);
    const gateFlagRow = fullConfigSearchResultRow(searchPopup, /gate flag/i);
    const gateFlagControl = expectBooleanSegmentedControl(
      gateFlagRow,
      /current value/i,
    );

    await user.click(gateFlagControl.on);

    expect(gateFlagRow).toHaveTextContent(/current\s*true/i);
    expect(within(dialog).getByRole("switch", { name: /gate flag/i }))
      .toHaveAttribute("aria-checked", "true");
    expect(within(dialog).getAllByLabelText("2 overrides").length).toBeGreaterThan(0);

    await user.click(gateFlagControl.off);

    await waitFor(() => {
      expect(within(gateFlagRow).queryByText("override")).not.toBeInTheDocument();
    });
    expect(gateFlagRow).not.toHaveTextContent(/current\s*false/i);
    expect(within(dialog).getByRole("switch", { name: /gate flag/i }))
      .toHaveAttribute("aria-checked", "false");
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
  });

  it("resets a modified select field from full config search results", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "activation");
    const searchPopup = fullConfigSearchPopup(dialog);
    const stackActivationRow = fullConfigSearchResultRow(searchPopup, /stack activation/i);
    const stackActivationSelect = within(stackActivationRow).getByRole("combobox", {
      name: /current value/i,
    });
    const selectRoot = stackActivationSelect.parentElement;

    if (!(selectRoot instanceof HTMLElement)) {
      throw new Error("Expected stack activation select root");
    }

    await selectSearchableDropdownOption(
      user,
      stackActivationSelect,
      "RELU",
      "RELU",
    );

    const resetButton = within(stackActivationRow).getByRole("button", {
      name: /reset search result override/i,
    });

    expect(stackActivationRow).toHaveTextContent(/current\s*RELU/i);
    expect(stackActivationSelect).toHaveTextContent("RELU");
    expect(selectRoot).toHaveClass("z-20");
    expect(resetButton).toHaveClass("absolute", "right-1", "z-40");

    await user.click(resetButton);

    await waitFor(() => {
      expect(within(stackActivationRow).queryByText("override")).not.toBeInTheDocument();
    });
    expect(stackActivationRow).not.toHaveTextContent(/current\s*GELU/i);
    expect(stackActivationSelect).toHaveTextContent("GELU");
    expect(within(dialog).getByLabelText(/stack activation/i)).toHaveTextContent("GELU");
    expect(
      within(stackActivationRow).queryByRole("button", {
        name: /reset search result override/i,
      }),
    ).not.toBeInTheDocument();
    expect(within(dialog).getAllByLabelText("0 overrides").length).toBeGreaterThan(0);
  });

  it("clears full config search and restores all sections and fields", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));

    expect(search).toHaveValue("");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", { name: /jump to gate options/i }),
    ).toBeInTheDocument();
  });

  it("shows empty states when full config search has no matches", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "missing option");

    const noResults = within(dialog).getByText('No config fields match "missing option".');
    expect(noResults).toBeInTheDocument();
    expect(noResults).not.toHaveClass("md:col-span-2");
    expect(noResults).not.toHaveClass("2xl:col-span-3");
    const sectionGrid = noResults.parentElement;
    expect(sectionGrid).toBeInstanceOf(HTMLElement);
    expectFullConfigSectionGrid(sectionGrid as HTMLElement);
    expect(within(sectionNav).getByText("No matching sections")).toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to layer stack options/i }),
    ).not.toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to gate options/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps sidebar jumps and toggles working while full config search is filtered", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });

    await user.click(within(sectionNav).getByRole("button", { name: /^close all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();

    await user.click(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("keeps filtered accordion metric pills non-tabbable and number-only", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });
    const layerSection = fullConfigSectionFor(layerAccordion);

    expect(layerAccordion).not.toHaveTextContent(/1 field|0 overrides/i);
    expect(within(layerSection).getByLabelText("1 field")).toHaveTextContent("1");
    expect(within(layerSection).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(within(layerSection).getByLabelText("0 overrides")).toHaveTextContent("0");
    expect(within(layerSection).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
  });

  it("shows popup config metric tooltips on hover and focus", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const fieldMetric = within(dialog).getByLabelText("4 fields");
    const overrideMetric = within(dialog).getAllByLabelText("0 overrides")[0];

    expect(fieldMetric).toHaveAttribute("tabindex", "0");
    expect(overrideMetric).toHaveAttribute("tabindex", "0");

    await user.click(fieldMetric);
    expect(within(dialog).getByRole("tooltip")).toHaveTextContent("Fields");

    await user.click(within(dialog).getByLabelText(/hidden dim/i));
    expect(within(dialog).queryByRole("tooltip")).not.toBeInTheDocument();

    await user.hover(overrideMetric);
    expect(within(dialog).getByRole("tooltip")).toHaveTextContent("Overrides");

    await user.unhover(overrideMetric);
    expect(within(dialog).queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("global section toggle closes and opens every popup config accordion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate options section, 1 field, 0 overrides/i,
    });

    const closeAllButton = within(sectionNav).getByRole("button", { name: /^close all$/i });
    expect(closeAllButton).toBeInTheDocument();
    expect(closeAllButton).toHaveClass("whitespace-nowrap", "min-w-[5.75rem]");

    await user.click(closeAllButton);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(sectionNav).getByRole("button", { name: /^open all$/i })).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toBeDisabled();
    expect(within(sectionNav).getByRole("button", { name: /^close all$/i })).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("global section toggle opens every popup config accordion from a partial state", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate options section, 1 field, 0 overrides/i,
    });

    await user.click(layerAccordion);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toBeDisabled();
    expect(within(sectionNav).getByRole("button", { name: /^open all$/i })).toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(sectionNav).getByRole("button", { name: /^close all$/i })).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("collapsing a popup config section hides its field controls", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });

    await user.click(layerAccordion);

    const layerPanel = accordionPanelFor(layerAccordion);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(layerAccordion).toHaveClass("bg-white/[0.025]");
    expect(layerPanel).toHaveAttribute("hidden");
    expect(layerPanel).not.toHaveClass("grid", "px-3", "py-3");
    expect(layerAccordion.closest("section")).toHaveClass(
      "rounded-[10px]",
      "border-line-soft",
      "bg-white/[0.012]",
    );
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("sidebar section clicks reopen collapsed sections and scroll to them", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await user.click(layerAccordion);
    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");

    await user.click(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("sidebar section triggers toggle their matching accordion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await user.click(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(
      within(sectionNav).getByRole("button", { name: /open layer stack options/i }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(scrollIntoViewMock).not.toHaveBeenCalled();

    await user.click(
      within(sectionNav).getByRole("button", { name: /open layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).not.toHaveBeenCalled();
  });

  it("editing a popup field updates overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    const hiddenDimInput = await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const hiddenDimRow = configFieldRowFor(hiddenDimInput);
    const overrideBadge = within(hiddenDimRow).getByText("override");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 1 override/i,
    });
    const layerSection = fullConfigSectionFor(layerAccordion);
    const layerNavRow = fullConfigSectionNavRowFor(
      sectionNav,
      /jump to layer stack options/i,
    );

    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
    expect(within(dialog).queryByText("1 override")).not.toBeInTheDocument();
    expect(layerAccordion).toHaveClass("bg-violet/[0.08]", "hover:bg-violet/[0.12]");
    expect(within(layerSection).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerSection).queryByText("1 preset")).not.toBeInTheDocument();
    expect(layerSection).toHaveClass("border-violet/35", "bg-violet/[0.06]");
    expect(layerSection).not.toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(layerNavRow).toHaveClass(
      "border-violet/30",
      "bg-violet/[0.055]",
      "hover:bg-violet/15",
    );
    expect(layerNavRow).not.toHaveClass("border-amber/30", "bg-amber/[0.055]");
    expect(within(layerNavRow).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerNavRow).queryByText("1 preset")).not.toBeInTheDocument();
    expect(overrideBadge).toBeInTheDocument();
    expect(overrideBadge).toHaveClass("text-violet");
    expect(hiddenDimRow).toHaveClass("border-violet/40");
    expect(hiddenDimRow).not.toHaveClass("border-amber/55", "bg-amber/[0.055]");
    expect(within(hiddenDimRow).queryByText("preset")).not.toBeInTheDocument();
    expect(within(dialog).queryByText(/\d+ preset/i)).not.toBeInTheDocument();
    expect(within(dialog).getAllByText("stack hidden dim")).toHaveLength(1);
  });

  it("clears only the reverted popup field override when it returns to default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    let hiddenDimInput = await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const stackLayersInput = await typeConfigFieldValue(
      user,
      dialog,
      /stack num layers/i,
      "7",
    );

    expect(within(dialog).getAllByLabelText("2 overrides").length).toBeGreaterThan(0);

    hiddenDimInput = await typeConfigFieldValue(user, dialog, /hidden dim/i, "256");
    const hiddenDimRow = configFieldRowFor(hiddenDimInput);
    const stackLayersRow = configFieldRowFor(stackLayersInput);

    await waitFor(() => {
      expect(within(hiddenDimRow).queryByText("override")).not.toBeInTheDocument();
    });
    expect(hiddenDimInput).toHaveValue("256");
    expect(hiddenDimRow).not.toHaveClass("border-violet/40");
    expect(stackLayersInput).toHaveValue("7");
    expect(within(stackLayersRow).getByText("override")).toHaveClass("text-violet");
    expect(stackLayersRow).toHaveClass("border-violet/40");
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
    expect(
      within(dialog).getByRole("button", {
        name: /layer stack options section, 3 fields, 1 override/i,
      }),
    ).toBeInTheDocument();
  });

  it("clears section-header boolean override styling when toggled back to default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const gateSwitch = within(dialog).getByRole("switch", { name: /^gate flag$/i });

    await user.click(gateSwitch);

    const enabledGateAccordion = within(dialog).getByRole("button", {
      name: /gate options section, 1 field, 1 override/i,
    });
    const gateSection = fullConfigSectionFor(enabledGateAccordion);

    expect(gateSwitch).toHaveAttribute("aria-checked", "true");
    expect(within(gateSection).getByText("override")).toHaveClass("text-violet");
    expect(within(gateSection).getByLabelText("1 override")).toHaveClass("text-violet");

    await user.click(gateSwitch);

    await waitFor(() => {
      expect(
        within(dialog).getByRole("button", {
          name: /gate options section, 1 field, 0 overrides/i,
        }),
      ).toBeDisabled();
    });
    expect(gateSwitch).toHaveAttribute("aria-checked", "false");
    expect(within(gateSection).queryByText("override")).not.toBeInTheDocument();
    expect(within(gateSection).getByLabelText("0 overrides")).not.toHaveClass("text-violet");
    expect(within(dialog).queryAllByLabelText("1 override")).toHaveLength(0);
  });

  it("highlights a section gradient when it has an override and a preset-owned field", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: schemaResponse.fields.map((field) =>
          field.key === "stack_activation"
            ? {
                ...field,
                locked: true,
                lockedValue: "GELU",
                lockedReason:
                  "Locked by the ACTIVATION preset because this preset fixes stack activation.",
              }
            : field,
        ),
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");

    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 1 override, 1 preset/i,
    });
    const layerSection = fullConfigSectionFor(layerAccordion);
    const layerNavRow = fullConfigSectionNavRowFor(
      sectionNav,
      /jump to layer stack options/i,
    );

    expect(layerAccordion).toHaveClass(
      "bg-[linear-gradient(90deg,rgba(255,209,102,0.12),rgba(167,139,250,0.13))]",
      "hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.16),rgba(167,139,250,0.17))]",
    );
    expect(within(layerSection).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerSection).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerSection).toHaveClass(
      "border-amber/35",
      "bg-[linear-gradient(135deg,rgba(255,209,102,0.075),rgba(167,139,250,0.105))]",
      "ring-violet/25",
    );
    expect(layerSection).not.toHaveClass("bg-amber/[0.045]", "bg-violet/[0.06]");
    expect(layerNavRow).toHaveClass(
      "border-amber/35",
      "bg-[linear-gradient(90deg,rgba(255,209,102,0.075),rgba(167,139,250,0.095))]",
      "hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.11),rgba(167,139,250,0.13))]",
      "ring-violet/20",
    );
    expect(within(layerNavRow).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerNavRow).getByText("1 preset")).toHaveClass("text-amber");
  });

  it("renders stack layer count as an editable text input with inline reset", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const stackLayersInput = within(dialog).getByLabelText(/stack num layers/i);
    const stackLayersRow = configFieldRowFor(stackLayersInput);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(stackLayersInput).toHaveAttribute("type", "text");
    expect(stackLayersInput).toHaveAttribute("inputmode", "numeric");
    expect(stackLayersInput).toHaveValue("5");
    expect(
      within(stackLayersRow).queryByRole("button", {
        name: /reset field override/i,
      }),
    ).not.toBeInTheDocument();

    await user.clear(stackLayersInput);
    await user.type(stackLayersInput, "7");

    const resetButton = within(stackLayersRow).getByRole("button", {
      name: /reset field override/i,
    });

    expect(stackLayersInput).toHaveValue("7");
    expect(stackLayersInput).toHaveClass("pr-11");
    expect(resetButton).toBeEnabled();
    expect(resetButton).toHaveClass("absolute", "right-1", "top-1/2");
    resetButton.focus();
    expect(resetButton).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(stackLayersInput).toHaveValue("5");
    expect(
      within(stackLayersRow).queryByRole("button", {
        name: /reset field override/i,
      }),
    ).not.toBeInTheDocument();
    expect(within(dialog).getAllByLabelText("0 overrides").length).toBeGreaterThan(0);
  });

  it("inline field reset restores only one modified popup field", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const hiddenDimInput = await typeConfigFieldValue(
      user,
      dialog,
      /hidden dim/i,
      "128",
    );
    const stackLayersInput = await typeConfigFieldValue(
      user,
      dialog,
      /stack num layers/i,
      "7",
    );
    const hiddenDimRow = configFieldRowFor(hiddenDimInput);
    const stackLayersRow = configFieldRowFor(stackLayersInput);
    const hiddenResetButton = within(hiddenDimRow).getByRole("button", {
      name: /reset field override/i,
    });

    expect(within(dialog).getAllByLabelText("2 overrides").length).toBeGreaterThan(0);
    expect(
      within(stackLayersRow).getByRole("button", {
        name: /reset field override/i,
      }),
    ).toBeEnabled();

    await user.click(hiddenResetButton);

    expect(hiddenDimInput).toHaveValue("256");
    expect(stackLayersInput).toHaveValue("7");
    expect(
      within(hiddenDimRow).queryByRole("button", { name: /reset field override/i }),
    ).not.toBeInTheDocument();
    expect(
      within(stackLayersRow).getByRole("button", {
        name: /reset field override/i,
      }),
    ).toBeEnabled();
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
  });

  it("popup Update Preview posts selected model, preset, and overrides", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: { stack_hidden_dim: "128" },
      });
    });
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
  });

  it("opens a training command popup without closing full config", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandDialog).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline",
    );
  });

  it("uses the current selected preset and omits --config when there are no overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await selectTargetOption(user, "preset", "recurrent-gating-halting");
    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset recurrent-gating-halting",
    );
    expect((commandField(commandDialog) as HTMLTextAreaElement).value).not.toContain("--config");
  });

  it("includes live overrides in display order before Update Preview", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await selectSearchableDropdownOption(
      user,
      within(dialog).getByLabelText(/stack activation/i),
      "RELU",
      "RELU",
    );
    await user.click(within(dialog).getByRole("switch", { name: /gate flag/i }));

    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --stack-hidden-dim 128 --stack-activation RELU --gate-flag true",
    );
  });

  it("keeps invalid numeric text as a draft override for preview and command generation", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const hiddenDimInput = within(dialog).getByLabelText(/hidden dim/i);

    await user.clear(hiddenDimInput);
    await user.type(hiddenDimInput, "abc");

    expect(hiddenDimInput).toHaveValue("abc");
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);

    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: { stack_hidden_dim: "abc" },
      });
    });

    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --stack-hidden-dim abc",
    );
  });

  it("updates the training command after resetting overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    let commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --stack-hidden-dim 128",
    );

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));
    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));
    commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline",
    );
  });

  it("shell-quotes override values and omits default-equivalent nullable empty overrides", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "dropout_schedule",
            configKey: "DROPOUT_SCHEDULE",
            flag: "--dropout-schedule",
            label: "dropout schedule",
            section: "Layer Stack Options",
            type: "enum",
            default: null,
            nullable: true,
            choices: ["cosine decay"],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const scheduleInput = within(dialog).getByLabelText(/dropout schedule/i);
    await selectSearchableDropdownOption(
      user,
      scheduleInput,
      "cosine decay",
      "cosine decay",
    );
    let commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --config --dropout-schedule 'cosine decay'",
    );

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));
    await selectSearchableDropdownOption(user, scheduleInput, "None", "None");
    commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline",
    );
  });

  it("copies the exact training command to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    installFetchMock();
    renderViewer();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const commandDialog = await openTrainingCommand(user, dialog);
    const expectedCommand = "source experiment.sh --model-type linears --model linear --preset baseline --config --stack-hidden-dim 128";

    await user.click(within(commandDialog).getByRole("button", { name: /copy command/i }));

    expect(writeText).toHaveBeenCalledWith(expectedCommand);
    expect(within(commandDialog).getByRole("status")).toHaveTextContent("Command copied");
  });

  it("closing the training command popup leaves the full config dialog open", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));

    expect(screen.queryByRole("dialog", { name: /training command/i })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
  });

  it("popup Reset Overrides clears override state", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const hiddenInput = within(dialog).getByLabelText(/hidden dim/i);

    await user.clear(hiddenInput);
    await user.type(hiddenInput, "128");
    expect(hiddenInput).toHaveValue("128");

    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));

    expect(hiddenInput).toHaveValue("256");
    expect(within(dialog).getAllByLabelText("0 overrides").length).toBeGreaterThan(0);
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
  });

  it("close button removes the full config dialog", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });

    await user.click(within(dialog).getByRole("button", { name: /close full config/i }));

    expect(screen.queryByRole("dialog", { name: /full configuration/i })).not.toBeInTheDocument();
  });

});
