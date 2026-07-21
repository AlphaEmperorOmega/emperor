import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn

from emperor.neuron import (
    AxonsConfig,
    NeuronCluster,
    NeuronClusterConfig,
    NeuronClusterMonitorCallback,
    NeuronClusterOptimizerSyncCallback,
    NeuronClusterTrace,
    NeuronClusterTraceStep,
    NeuronConfig,
    NucleusConfig,
    TerminalConfig,
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)
from unit.test_neuron import (
    NeuronTestCase,
    ScriptedNeuron,
    ScriptedSampler,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "Axons",
    "AxonsConfig",
    "Neuron",
    "NeuronCluster",
    "NeuronClusterConfig",
    "NeuronClusterMonitorCallback",
    "NeuronClusterOptimizerSyncCallback",
    "NeuronClusterTrace",
    "NeuronClusterTraceStep",
    "NeuronConfig",
    "Nucleus",
    "NucleusConfig",
    "Terminal",
    "TerminalConfig",
    "TerminalConnectionShapeOptions",
    "TerminalRangeOptions",
    "TerminalZAxisOffsetOptions",
)

EXPECTED_OWNERS = {
    "Axons": "emperor.neuron._parts",
    "AxonsConfig": "emperor.neuron._config",
    "Neuron": "emperor.neuron._parts",
    "NeuronCluster": "emperor.neuron._cluster.model",
    "NeuronClusterConfig": "emperor.neuron._config",
    "NeuronClusterMonitorCallback": "emperor.neuron._monitoring.callback",
    "NeuronClusterOptimizerSyncCallback": "emperor.neuron._optimizer_sync",
    "NeuronClusterTrace": "emperor.neuron._trace",
    "NeuronClusterTraceStep": "emperor.neuron._trace",
    "NeuronConfig": "emperor.neuron._config",
    "Nucleus": "emperor.neuron._parts",
    "NucleusConfig": "emperor.neuron._config",
    "Terminal": "emperor.neuron._parts",
    "TerminalConfig": "emperor.neuron._config",
    "TerminalConnectionShapeOptions": "emperor.neuron._options",
    "TerminalRangeOptions": "emperor.neuron._options",
    "TerminalZAxisOffsetOptions": "emperor.neuron._options",
}

PRIVATE_MODULES = (
    "emperor.neuron._config",
    "emperor.neuron._options",
    "emperor.neuron._parts",
    "emperor.neuron._trace",
    "emperor.neuron._validation",
    "emperor.neuron._optimizer_sync",
    "emperor.neuron._monitoring",
    "emperor.neuron._monitoring.callback",
    "emperor.neuron._monitoring.diagnostics",
    "emperor.neuron._cluster",
    "emperor.neuron._cluster.model",
    "emperor.neuron._cluster.topology",
    "emperor.neuron._cluster.state",
    "emperor.neuron._cluster.recurrent_routes",
    "emperor.neuron._cluster.beam_routes",
    "emperor.neuron._cluster.plasticity",
    "emperor.neuron._cluster.checkpointing",
)

NUCLEUS_CONFIG_FIELDS = ("model_config",)
AXONS_CONFIG_FIELDS = ("memory_config",)
TERMINAL_CONFIG_FIELDS = (
    "input_dim",
    "x_axis_position",
    "y_axis_position",
    "z_axis_position",
    "xy_axis_range",
    "z_axis_range",
    "z_axis_offset",
    "sampler_config",
    "connection_shape",
)
NEURON_CONFIG_FIELDS = (
    "nucleus_config",
    "axons_config",
    "terminal_config",
    "coordinate_embedding_flag",
)
CLUSTER_CONFIG_FIELDS = (
    "x_axis_total_neurons",
    "y_axis_total_neurons",
    "z_axis_total_neurons",
    "initial_x_axis_total_neurons",
    "initial_y_axis_total_neurons",
    "initial_z_axis_total_neurons",
    "entry_sampler_config",
    "max_steps",
    "beam_width",
    "growth_threshold",
    "growth_cooldown_steps",
    "max_total_growths",
    "growth_warmup_steps",
    "pruning_threshold",
    "escape_driven_growth_flag",
    "mitosis_initialization_flag",
    "halting_config",
    "neuron_config",
)
TRACE_STEP_FIELDS = (
    "probabilities",
    "selected_coordinates",
    "valid_mask",
    "escape_mask",
    "chosen_branch_indices",
    "halt_mask",
    "active_mask",
)
TRACE_FIELDS = (
    "input_shape",
    "entry_coordinates",
    "entry_probabilities",
    "entry_selected_coordinates",
    "entry_valid_mask",
    "entry_escape_mask",
    "entry_chosen_branch_indices",
    "entry_halt_mask",
    "entry_active_mask",
    "steps",
)

NEURON_TOPOLOGY = (
    ("batch_counter", (), torch.int64),
    ("atrophy_counter", (), torch.int64),
    ("nucleus.model.weight", (4, 4), torch.float32),
    ("terminal.sampler.sampler_model.default_loss", (), torch.float32),
    (
        "terminal.sampler.sampler_model.auxiliary_loss_model.default_loss",
        (),
        torch.float32,
    ),
    (
        "terminal.sampler.router.model.layers.0.model.weight_params",
        (4, 18),
        torch.float32,
    ),
    (
        "terminal.sampler.router.model.layers.0.model.bias_params",
        (18,),
        torch.float32,
    ),
)

CLUSTER_TOPOLOGY = (
    ("cluster.neuron_1_1_1.batch_counter", (), torch.int64),
    ("cluster.neuron_1_1_1.atrophy_counter", (), torch.int64),
    ("cluster.neuron_1_1_1.nucleus.model.weight", (4, 4), torch.float32),
    (
        "cluster.neuron_1_1_1.terminal.sampler.sampler_model.default_loss",
        (),
        torch.float32,
    ),
    (
        "cluster.neuron_1_1_1.terminal.sampler.sampler_model."
        "auxiliary_loss_model.default_loss",
        (),
        torch.float32,
    ),
    (
        "cluster.neuron_1_1_1.terminal.sampler.router.model.layers.0."
        "model.weight_params",
        (4, 18),
        torch.float32,
    ),
    (
        "cluster.neuron_1_1_1.terminal.sampler.router.model.layers.0.model.bias_params",
        (18,),
        torch.float32,
    ),
    ("entry_sampler.sampler_model.default_loss", (), torch.float32),
    (
        "entry_sampler.sampler_model.auxiliary_loss_model.default_loss",
        (),
        torch.float32,
    ),
    (
        "entry_sampler.router.model.layers.0.model.weight_params",
        (4, 1),
        torch.float32,
    ),
    (
        "entry_sampler.router.model.layers.0.model.bias_params",
        (1,),
        torch.float32,
    ),
)

NEURON_RNG_DIGEST = "e06a4f0a50552801f019d081e8abb2a5b62506eddf44e06799d462d088eed7ab"
NEURON_OUTPUT_DIGEST = (
    "f2d1b2318747f51fc7e713d018a66fce77d02074078de65929189a2865375046"
)
CLUSTER_RNG_DIGEST = "0c8a84abf80270c9c19d9564fa63bb9e0314b5a8d843c7c0d50b5a4d12c9485b"
CLUSTER_TRACE_DIGEST = (
    "20dbd2eff718e4d736b297f18f115741867121437e12c8ac58c5011f30f65069"
)
GROWTH_OUTPUT_DIGEST = (
    "f8c3c147d86bc6b33cd045ef6a9225274753c2a333a85c685168dde23c8788a5"
)
GROWTH_STATE_DIGEST = "fb1a0212086e8ad262b06ab6f1bad203d94afa732e6d7c521a6196fe89542e92"


def _input(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        (
            (0.25, -0.50, 0.75, 1.00),
            (-1.00, 0.50, 0.125, -0.25),
        ),
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def _digest(*values: torch.Tensor | None) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value is not None:
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _topology(model: nn.Module):
    return tuple(
        (name, tuple(value.shape), value.dtype)
        for name, value in model.state_dict().items()
    )


def _trace_tensors(trace: NeuronClusterTrace) -> tuple[torch.Tensor, ...]:
    tensors = [
        trace.entry_coordinates,
        trace.entry_probabilities,
        trace.entry_selected_coordinates,
        trace.entry_valid_mask,
        trace.entry_escape_mask,
        trace.entry_chosen_branch_indices,
        trace.entry_halt_mask,
        trace.entry_active_mask,
    ]
    for step in trace.steps:
        tensors.extend(
            (
                step.probabilities,
                step.selected_coordinates,
                step.valid_mask,
                step.escape_mask,
                step.chosen_branch_indices,
                step.halt_mask,
                step.active_mask,
            )
        )
    return tuple(tensors)


class TestNeuronPublicInterface(NeuronTestCase):
    def cluster_config(
        self,
        *,
        x_axis_total_neurons: int = 1,
        initial_x_axis_total_neurons: int | None = None,
        growth_threshold: int | None = None,
        pruning_threshold: int | None = None,
    ) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=x_axis_total_neurons,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=initial_x_axis_total_neurons,
            initial_y_axis_total_neurons=(
                1 if initial_x_axis_total_neurons is not None else None
            ),
            initial_z_axis_total_neurons=(
                1 if initial_x_axis_total_neurons is not None else None
            ),
            max_steps=1,
            growth_threshold=growth_threshold,
            pruning_threshold=pruning_threshold,
            neuron_config=self.full_sampler_neuron_config(),
        )

    def scripted_cluster(self, *, beam_width: int | None) -> NeuronCluster:
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            beam_width=beam_width,
            growth_threshold=None,
            neuron_config=NeuronConfig(
                nucleus_config=NucleusConfig(
                    model_config=self.projection_config(
                        input_dim=1,
                        output_dim=1,
                        scale=0.25,
                    )
                ),
                axons_config=AxonsConfig(memory_config=None),
                terminal_config=self.terminal_config(input_dim=1),
            ),
        ).build()
        model.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )
        return model

    def test_exact_exports_resolve_lazily_from_their_owning_modules(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import importlib
import json
import sys

import emperor.neuron as neuron

private_modules = {PRIVATE_MODULES!r}
before = {{name: name in sys.modules for name in private_modules}}
runtime_before = {{
    "emperor.experts": "emperor.experts" in sys.modules,
    "lightning": "lightning" in sys.modules,
    "torch": "torch" in sys.modules,
}}
private_packages = {{}}
for module_name in ("emperor.neuron._cluster", "emperor.neuron._monitoring"):
    module = importlib.import_module(module_name)
    private_packages[module_name] = sorted(
        name for name in vars(module) if not name.startswith("_")
    )

import torch

torch.manual_seed(73)
expected_next_values = torch.randn(8)
torch.manual_seed(73)
owners = {{name: getattr(neuron, name).__module__ for name in neuron.__all__}}
actual_next_values = torch.randn(8)

print(json.dumps({{
    "all": neuron.__all__,
    "before": before,
    "owners": owners,
    "private_packages": private_packages,
    "private_exports": {{
        name: hasattr(neuron, name)
        for name in (
            "NeuronClusterRouteState",
            "NeuronClusterValidator",
            "NeuronValidationMixin",
            "TerminalValidator",
        )
    }},
    "rng_unchanged": torch.equal(expected_next_values, actual_next_values),
    "runtime_before": runtime_before,
}}))
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(
                    Path(tempfile.gettempdir()) / "matplotlib-neuron-interface"
                ),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["all"]), EXPECTED_EXPORTS)
        self.assertEqual(result["owners"], EXPECTED_OWNERS)
        self.assertEqual(result["before"], dict.fromkeys(PRIVATE_MODULES, False))
        self.assertEqual(
            result["private_packages"],
            {
                "emperor.neuron._cluster": [],
                "emperor.neuron._monitoring": [],
            },
        )
        self.assertEqual(
            result["private_exports"],
            dict.fromkeys(result["private_exports"], False),
        )
        self.assertTrue(result["rng_unchanged"])
        self.assertEqual(
            result["runtime_before"],
            {"emperor.experts": False, "lightning": False, "torch": False},
        )

    def test_lazy_export_resolution_and_unknown_name_run_in_process(self) -> None:
        import emperor.neuron as neuron_package

        export_name = "AxonsConfig"
        cached_export = vars(neuron_package).pop(export_name)
        try:
            resolved_export = getattr(neuron_package, export_name)

            self.assertIs(resolved_export, AxonsConfig)
            self.assertIs(vars(neuron_package)[export_name], AxonsConfig)
        finally:
            setattr(neuron_package, export_name, cached_export)

        missing_export = "MissingNeuronExport"
        with self.assertRaisesRegex(
            AttributeError,
            f"module 'emperor.neuron' has no attribute '{missing_export}'",
        ):
            getattr(neuron_package, missing_export)

    def test_config_enum_trace_and_callback_contracts_are_preserved(self):
        schemas = (
            (NucleusConfig, NUCLEUS_CONFIG_FIELDS),
            (AxonsConfig, AXONS_CONFIG_FIELDS),
            (TerminalConfig, TERMINAL_CONFIG_FIELDS),
            (NeuronConfig, NEURON_CONFIG_FIELDS),
            (NeuronClusterConfig, CLUSTER_CONFIG_FIELDS),
            (NeuronClusterTraceStep, TRACE_STEP_FIELDS),
            (NeuronClusterTrace, TRACE_FIELDS),
        )
        for schema, expected_fields in schemas:
            with self.subTest(schema=schema.__name__):
                self.assertEqual(
                    tuple(field.name for field in fields(schema)),
                    expected_fields,
                )

        for config_type in (
            NucleusConfig,
            AxonsConfig,
            TerminalConfig,
            NeuronConfig,
            NeuronClusterConfig,
        ):
            with self.subTest(config=config_type.__name__):
                config = config_type()
                self.assertTrue(
                    all(getattr(config, field.name) is None for field in fields(config))
                )

        self.assertEqual(
            tuple((option.name, option.value) for option in TerminalRangeOptions),
            (
                ("ONE", 1),
                ("TWO", 2),
                ("THREE", 3),
                ("FOUR", 4),
                ("FIVE", 5),
                ("SIX", 6),
                ("SEVEN", 7),
                ("EIGHT", 8),
            ),
        )
        self.assertEqual(
            tuple((option.name, option.value) for option in TerminalZAxisOffsetOptions),
            (
                ("ZERO", 0),
                ("ONE", 1),
                ("TWO", 2),
                ("THREE", 3),
                ("FOUR", 4),
                ("FIVE", 5),
            ),
        )
        self.assertEqual(
            tuple(
                (option.name, option.value) for option in TerminalConnectionShapeOptions
            ),
            (
                ("BOX", "box"),
                ("CROSS", "cross"),
                ("SPHERE", "sphere"),
                ("DIAGONAL_X", "diagonal_x"),
                ("LINE_LEFT_RIGHT", "line_left_right"),
                ("LINE_UP_DOWN", "line_up_down"),
                ("LINE_FRONT_BACK", "line_front_back"),
            ),
        )

        monitor_parameters = inspect.signature(NeuronClusterMonitorCallback).parameters
        self.assertEqual(
            tuple(monitor_parameters),
            ("log_every_n_steps", "history_size"),
        )
        self.assertEqual(monitor_parameters["log_every_n_steps"].default, 100)
        self.assertEqual(monitor_parameters["history_size"].default, 128)
        self.assertEqual(
            tuple(inspect.signature(NeuronClusterOptimizerSyncCallback).parameters),
            (),
        )

    def test_exact_children_state_topology_and_strict_loading_are_preserved(self):
        with torch.random.fork_rng():
            torch.manual_seed(20260716)
            neuron = self.neuron_config(coordinate_embedding_flag=True).build()
            self.assertEqual(tuple(neuron._modules), ("nucleus", "axons", "terminal"))
            self.assertEqual(_topology(neuron), NEURON_TOPOLOGY)

            torch.manual_seed(20260716)
            restored_neuron = self.neuron_config(coordinate_embedding_flag=True).build()
            result = restored_neuron.load_state_dict(
                neuron.state_dict(),
                strict=True,
            )
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])

            torch.manual_seed(20260716)
            cluster = self.cluster_config().build()
            self.assertEqual(tuple(cluster._modules), ("cluster", "entry_sampler"))
            self.assertEqual(_topology(cluster), CLUSTER_TOPOLOGY)

            torch.manual_seed(20260716)
            restored_cluster = self.cluster_config().build()
            result = restored_cluster.load_state_dict(
                cluster.state_dict(),
                strict=True,
            )
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])

        for source, restored in (
            (neuron, restored_neuron),
            (cluster, restored_cluster),
        ):
            for name, value in source.state_dict().items():
                with self.subTest(model=type(source).__name__, state=name):
                    torch.testing.assert_close(restored.state_dict()[name], value)

    def test_seeded_construction_output_route_and_trace_fingerprints_are_preserved(
        self,
    ):
        with torch.random.fork_rng():
            torch.manual_seed(20260716)
            neuron = self.neuron_config(coordinate_embedding_flag=True).build().eval()
            self.assertEqual(
                _digest(torch.random.get_rng_state()),
                NEURON_RNG_DIGEST,
            )
            output, probabilities, selected_coordinates, auxiliary_loss = neuron(
                _input()
            )
            self.assertEqual(
                _digest(
                    output,
                    probabilities,
                    selected_coordinates,
                    auxiliary_loss,
                ),
                NEURON_OUTPUT_DIGEST,
            )

            torch.manual_seed(20260716)
            cluster = self.cluster_config().build().eval()
            self.assertEqual(
                _digest(torch.random.get_rng_state()),
                CLUSTER_RNG_DIGEST,
            )
            output, auxiliary_loss, trace = cluster(_input(), return_trace=True)
            self.assertEqual(trace.input_shape, (2, 4))
            self.assertEqual(len(trace.steps), 1)
            self.assertTrue(
                all(not tensor.requires_grad for tensor in _trace_tensors(trace))
            )
            self.assertEqual(
                _digest(output, auxiliary_loss, *_trace_tensors(trace)),
                CLUSTER_TRACE_DIGEST,
            )

    def test_invalid_configs_are_rejected_before_rng_consumption(self):
        invalid_neuron = self.neuron_config(coordinate_embedding_flag=True)
        invalid_neuron.terminal_config.input_dim = 2
        invalid_cluster = self.cluster_config()
        invalid_cluster.x_axis_total_neurons = 0

        for name, config in (
            ("neuron", invalid_neuron),
            ("cluster", invalid_cluster),
        ):
            with self.subTest(name=name), torch.random.fork_rng():
                torch.manual_seed(17)
                expected_next_values = torch.randn(8)

                torch.manual_seed(17)
                with self.assertRaises(ValueError):
                    config.build()
                actual_next_values = torch.randn(8)

                torch.testing.assert_close(actual_next_values, expected_next_values)

    def test_recurrent_and_beam_routes_cross_the_split_mixin_seams(self):
        recurrent_cluster = self.scripted_cluster(beam_width=1)

        recurrent_output, recurrent_loss, trace = recurrent_cluster(
            torch.zeros(1, 1),
            return_trace=True,
        )

        torch.testing.assert_close(recurrent_output, torch.tensor([[17.75]]))
        self.assertEqual(recurrent_loss.shape, ())
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(
            int(recurrent_cluster.cluster["neuron_2_1_1"].route_call_counter.item()),
            1,
        )

        beam_cluster = self.scripted_cluster(beam_width=2)
        beam_output, beam_loss = beam_cluster(torch.zeros(1, 1))

        torch.testing.assert_close(beam_output, torch.tensor([[15.5]]))
        self.assertEqual(beam_loss.shape, ())
        self.assertEqual(
            int(beam_cluster.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )
        self.assertEqual(
            int(beam_cluster.cluster["neuron_2_1_1"].route_call_counter.item()),
            1,
        )

    def test_plasticity_checkpoint_and_pruning_cross_the_split_mixin_seams(self):
        with torch.random.fork_rng():
            torch.manual_seed(20260716)
            source = self.cluster_config(
                x_axis_total_neurons=2,
                initial_x_axis_total_neurons=1,
                growth_threshold=1,
            ).build()
            output, auxiliary_loss = source(_input())

            self.assertEqual(
                tuple(source.cluster),
                ("neuron_1_1_1", "neuron_2_1_1"),
            )
            self.assertEqual(
                _digest(output, auxiliary_loss),
                GROWTH_OUTPUT_DIGEST,
            )
            self.assertEqual(
                _digest(*source.state_dict().values()),
                GROWTH_STATE_DIGEST,
            )

            torch.manual_seed(20260716)
            restored = self.cluster_config(
                x_axis_total_neurons=2,
                initial_x_axis_total_neurons=1,
                growth_threshold=1,
            ).build()
            result = restored.load_state_dict(source.state_dict(), strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            self.assertEqual(tuple(restored.cluster), tuple(source.cluster))

        pruning_cluster = self.cluster_config(
            x_axis_total_neurons=5,
            initial_x_axis_total_neurons=1,
            pruning_threshold=2,
        ).build()
        idle_name = "neuron_5_1_1"
        pruning_cluster.cluster[idle_name] = pruning_cluster._initialize_neuron(
            5,
            1,
            1,
        )

        pruning_cluster(_input())
        self.assertIn(idle_name, pruning_cluster.cluster)
        self.assertEqual(
            int(pruning_cluster.cluster[idle_name].atrophy_counter.item()),
            1,
        )
        pruning_cluster(_input())
        self.assertNotIn(idle_name, pruning_cluster.cluster)

    def test_seeded_neuron_payload_preserves_gradient_paths(self):
        with torch.random.fork_rng():
            torch.manual_seed(20260716)
            model = self.neuron_config(coordinate_embedding_flag=True).build().eval()
            input_batch = _input(requires_grad=True)
            output, probabilities, _selected_coordinates, auxiliary_loss = model(
                input_batch
            )
            (
                output.square().sum() + probabilities.square().sum() + auxiliary_loss
            ).backward()

        self.assertIsNotNone(input_batch.grad)
        self.assertTrue(torch.isfinite(input_batch.grad).all())
        self.assertGreater(torch.count_nonzero(input_batch.grad).item(), 0)

        gradients = {
            name: parameter.grad for name, parameter in model.named_parameters()
        }
        for name in (
            "nucleus.model.weight",
            "terminal.sampler.router.model.layers.0.model.weight_params",
            "terminal.sampler.router.model.layers.0.model.bias_params",
        ):
            with self.subTest(parameter=name):
                gradient = gradients[name]
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(torch.count_nonzero(gradient).item(), 0)


if __name__ == "__main__":
    unittest.main()
