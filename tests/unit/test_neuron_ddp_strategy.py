import unittest
from types import SimpleNamespace

from lightning.pytorch.strategies import DDPStrategy
from torch import nn

from emperor.neuron import NeuronClusterOptimizerSyncCallback


class TestConditionalNeuronDDPConfiguration(unittest.TestCase):
    def test_fit_setup_enables_unused_parameter_detection(self) -> None:
        strategy = DDPStrategy()

        NeuronClusterOptimizerSyncCallback().setup(
            SimpleNamespace(strategy=strategy),
            nn.Module(),
            "fit",
        )

        self.assertIs(strategy._ddp_kwargs["find_unused_parameters"], True)

    def test_non_fit_setup_leaves_strategy_unchanged(self) -> None:
        strategy = DDPStrategy(find_unused_parameters=False)

        NeuronClusterOptimizerSyncCallback().setup(
            SimpleNamespace(strategy=strategy),
            nn.Module(),
            "validate",
        )

        self.assertIs(strategy._ddp_kwargs["find_unused_parameters"], False)

    def test_incompatible_ddp_options_are_rejected(self) -> None:
        cases = (
            (
                {"find_unused_parameters": False},
                "find_unused_parameters=True",
            ),
            ({"static_graph": True}, "static_graph=False"),
            (
                {"skip_all_reduce_unused_params": True},
                "skip_all_reduce_unused_params=False",
            ),
        )
        for options, expected_message in cases:
            with self.subTest(options=options):
                strategy = DDPStrategy(**options)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    NeuronClusterOptimizerSyncCallback().setup(
                        SimpleNamespace(strategy=strategy),
                        nn.Module(),
                        "fit",
                    )

    def test_missing_lightning_ddp_configuration_is_reported(self) -> None:
        strategy = DDPStrategy()
        del strategy._ddp_kwargs

        with self.assertRaisesRegex(
            RuntimeError,
            "does not expose the DDP configuration",
        ):
            NeuronClusterOptimizerSyncCallback().setup(
                SimpleNamespace(strategy=strategy),
                nn.Module(),
                "fit",
            )

    def test_non_ddp_strategy_is_ignored(self) -> None:
        NeuronClusterOptimizerSyncCallback().setup(
            SimpleNamespace(strategy=object()),
            nn.Module(),
            "fit",
        )
