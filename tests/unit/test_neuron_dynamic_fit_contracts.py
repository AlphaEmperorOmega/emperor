from types import SimpleNamespace
from unittest.mock import Mock

import torch
from lightning import LightningModule
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

from emperor.neuron import NeuronClusterConfig, NeuronClusterOptimizerSyncCallback
from unit.test_neuron import NeuronTestCase


class _DynamicModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cluster = config.build()


class TestNeuronDynamicFitContracts(NeuronTestCase):
    def module(self):
        return _DynamicModule(
            NeuronClusterConfig(
                x_axis_total_neurons=2,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                initial_x_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=1,
                neuron_config=self.neuron_config(),
            )
        )

    def test_manual_dynamic_training_rejects_before_fit_updates(self):
        module = self.module()
        module.automatic_optimization = False
        trainer = SimpleNamespace(strategy=SingleDeviceStrategy())
        callback = NeuronClusterOptimizerSyncCallback()
        with self.assertRaisesRegex(RuntimeError, "automatic optimization"):
            callback.setup(trainer, module, "fit")

    def test_unsupported_strategies_and_custom_communication_reject(self):
        for strategy in (
            object(),
            DDPStrategy(ddp_comm_hook=lambda *_: None),
            DDPStrategy(ddp_comm_state=object()),
            DDPStrategy(ddp_comm_wrapper=lambda value: value),
        ):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(RuntimeError, "standard DDP|communication"):
                    NeuronClusterOptimizerSyncCallback().setup(
                        SimpleNamespace(strategy=strategy), self.module(), "fit"
                    )

    def test_unsupported_optimizer_rejects_before_membership_mutation(self):
        module = self.module()
        parameter = next(module.parameters())
        for optimizers in (
            [],
            [torch.optim.SGD([parameter], lr=0.1), torch.optim.Adam([parameter])],
            [torch.optim.LBFGS([parameter])],
        ):
            with self.subTest(optimizers=optimizers):
                callback = NeuronClusterOptimizerSyncCallback()
                with self.assertRaisesRegex(RuntimeError, "one ordinary"):
                    callback.on_fit_start(
                        SimpleNamespace(optimizers=optimizers), module
                    )
                self.assertIsNone(callback._fit_optimizers)
                self.assertFalse(callback._synced_param_ids)

    def test_hooks_registered_on_ddp_wrapper_after_setup_reject(self):
        module = self.module()
        strategy = DDPStrategy()
        trainer = SimpleNamespace(
            strategy=strategy,
            optimizers=[torch.optim.Adam(module.parameters())],
            lr_scheduler_configs=[],
        )
        callback = NeuronClusterOptimizerSyncCallback()
        callback.setup(trainer, module, "fit")
        wrapper = Mock(spec=torch.nn.parallel.DistributedDataParallel)
        wrapper._comm_hooks = []
        strategy.model = wrapper
        callback.on_fit_start(trainer, module)
        wrapper._comm_hooks = [(lambda *_: None, None)]
        for hook in (callback.on_fit_start, callback.on_train_start):
            with self.assertRaisesRegex(RuntimeError, "communication hooks"):
                hook(trainer, module)
        with self.assertRaisesRegex(RuntimeError, "communication hooks"):
            callback.on_before_backward(trainer, module, torch.tensor(0.0))

    def test_supported_optimizers_and_foreach_are_accepted(self):
        for optimizer_type in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            with self.subTest(optimizer=optimizer_type):
                module = self.module()
                optimizer = optimizer_type(module.parameters(), lr=0.01, foreach=True)
                trainer = SimpleNamespace(
                    strategy=SingleDeviceStrategy(),
                    optimizers=[optimizer],
                    lr_scheduler_configs=[],
                )
                callback = NeuronClusterOptimizerSyncCallback()
                callback.setup(trainer, module, "fit")
                callback.on_fit_start(trainer, module)
                callback.on_train_start(trainer, module)
                self.assertTrue(callback._fit_started)

    def test_restored_unsupported_options_reject_before_checkpoint_commit(self):
        for option in ("fused", "capturable", "differentiable"):
            with self.subTest(option=option):
                module = self.module()
                optimizer = torch.optim.Adam(module.parameters())
                trainer = SimpleNamespace(
                    optimizers=[optimizer], lr_scheduler_configs=[]
                )
                callback = NeuronClusterOptimizerSyncCallback()
                checkpoint = {"optimizer_states": [], "lr_schedulers": []}
                callback.on_load_checkpoint(trainer, module, checkpoint)
                callback.on_fit_start(trainer, module)
                optimizer.param_groups[0][option] = True
                with self.assertRaisesRegex(RuntimeError, option):
                    callback.on_train_start(trainer, module)
                self.assertTrue(callback._checkpoint_load_prepared)
                callback.on_exception(trainer, module, RuntimeError("failed fit"))
                self.assertFalse(optimizer.param_groups[0].get(option, False))

    def test_fixed_topology_and_direct_sync_keep_existing_flexibility(self):
        module = self.module()
        module.automatic_optimization = False
        optimizer = torch.optim.LBFGS(module.parameters())
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.sync_optimizers(trainer, module)
        module.cluster.growth_threshold = None
        callback.on_fit_start(trainer, module)
        callback.on_train_start(trainer, module)
