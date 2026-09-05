"""Synthetic qualification: three updates / six microbatches per fit at most."""

import copy
from datetime import timedelta

import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset

from emperor.neuron import NeuronClusterConfig, NeuronClusterOptimizerSyncCallback
from model_packages.test_neuron_precision import _build_model
from model_packages.test_neuron_terminal_routing_tree_runtime import PACKAGE_BUILDERS
from unit import test_neuron as fixtures


class _QualificationModule(LightningModule):
    def __init__(self, optimizer_name="SGD", *, prune=False):
        super().__init__()
        fixture = fixtures.NeuronTestCase()
        self.cluster = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None if prune else 1,
            max_total_growths=None if prune else 1,
            pruning_threshold=1 if prune else None,
            neuron_config=(
                fixture.neuron_config()
                if prune
                else fixture.full_sampler_neuron_config()
            ),
        ).build()
        self.prune = prune
        if prune:
            self.cluster.cluster["neuron_2_1_1"] = self.cluster._initialize_neuron(
                2, 1, 1
            )
        self.optimizer_name = optimizer_name
        self.expected_updates = {}
        self.checked_updates = 0
        self.initial_snapshot = None
        self.final_snapshot = None
        self.checkpoint_destination = None
        self.expected_restoration = None

    def configure_callbacks(self):
        return [NeuronClusterOptimizerSyncCallback()]

    def configure_optimizers(self):
        options = dict(lr=0.01)
        if self.optimizer_name == "SGD":
            options["momentum"] = 0.9
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), **options
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        source = batch[0]
        if self.trainer.world_size > 1 and self.global_rank == 0:
            source = source[:0]
        source = source.detach().requires_grad_()
        output, auxiliary = self.cluster(source)
        # Empty local batches must still participate in backward and collectives.
        return output.square().sum() / max(output.numel(), 1) + auxiliary

    def on_train_start(self):
        self.initial_snapshot = self.snapshot()
        if self.expected_restoration is not None:
            torch.testing.assert_close(
                self.initial_snapshot, self.expected_restoration, rtol=0, atol=0
            )

    def on_before_optimizer_step(self, optimizer):
        if self.trainer.world_size == 1:
            return
        gradients = {
            name: parameter.grad for name, parameter in self.named_parameters()
        }
        gathered = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(gathered, gradients)
        for peer in gathered[1:]:
            torch.testing.assert_close(peer, gathered[0], rtol=0, atol=0)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )
        names = {id(parameter): name for name, parameter in self.named_parameters()}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                assert torch.isfinite(gradient).all()
                state = optimizer.state[parameter]
                value = parameter.detach().clone()
                if self.optimizer_name == "SGD":
                    momentum = state.get("momentum_buffer", torch.zeros_like(gradient))
                    momentum = 0.9 * momentum + gradient
                    expected_state = {"momentum_buffer": momentum}
                    value = value - group["lr"] * momentum
                else:
                    step = int(state.get("step", 0)) + 1
                    beta1, beta2 = group["betas"]
                    first = (
                        beta1 * state.get("exp_avg", torch.zeros_like(gradient))
                        + (1 - beta1) * gradient
                    )
                    second = (
                        beta2 * state.get("exp_avg_sq", torch.zeros_like(gradient))
                        + (1 - beta2) * gradient.square()
                    )
                    if self.optimizer_name == "AdamW":
                        value = value * (1 - group["lr"] * group["weight_decay"])
                    value = value - group["lr"] * (first / (1 - beta1**step)) / (
                        (second / (1 - beta2**step)).sqrt() + group["eps"]
                    )
                    expected_state = {
                        "step": torch.tensor(float(step)),
                        "exp_avg": first,
                        "exp_avg_sq": second,
                    }
                self.expected_updates[names[id(parameter)]] = (value, expected_state)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.expected_updates:
            return
        optimizer = self.trainer.optimizers[0]
        parameters = dict(self.named_parameters())
        for name, (expected_value, expected_state) in self.expected_updates.items():
            parameter = parameters[name]
            torch.testing.assert_close(parameter, expected_value)
            torch.testing.assert_close(optimizer.state[parameter], expected_state)
        self.expected_updates.clear()
        self.checked_updates += 1
        assert {id(parameter) for parameter in self.parameters()} == {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if self.trainer.world_size > 1:
            snapshots = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(snapshots, self.snapshot())
            for peer in snapshots[1:]:
                torch.testing.assert_close(peer, snapshots[0], rtol=0, atol=0)

    def snapshot(self):
        optimizer = self.trainer.optimizers[0]
        return copy.deepcopy(
            {
                "parameters": {
                    name: parameter.detach()
                    for name, parameter in self.named_parameters()
                },
                "moments": {
                    name: optimizer.state.get(parameter, {})
                    for name, parameter in self.named_parameters()
                },
                "scheduler": self.lr_schedulers().state_dict(),
            }
        )

    def on_train_end(self):
        assert self.checked_updates >= 1
        assert ("neuron_2_1_1" in self.cluster.cluster) != self.prune
        if not self.prune:
            parameter = self.cluster.cluster["neuron_2_1_1"].nucleus.model.weight
            assert self.trainer.optimizers[0].state[parameter]
        self.final_snapshot = self.snapshot()
        if self.checkpoint_destination is not None:
            self.trainer.save_checkpoint(self.checkpoint_destination)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["qualification_snapshot"] = self.snapshot()


def _loader():
    source = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4) / 100
    return DataLoader(TensorDataset(source), batch_size=4, shuffle=False)


def _trainer(directory, precision, *, distributed=False, max_steps=3):
    return Trainer(
        accelerator="cpu",
        devices=2 if distributed else 1,
        strategy=DDPStrategy(
            start_method="spawn",
            process_group_backend="gloo",
            timeout=timedelta(seconds=45),
        )
        if distributed
        else "auto",
        precision=precision,
        max_epochs=3,
        max_steps=max_steps,
        accumulate_grad_batches=2,
        limit_train_batches=6,
        gradient_clip_val=0.25,
        use_distributed_sampler=False,
        default_root_dir=directory,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


@pytest.mark.training
@pytest.mark.parametrize("precision", ["32-true", "64-true", "bf16-mixed"])
@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam", "AdamW"])
def test_single_device_updates_match_parameter_and_moment_equations(
    tmp_path, precision, optimizer_name
):
    torch.manual_seed(17)
    module = _QualificationModule(optimizer_name)
    _trainer(tmp_path, precision).fit(module, train_dataloaders=_loader())
    assert module.checked_updates == 3
    assert module.final_snapshot["scheduler"]["last_epoch"] == 3


@pytest.mark.training
@pytest.mark.parametrize("precision", ["32-true", "64-true", "bf16-mixed"])
@pytest.mark.parametrize("prune", [False, True])
def test_lightning_gloo_empty_peer_accumulation_and_topology(
    tmp_path, precision, prune
):
    torch.manual_seed(17)
    module = _QualificationModule("SGD", prune=prune)
    _trainer(tmp_path, precision, distributed=True).fit(
        module, train_dataloaders=_loader()
    )


@pytest.mark.training
@pytest.mark.parametrize("precision", ["32-true", "64-true", "bf16-mixed"])
def test_precision_checkpoint_restores_parameters_moments_and_scheduler(
    tmp_path, precision
):
    torch.manual_seed(17)
    source = _QualificationModule("AdamW")
    trainer = _trainer(tmp_path, precision, max_steps=2)
    trainer.fit(source, train_dataloaders=_loader())
    checkpoint = tmp_path / "continuation.ckpt"
    trainer.save_checkpoint(checkpoint)
    target = _QualificationModule("AdamW")
    # Lightning's early restore precedes strategy precision conversion.
    if precision == "64-true":
        target.double()
    resumed = _trainer(tmp_path, precision, max_steps=3)
    resumed.fit(target, train_dataloaders=_loader(), ckpt_path=checkpoint)
    torch.testing.assert_close(
        target.initial_snapshot, source.final_snapshot, rtol=0, atol=0
    )
    assert target.checked_updates == 1
    assert target.final_snapshot["scheduler"]["last_epoch"] == 3


class _EmptyPeerBatch(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_rank == 0:
            batch[0] = batch[0][:0]
            batch[1] = batch[1][:0]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert torch.isfinite(outputs["loss"])
        optimizer = trainer.optimizers[0]
        snapshot = {
            name: (parameter.detach(), optimizer.state.get(parameter, {}))
            for name, parameter in pl_module.named_parameters()
        }
        gathered = [None] * trainer.world_size
        torch.distributed.all_gather_object(gathered, snapshot)
        for peer in gathered[1:]:
            torch.testing.assert_close(peer, gathered[0], rtol=0, atol=0)
        assert pl_module.neuron_cluster.total_growth_count.item() == 1


@pytest.mark.training
@pytest.mark.parametrize("package,builder_name", PACKAGE_BUILDERS)
def test_real_packages_train_with_empty_gloo_peer(tmp_path, package, builder_name):
    model = _build_model(
        package,
        builder_name,
        2,
        2,
        cluster_x_axis_total_neurons=2,
        cluster_growth_threshold=1,
        cluster_max_total_growths=1,
    ).train()
    trainer = _trainer(tmp_path, "bf16-mixed", distributed=True)
    trainer.callbacks.extend([NeuronClusterOptimizerSyncCallback(), _EmptyPeerBatch()])
    source = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4) / 100
    labels = torch.arange(24) % 2
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            TensorDataset(source, labels), batch_size=4, shuffle=False
        ),
    )


@pytest.mark.training
@pytest.mark.parametrize("precision", ["32-true", "64-true", "bf16-mixed"])
def test_lightning_gloo_checkpoint_continuation(tmp_path, precision):
    torch.manual_seed(17)
    checkpoint = tmp_path / "distributed.ckpt"
    source = _QualificationModule("AdamW")
    source.checkpoint_destination = checkpoint
    _trainer(tmp_path, precision, distributed=True, max_steps=2).fit(
        source, train_dataloaders=_loader()
    )
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    target = _QualificationModule("AdamW")
    target.expected_restoration = saved["qualification_snapshot"]
    if precision == "64-true":
        target.double()
    _trainer(tmp_path, precision, distributed=True, max_steps=3).fit(
        target, train_dataloaders=_loader(), ckpt_path=checkpoint
    )
