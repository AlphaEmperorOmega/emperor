import copy
import unittest
import torch
import torch.nn as nn

from Emperor.config import ModelConfig
from Emperor.components.parameter_generators.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    ParameterLayerConfig,
    VectorParameterLayer,
)
from Emperor.components.parameter_generators.utils.mixture import MixtureConfig
from Emperor.components.parameter_generators.utils.samplers import (
    SamplerFull,
    SamplerSparse,
    SamplerTopk,
)
from Emperor.experiments.layers_experiments import (
    ParameterLayerFactory,
    ParameterLayerPreset,
)
from Emperor.components.parameter_generators.utils.routers import (
    RouterConfig,
)
from Emperor.components.parameter_generators.utils.samplers import (
    SamplerConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class TestParameterLayerFactory(unittest.TestCase):
    def setUp(self):
        # MODEL WISE CONFI
        BATCH_SIZE = 2
        INPUT_DIM = 4
        HIDDEN_DIM = 5
        OUTPUT_DIM = 6
        GATHER_FREQUENCY_FLAG = False

        # AUXILIARY LOSSES OPITONS
        COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
        SWITCH_LOSS_WEIGHT: float = 0.0
        ZERO_CENTERED_LOSS_WEIGHT: float = 0.0
        MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

        # PARAMETER GENRETOR ROUTER OPITONS
        ROUTER_INPUT_DIM = HIDDEN_DIM
        ROUTER_HIDDEN_DIM = 8
        ROUTER_OUTPUT_DIM = 9
        ROUTER_NOISY_TOPK_FLAG = False
        ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
        ROUTER_NUM_LAYERNUM_LAYERSS = 5

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = 3
        SAMPLER_THRESHOLD = 0.1
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM

        # PARAMETER GENRETOR MIXTURE OPITONS
        MIXTURE_INPUT_DIM = ROUTER_INPUT_DIM
        MIXTURE_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_TOP_K = SAMPLER_TOP_K
        MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
        MIXTURE_BIAS_PARAMETERS_FLAG = False
        MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_CROSS_DIAGONAL_FLAG = False

        # PARAMETER GENERATOR OPTIONS
        PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG

        self.cfg = ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            gather_frequency_flag=GATHER_FREQUENCY_FLAG,
            coefficient_of_variation_loss_weight=COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
            switch_loss_weight=SWITCH_LOSS_WEIGHT,
            zero_centered_loss_weight=ZERO_CENTERED_LOSS_WEIGHT,
            mutual_information_loss_weight=MUTUAL_INFORMATION_LOSS_WEIGHT,
            router_model_config=RouterConfig(
                input_dim=ROUTER_INPUT_DIM,
                hidden_dim=ROUTER_HIDDEN_DIM,
                output_dim=ROUTER_OUTPUT_DIM,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION_FUNCTION,
                num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                router_output_dim=SAMPLER_ROUTER_OUTPUT_DIM,
            ),
            mixture_model_config=MixtureConfig(
                input_dim=MIXTURE_INPUT_DIM,
                output_dim=MIXTURE_OUTPUT_DIM,
                depth_dim=MIXTURE_DEPTH_DIM,
                top_k=MIXTURE_TOP_K,
                bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
                weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
                router_output_dim=MIXTURE_ROUTER_OUTPUT_DIM,
                cross_diagonal_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
            ),
            parameter_generator_model_config=ParameterLayerConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
            ),
        )

    def test__vector__create_parameter_layer__create_sparse_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.SPARSE
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerSparse)

    def test__vector__create_parameter_layer__create_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerTopk)

    def test__vector__create_parameter_layer__create_full_mixture_layer(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.weighted_parameters_flag = True
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.FULL_MIXTURE
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerFull)

    def test__vector__create_parameter_layer__create_random_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.RANDOM_TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)

    def test__vector__create_parameter_layer__create_sparse_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.SPARSE_THRESHOLD
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__vector__create_parameter_layer__create_topk_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.TOPK_THRESHOLD

        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__vector__create_parameter_layer__create_full_mixture_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.weighted_parameters_flag = True
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.FULL_MIXTURE_THRESHOLD
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__vector__create_parameter_layer__create_sparse_noisy_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.VECTOR
        model_preset = ParameterLayerPreset.SPARSE_NOISY_TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, VectorParameterLayer)
        self.assertTrue(m.sampler.sampler_model.noisy_topk_flag)
        self.assertTrue(m.weight_router.noisy_topk_flag)

    def test__matrix__create_parameter_layer__create_sparse_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.SPARSE
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerSparse)

    def test__matrix__create_parameter_layer__create_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerTopk)

    def test__matrix__create_parameter_layer__create_full_mixture_layer(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.weighted_parameters_flag = True
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.FULL_MIXTURE
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerFull)

    def test__matrix__create_parameter_layer__create_random_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.RANDOM_TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)

    def test__matrix__create_parameter_layer__create_sparse_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.SPARSE_THRESHOLD
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__matrix__create_parameter_layer__create_topk_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.TOPK_THRESHOLD

        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__matrix__create_parameter_layer__create_full_mixture_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.weighted_parameters_flag = True
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.FULL_MIXTURE_THRESHOLD
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__matrix__create_parameter_layer__create_sparse_noisy_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.MATRIX
        model_preset = ParameterLayerPreset.SPARSE_NOISY_TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, MatrixParameterLayer)
        self.assertTrue(m.sampler.sampler_model.noisy_topk_flag)
        self.assertTrue(m.weight_router.noisy_topk_flag)

    def test__generator__create_parameter_layer__create_sparse_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.SPARSE
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerSparse)

    def test__generator__create_parameter_layer__create_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerTopk)

    def test__generator__create_parameter_layer__create_full_mixture_layer(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.weighted_parameters_flag = True
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.FULL_MIXTURE
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)
        self.assertIsInstance(m.sampler.sampler_model, SamplerFull)

    def test__generator__create_parameter_layer__create_random_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.RANDOM_TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)

    def test__generator__create_parameter_layer__create_sparse_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.SPARSE_THRESHOLD
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__generator__create_parameter_layer__create_topk_threshold_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.TOPK_THRESHOLD

        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__generator__create_parameter_layer__create_full_mixture_threshold_layer(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.weighted_parameters_flag = True
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.FULL_MIXTURE_THRESHOLD
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)
        self.assertEqual(m.sampler.sampler_model.threshold, 0.1)

    def test__generator__create_parameter_layer__create_sparse_noisy_topk_layer(self):
        c = copy.deepcopy(self.cfg)
        model_type = ParameterLayerFactory.GENERATOR
        model_preset = ParameterLayerPreset.SPARSE_NOISY_TOPK
        m = model_preset.create(model_type, c)

        self.assertIsInstance(m, GeneratorParameterLayer)
        self.assertTrue(m.sampler.sampler_model.noisy_topk_flag)
        self.assertTrue(m.weight_router.noisy_topk_flag)
