import copy
import torch.nn as nn

from torch import Tensor
from Emperor.base.utils import Module
from Emperor.layers.utils.base import LayerBlock, ParameterGeneratorLayerBlock
from Emperor.layers.utils.linears import LinearLayer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.attention.attention import MultiHeadAttentionConfig


class ProjectorBase(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        main_cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.main_cfg = main_cfg
        self.model_type = self.cfg.model_type
        self.embedding_dim = self.cfg.embedding_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.__resolve_kv_dimensions()
        self.layer_block_model = self.__resolve_layer_block_class()

    def __resolve_layer_block_class(self) -> type[LayerBlock]:
        # TODO: move this somewhere else in the future since it is used in
        # `LayerBlockStack` as well
        from Emperor.layers.utils.enums import LinearLayerTypes, ParameterGeneratorTypes

        if isinstance(self.model_type, LinearLayerTypes):
            return LayerBlock
        elif isinstance(self.model_type, ParameterGeneratorTypes):
            return ParameterGeneratorLayerBlock
        else:
            raise RuntimeError(
                f"Unsupported `model_type` {type(self.model_type)} for `LayerBlockStack`"
            )

    def __resolve_kv_dimensions(self):
        self.query_key_projection_dim = (
            self.embedding_dim
            if self.query_key_projection_dim == 0
            else self.query_key_projection_dim
        )
        self.value_projection_dim = (
            self.embedding_dim
            if self.value_projection_dim == 0
            else self.value_projection_dim
        )

    def _create_model(self, input_dim: int, output_dim: int) -> LayerBlock:
        config = self.__resolve_model_type_overrides(
            self.main_cfg, input_dim, output_dim
        )
        output_model = self.model_type.value(config)
        return self.layer_block_model(model=output_model)

    def __resolve_model_type_overrides(
        self,
        cfg: "ModelConfig",
        input_dim: int,
        output_dim: int,
    ):
        c = copy.deepcopy(cfg)
        if issubclass(self.model_type.value, LinearLayer):
            c.linear_layer_model_config.input_dim = input_dim
            c.linear_layer_model_config.output_dim = output_dim
            return c
        c.mixture_model_config.input_dim = input_dim
        c.mixture_model_config.output_dim = output_dim
        return c


class Projector(ProjectorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        main_cfg: "ModelConfig",
    ):
        super().__init__(cfg, main_cfg)
        self.use_separate_projection_weight_flag = (
            self.cfg.use_separate_projection_weight_flag
        )
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag

        m = self.__build_projection_models()
        if isinstance(m, tuple):
            self.query_model, self.key_model, self.value_model = m
        else:
            self.qkv_model = m

        self.output_model = self._create_model(
            self.value_projection_dim, self.embedding_dim
        )

    def __build_projection_models(self) -> tuple:
        if (
            not self.use_separate_projection_weight_flag
            and self.__are_qkv_dimensions_equal()
        ):
            return self.__build_shared_projection_models()
        return self.__build_separate_projection_models()

    def __build_separate_projection_models(self) -> tuple:
        query_model = self._create_model(
            self.embedding_dim, self.query_key_projection_dim
        )
        key_model = self._create_model(
            self.embedding_dim, self.query_key_projection_dim
        )
        value_model = self._create_model(self.embedding_dim, self.value_projection_dim)
        self.register_parameter("qkv_model", None)
        return query_model, key_model, value_model

    def __build_shared_projection_models(self) -> LayerBlock:
        self.register_parameter("query_model", None)
        self.register_parameter("key_model", None)
        self.register_parameter("value_model", None)
        qkv_model = self._create_model(self.embedding_dim, self.embedding_dim * 3)
        return qkv_model

    def __are_qkv_dimensions_equal(self) -> bool:
        are_qk_dims_same = self.embedding_dim == self.query_key_projection_dim
        are_qv_dims_same = self.embedding_dim == self.value_projection_dim
        return are_qk_dims_same and are_qv_dims_same

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        are_qkv_same = key is value and query is key
        should_perform_self_attention = (
            are_qkv_same and not self.use_separate_projection_weight_flag
        )
        if should_perform_self_attention:
            self.__check_self_attention_projection_inputs(key, value)
            return self.__compute_self_attention_projections(query)

        assert not self.return_attention_weights_flag, (
            "`attention_weights` can be returned only when self attention is performed, ensure that `use_separate_projection_weight_flag` is set to `False` and the `query`, `key` and `value` tensors are the same tensor."
        )
        self.__validate_attention_weights_flag_with_projection_type()
        self.__are_separate_projection_models_initialized()
        self.__check_indepentent_projections_inputs(key, value)
        return self.__compute_indepentet_projections(query, key, value)

    def __compute_self_attention_projections(
        self, query: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        qkv_projection = self.__compute_projection(query, self.qkv_model)
        query_projections, key_projections, value_projections = (
            self.__split_self_attention_projection(qkv_projection)
        )
        return query_projections, key_projections, value_projections

    def __split_self_attention_projection(
        self, qkv_projections: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        projections = qkv_projections.unflatten(-1, (3, -1))
        projections = projections.unsqueeze(0)
        projections = projections.transpose(0, -2)
        projections = projections.squeeze(-2)
        projections = projections.contiguous()
        query_projections = projections[0]
        key_projections = projections[1]
        value_projections = projections[2]

        return query_projections, key_projections, value_projections

    def __compute_indepentet_projections(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_projections = self.__compute_projection(query, self.query_model)
        key_projections = self.__compute_projection(key, self.key_model)
        value_projections = self.__compute_projection(value, self.value_model)

        return query_projections, key_projections, value_projections

    def __compute_projection(self, tensor: Tensor, model: nn.Module) -> Tensor:
        sequence_length, batch_size, embedding_dim = tensor.shape
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensor_reshaped = tensor.view(-1, embedding_dim)
        projections = model(tensor_reshaped)
        if isinstance(projections, tuple):
            projections, _ = projections
        return projections.view(sequence_length, batch_size, -1)

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        output = self.output_model(weighted_values)
        if isinstance(output, tuple):
            output, _ = output
        return output

    def __validate_attention_weights_flag_with_projection_type(self):
        assert not self.return_attention_weights_flag, (
            "`attention_weights` can be returned only when self attention is performed, ensure that `use_separate_projection_weight_flag` is set to `False` and the `query`, `key` and `value` tensors are the same tensor."
        )

    def __are_separate_projection_models_initialized(self) -> None:
        ensure_qkv_models_exist = (
            self.query_model is not None
            and self.key_model is not None
            and self.value_model is not None
        )
        assert ensure_qkv_models_exist, (
            "When query, key, and value are not the same and self attention is not performed, ensure `use_separate_projection_weight_flag` is `True`"
        )

    def __check_indepentent_projections_inputs(
        self, key: Tensor, value: Tensor
    ) -> None:
        k_sequence_length, k_batch_size, _ = key.shape
        v_sequence_length, v_batch_size, _ = value.shape
        is_kv_sequence_length_same = k_sequence_length == v_sequence_length
        is_kv_batch_size_same = k_batch_size == v_batch_size
        if not (is_kv_sequence_length_same and is_kv_batch_size_same):
            raise RuntimeError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )

    def __check_self_attention_projection_inputs(
        self, key: Tensor, value: Tensor
    ) -> None:
        are_kv_shapes_same = key.shape == value.shape
        if not are_kv_shapes_same:
            raise RuntimeError(
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )
