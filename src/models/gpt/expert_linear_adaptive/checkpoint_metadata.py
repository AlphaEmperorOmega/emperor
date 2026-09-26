"""Recover only dimensions that are unambiguous in hierarchical checkpoints."""

import re
from collections.abc import Mapping


def checkpoint_config_overrides(
    tensor_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
    prefix = "hierarchical_model."
    encoder = tensor_shapes.get(prefix + "encoder.byte_embedding.weight")
    decoder = tensor_shapes.get(prefix + "decoder.byte_embedding.weight")
    if encoder is None or decoder is None:
        return {}
    if (
        len(encoder) != 2
        or len(decoder) != 2
        or encoder[0] != 257
        or decoder[0] != 256
        or encoder[1] != decoder[1]
    ):
        raise ValueError("Incompatible hierarchical byte tables in checkpoint")
    result = {
        "hierarchical_language_model_flag": True,
        "lm_head_weight_tying_flag": False,
        "hierarchical_byte_embedding_dim": encoder[1],
        "input_dim": 256,
        "output_dim": 258,
    }
    beginning = tensor_shapes.get(prefix + "beginning_of_document")
    if beginning and len(beginning) == 1:
        result["hidden_dim"] = beginning[0]
    positions = tensor_shapes.get(
        prefix + "decoder.byte_position.embedding_model.weight"
    )
    if positions and len(positions) == 2:
        result["hierarchical_max_token_bytes"] = positions[0] - 1
    for key, owner in (
        ("hierarchical_byte_encoder_num_layers", "encoder.encoder.encoder_model"),
        ("hierarchical_byte_decoder_num_layers", "decoder.decoder.decoder_model"),
        ("stack_num_layers", "backbone"),
    ):
        pattern = re.compile(re.escape(prefix + owner) + r"\.layers\.(\d+)\.")
        indices = sorted(
            {int(match[1]) for name in tensor_shapes if (match := pattern.match(name))}
        )
        if indices and indices == list(range(len(indices))):
            result[key] = len(indices)
    # Attention head counts, routing, activations and adaptive settings must
    # come from the saved run configuration; weights cannot identify them.
    return result


__all__ = ["checkpoint_config_overrides"]
