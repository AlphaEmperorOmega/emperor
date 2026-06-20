def router_hidden_dim(input_dim: int) -> int:
    return max(4, min(input_dim, 32))
