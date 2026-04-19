def subclass_registry(cls):
    cls._registry = {}

    def register(outer_cls, option):
        def decorator(inner_cls):
            outer_cls._registry[option] = inner_cls
            return inner_cls

        return decorator

    def resolve(outer_cls, option):
        if option not in outer_cls._registry:
            raise ValueError(
                f"No handler registered for {type(option).__name__}: {option.name}"
            )
        return outer_cls._registry[option]

    def build_from_config(outer_cls, cfg, overrides=None):
        if cfg.model_type is None:
            raise ValueError(
                f"`model_type` must be set before building "
                f"{outer_cls.__name__}, got {cfg.model_type!r}"
            )
        target_cls = outer_cls.resolve(cfg.model_type)
        return target_cls(cfg, overrides)

    cls.register = classmethod(register)
    cls.resolve = classmethod(resolve)
    cls.build_from_config = classmethod(build_from_config)
    return cls
