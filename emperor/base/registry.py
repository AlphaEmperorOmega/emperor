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

    cls.register = classmethod(register)
    cls.resolve = classmethod(resolve)
    return cls
