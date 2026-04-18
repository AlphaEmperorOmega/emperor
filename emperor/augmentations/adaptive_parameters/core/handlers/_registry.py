from enum import Enum
from typing import Generic, Self, TypeVar

from emperor.base.utils import Module

HandlerOptionT = TypeVar("HandlerOptionT", bound=Enum)


class HandlerRegistryBase(Module, Generic[HandlerOptionT]):
    _registry: dict
    _registry_label: str = "handler"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if HandlerRegistryBase in cls.__bases__:
            cls._registry = {}

    @classmethod
    def register(cls, option: HandlerOptionT):
        def decorator(handler_cls: type[Self]) -> type[Self]:
            cls._registry[option] = handler_cls
            return handler_cls

        return decorator

    @classmethod
    def resolve(cls, option: HandlerOptionT) -> type[Self]:
        if option not in cls._registry:
            raise ValueError(
                f"No handler registered for {cls._registry_label} option: {option}"
            )
        return cls._registry[option]
