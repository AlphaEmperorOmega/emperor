from enum import Enum


class BaseOptions(Enum):
    @classmethod
    def cli_name(cls, name: str) -> str:
        return name.lower().replace("_", "-")

    @classmethod
    def _member_name(cls, name: str | None) -> str | None:
        if name is None:
            return None
        normalized_name = cls.cli_name(name)
        for member_name in cls.__members__:
            if normalized_name in {member_name, cls.cli_name(member_name)}:
                return member_name
        return None

    @classmethod
    def get_member(cls, name: str | None):
        member_name = cls._member_name(name)
        if member_name is None:
            if name is None:
                return None
            raise ValueError(f"Option '{name}' does not exist in {cls.__name__}.")
        return cls[member_name]

    @classmethod
    def cli_names(cls) -> list[str]:
        return [cls.cli_name(name) for name in cls.__members__]

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.__members__.keys())


__all__ = ["BaseOptions"]
