"""Authoritative HTTP operation policy declarations for the Workbench API."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from inspect import signature
from typing import ParamSpec, TypeVar

from fastapi import HTTPException
from fastapi.routing import APIRoute, RouteContext, iter_route_contexts
from starlette.routing import BaseRoute, Match, Mount
from starlette.types import Scope

from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.security import LOCAL_MUTATION_DISABLED_DETAIL

_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_POLICY_DECLARATIONS_ATTRIBUTE = "__workbench_http_operation_policies__"

P = ParamSpec("P")
R = TypeVar("R")


class HttpOperationPolicy(StrEnum):
    """Operational classification for an explicitly declared HTTP operation."""

    READ_ONLY = "read-only"
    LOCAL_MUTATION = "local-mutation"
    LOG_IMPORT = "log-import"

    def __str__(self) -> str:
        """Preserve the legacy ``str, Enum`` representation."""

        return f"{type(self).__name__}.{self.name}"

    def __format__(self, format_spec: str) -> str:
        """Apply formatting to the compatibility string representation."""

        return format(str(self), format_spec)

    @property
    def is_mutation(self) -> bool:
        return self is not HttpOperationPolicy.READ_ONLY


class MutationPolicyConfigurationError(RuntimeError):
    """Raised when the configured route graph has incomplete policy metadata."""


@dataclass(frozen=True, slots=True)
class HttpOperation:
    """One mounted non-safe-method operation and its declared policy."""

    method: str
    route: RouteContext
    policy: HttpOperationPolicy


@dataclass(frozen=True, slots=True)
class HttpOperationCatalog(Sequence[HttpOperation]):
    """Immutable catalog derived from the configured FastAPI route graph."""

    operations: tuple[HttpOperation, ...]

    def __getitem__(self, index: int) -> HttpOperation:
        return self.operations[index]

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self) -> Iterator[HttpOperation]:
        return iter(self.operations)

    @property
    def mutations(self) -> tuple[HttpOperation, ...]:
        return tuple(
            operation for operation in self.operations if operation.policy.is_mutation
        )

    def mutation_for_scope(self, scope: Scope) -> HttpOperation | None:
        """Match a mutation using Starlette's mounted-route semantics."""

        method = scope.get("method", "").upper()
        match_scope = (
            scope if scope.get("method") == method else {**scope, "method": method}
        )
        for operation in self.operations:
            if operation.method != method:
                continue
            match, _ = operation.route.matches(match_scope)
            if match is Match.FULL:
                return operation if operation.policy.is_mutation else None
        return None


def declare_http_operation(
    policy: HttpOperationPolicy,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Declare the policy for one route handler beside its registration."""

    def decorator(
        endpoint: Callable[P, Awaitable[R]],
    ) -> Callable[P, Awaitable[R]]:
        declarations = tuple(getattr(endpoint, _POLICY_DECLARATIONS_ATTRIBUTE, ()))
        endpoint_signature = signature(endpoint, eval_str=True)

        @wraps(endpoint)
        async def enforce_policy(*args: P.args, **kwargs: P.kwargs) -> R:
            if not isinstance(policy, HttpOperationPolicy):
                raise MutationPolicyConfigurationError(
                    "Route handler declares an unknown HTTP operation policy."
                )
            if policy.is_mutation:
                bound = endpoint_signature.bind_partial(*args, **kwargs)
                settings = bound.arguments.get("settings")
                if not isinstance(settings, WorkbenchApiSettings):
                    raise MutationPolicyConfigurationError(
                        "Mutation route handlers must receive Workbench API settings."
                    )
                enforce_operation_policy(policy, settings)
            return await endpoint(*args, **kwargs)

        setattr(
            enforce_policy,
            _POLICY_DECLARATIONS_ATTRIBUTE,
            (*declarations, policy),
        )
        enforce_policy.__signature__ = endpoint_signature  # type: ignore[attr-defined]
        return enforce_policy

    return decorator


def operation_policy_enabled(
    policy: HttpOperationPolicy,
    settings: WorkbenchApiSettings,
) -> bool:
    """Return the total operational-enablement fact for a known policy."""

    if policy is HttpOperationPolicy.READ_ONLY:
        return True
    if policy is HttpOperationPolicy.LOCAL_MUTATION:
        return settings.allow_unsafe_local_mutations
    if policy is HttpOperationPolicy.LOG_IMPORT:
        return settings.log_imports_enabled
    raise MutationPolicyConfigurationError("Unknown HTTP operation policy.")


def enforce_operation_policy(
    policy: HttpOperationPolicy,
    settings: WorkbenchApiSettings,
) -> None:
    """Fail closed when a declared operational policy is disabled."""

    if operation_policy_enabled(policy, settings):
        return
    raise HTTPException(status_code=403, detail=LOCAL_MUTATION_DISABLED_DETAIL)


def _non_safe_methods(route: RouteContext) -> tuple[str, ...]:
    return tuple(
        sorted(
            method for method in route.methods or () if method not in _SAFE_HTTP_METHODS
        )
    )


def _declared_route_counts(
    routes: Sequence[BaseRoute | RouteContext],
) -> Counter[tuple[int, tuple[str, ...]]]:
    return Counter(
        (id(route.endpoint), methods)
        for route in iter_route_contexts(routes)
        if isinstance(route.original_route, APIRoute)
        if (methods := _non_safe_methods(route))
    )


def _opaque_non_safe_route(
    routes: Sequence[BaseRoute | RouteContext],
) -> RouteContext | None:
    for route in iter_route_contexts(routes):
        methods = route.methods
        if (
            not isinstance(route.original_route, APIRoute)
            and methods
            and any(method not in _SAFE_HTTP_METHODS for method in methods)
        ):
            return route
    return None


def build_http_operation_catalog(
    routes: Sequence[BaseRoute | RouteContext],
    *,
    declared_routes: Sequence[BaseRoute | RouteContext] | None = None,
) -> HttpOperationCatalog:
    """Build a total non-safe-method catalog from final mounted routes."""

    if any(
        isinstance(route.original_route, Mount) for route in iter_route_contexts(routes)
    ) or (
        declared_routes is not None
        and any(
            isinstance(route.original_route, Mount)
            for route in iter_route_contexts(declared_routes)
        )
    ):
        raise MutationPolicyConfigurationError(
            "Opaque child mounts are not supported by HTTP operation policy discovery."
        )

    opaque_route = _opaque_non_safe_route(routes)
    if opaque_route is None and declared_routes is not None:
        opaque_route = _opaque_non_safe_route(declared_routes)
    if opaque_route is not None:
        raise MutationPolicyConfigurationError(
            "Non-safe HTTP operations must be registered as declared FastAPI routes."
        )

    if declared_routes is not None:
        declared_counts = _declared_route_counts(declared_routes)
        mounted_counts = _declared_route_counts(routes)
        if declared_counts != mounted_counts:
            raise MutationPolicyConfigurationError(
                "Declared and mounted non-safe HTTP operations do not match."
            )

    operations: list[HttpOperation] = []
    seen_operation_keys: set[tuple[str, str]] = set()
    for route in iter_route_contexts(routes):
        if not isinstance(route.original_route, APIRoute):
            continue
        methods = _non_safe_methods(route)
        if not methods:
            continue
        endpoint = route.endpoint
        path = route.path
        if endpoint is None or path is None:
            raise MutationPolicyConfigurationError(
                "FastAPI HTTP operations must expose an endpoint and path."
            )
        declarations = tuple(getattr(endpoint, _POLICY_DECLARATIONS_ATTRIBUTE, ()))
        if len(declarations) != 1:
            raise MutationPolicyConfigurationError(
                f"{','.join(methods)} {path} must declare exactly one "
                "HTTP operation policy."
            )
        policy = declarations[0]
        if not isinstance(policy, HttpOperationPolicy):
            raise MutationPolicyConfigurationError(
                f"{','.join(methods)} {path} declares an unknown HTTP operation policy."
            )
        if policy.is_mutation and "settings" not in signature(endpoint).parameters:
            raise MutationPolicyConfigurationError(
                f"{','.join(methods)} {path} must receive Workbench API "
                "settings for operational enforcement."
            )
        for method in methods:
            operation_key = (method, path)
            if operation_key in seen_operation_keys:
                raise MutationPolicyConfigurationError(
                    f"{method} {path} is registered more than once."
                )
            seen_operation_keys.add(operation_key)
            operations.append(HttpOperation(method=method, route=route, policy=policy))
    return HttpOperationCatalog(tuple(operations))


__all__ = [
    "HttpOperation",
    "HttpOperationCatalog",
    "HttpOperationPolicy",
    "MutationPolicyConfigurationError",
    "build_http_operation_catalog",
    "declare_http_operation",
    "enforce_operation_policy",
    "operation_policy_enabled",
]
