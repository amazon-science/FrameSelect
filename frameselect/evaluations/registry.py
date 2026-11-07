# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Registry for benchmark evaluators.

Mirrors the selector registry in :mod:`frameselect.registry`. Each
``BaseEvaluator`` subclass registers under a short dataset name
(e.g. ``"activitynet"``). :class:`frameselect.Evaluator`
looks up the class and constructs it with the caller's kwargs.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseEvaluator

_EVAL_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}


def register_evaluator(name: str):
    """Decorator: register an evaluator class under ``name``.

    Example
    -------
    >>> @register_evaluator("my_mcq")
    ... class MyMCQEvaluator(BaseEvaluator):
    ...     ...
    """

    def wrapper(cls: Type[BaseEvaluator]):
        _EVAL_REGISTRY[name] = cls
        return cls

    return wrapper


def list_evaluators() -> list[str]:
    """Return the sorted list of registered evaluator names."""
    return sorted(_EVAL_REGISTRY.keys())


def get_evaluator(name: str) -> Type[BaseEvaluator]:
    """Look up a registered evaluator class by short name."""
    if name not in _EVAL_REGISTRY:
        avail = ", ".join(list_evaluators())
        raise ValueError(f"Unknown evaluator '{name}'. Available: {avail}")
    return _EVAL_REGISTRY[name]
