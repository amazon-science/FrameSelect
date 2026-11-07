# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Dataset support: loader + evaluator, one file per dataset.

Each submodule owns:

* ``load(...)`` — returns ``list[Example]`` from a local file or the
  dataset's official download location.
* A ``*Evaluator`` subclass — wraps the shared :class:`BaseEvaluator`
  with dataset-specific annotation parsing.

Currently supported datasets (1):

* :mod:`frameselect.evaluations.activitynet`       — ActivityNet v1.3 activity recognition (5-way MCQ, 1,000 val Qs)

Further datasets plug in through a :class:`BaseEvaluator` subclass
registered with :func:`register_evaluator`.

Typical usage::

    from frameselect.evaluations import activitynet, ActivityNetEvaluator
    examples = activitynet.load()
    ev = ActivityNetEvaluator(video_dir="/path/to/videos", annotations=examples)
    result = ev.run(selector)
"""

from .base import BaseEvaluator, Example, EvaluationResult
from .registry import get_evaluator, list_evaluators, register_evaluator
from . import activitynet
from .activitynet import ActivityNetEvaluator

__all__ = [
    # Shared bases
    "BaseEvaluator",
    "Example",
    "EvaluationResult",
    # Registry helpers (for custom evaluators)
    "register_evaluator",
    "list_evaluators",
    "get_evaluator",
    # Dataset submodules
    "activitynet",
    # Evaluator classes
    "ActivityNetEvaluator",
]
