# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""String-factory entry points — ``Selector`` and ``Evaluator``.

These two classes are the canonical public entry points for most
users::

    from frameselect import Selector, Evaluator

    sel = Selector("efs",
                   n_frames=16,
                   encoder="ViT-SO400M-16-SigLIP2-384",
                   encoder_pretrained="webli")
    ev  = Evaluator("activitynet", video_dir="/path/to/videos",
                    vlm="openai:Qwen/Qwen2.5-VL-7B-Instruct")
    result = ev.run(sel)

Each class is a thin ``__new__`` wrapper that looks the name up in a
registry and returns the concrete subclass instance — so the returned
object is a real :class:`BaseSelector` / :class:`BaseEvaluator`, not a
proxy. Existing direct-class imports (``from frameselect.evaluations
import ActivityNetEvaluator``; ``fs.create("efs", ...)``) keep working
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseSelector
    from .evaluations.base import BaseEvaluator


class Selector:
    """String-named factory for registered frame selectors.

    ``Selector("efs", ...)`` is equivalent to ``fs.create("efs", ...)``
    — both call the same underlying registry. Use whichever style fits
    your codebase.

    Parameters
    ----------
    name : str
        Registered selector name. See :func:`frameselect.list_methods`
        for the full list.
    device : str
        ``"cpu"`` or ``"cuda"``. Default ``"cpu"``.
    **kwargs
        Unified axis controls (``n_frames``, ``candidate_frames``,
        ``candidate_fps``, ``encoder``, ``encoder_pretrained``, ``vlm``,
        ``seed``) plus any method-specific hyperparameters. See each
        selector's class docstring for the per-method knobs.

    Examples
    --------
    >>> from frameselect import Selector
    >>> sel = Selector("efs",
    ...                n_frames=16,
    ...                encoder="ViT-SO400M-16-SigLIP2-384",
    ...                encoder_pretrained="webli")
    >>> r = sel.select("video.mp4", query="what happens?")
    """

    def __new__(cls, name: str, device: str = "cpu", **kwargs) -> "BaseSelector":
        from .registry import get_method

        selector_cls = get_method(name)
        return selector_cls(device=device, **kwargs)

    @staticmethod
    def list() -> list[str]:
        """Return the sorted list of registered selector names."""
        from .registry import list_methods

        return list_methods()


class Evaluator:
    """String-named factory for registered benchmark evaluators.

    ``Evaluator("activitynet", ...)`` looks up ``ActivityNetEvaluator`` and
    constructs it with the caller's kwargs. Direct-class imports
    (``from frameselect.evaluations import ActivityNetEvaluator``) keep
    working for users who want static IDE autocomplete on per-dataset
    kwargs.

    Parameters
    ----------
    name : str
        Registered evaluator name. See :meth:`Evaluator.list` for the
        full list.
    **kwargs
        Constructor kwargs forwarded to the concrete evaluator class.
        Common kwargs: ``vlm`` (answering-VLM spec), ``cache_dir``,
        ``video_dir``, ``annotations``, ``data_dir``. Dataset-specific
        kwargs (``per_class`` and ``seed`` for ActivityNet) are documented
        on each concrete class.

    Examples
    --------
    >>> from frameselect import Selector, Evaluator
    >>> sel = Selector("efs", n_frames=16)
    >>> ev  = Evaluator("activitynet", video_dir="/path/to/videos",
    ...                 vlm="openai:Qwen/Qwen2.5-VL-7B-Instruct")
    >>> r   = ev.run(sel)

    ActivityNet with every single-label validation video::

        ev = Evaluator("activitynet",
                       video_dir="/path/to/videos",
                       per_class=None,
                       vlm="openai:Qwen/Qwen2.5-VL-7B-Instruct")
    """

    def __new__(cls, name: str, **kwargs) -> "BaseEvaluator":
        from .evaluations.registry import get_evaluator

        evaluator_cls = get_evaluator(name)
        return evaluator_cls(**kwargs)

    @staticmethod
    def list() -> list[str]:
        """Return the sorted list of registered evaluator names."""
        from .evaluations.registry import list_evaluators

        return list_evaluators()
