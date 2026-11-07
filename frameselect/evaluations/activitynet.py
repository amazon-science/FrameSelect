# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""ActivityNet: activity-recognition MCQ loader + evaluator.

ActivityNet v1.3 (Caba Heilbron et al., CVPR 2015) labels untrimmed web
videos with one of 200 activity classes arranged in a taxonomy whose
top level has five categories. The dataset ships no questions, so this
evaluator asks the same recognition question of every video -- "Which
activity is shown in the video?" -- with five options: the video's own
label and four other ActivityNet labels from the same top-level
category, so every distractor is a plausible activity of the same kind.
The options and their order come from a per-video seed, so the question
set is identical on every run.

By default 5 single-label validation videos are drawn per class (1,000
questions, ``seed=0``); ``per_class=None`` keeps all 4,919 single-label
validation videos. The label's top-level category is the subgroup axis.

Data
----
* Annotations: ``activity_net.v1-3.min.json``, read from ``data_dir``
  when present, otherwise downloaded from the official ActivityNet
  repository (override with ``FRAMESELECT_ACTIVITYNET_ANNOTATIONS``).
* Videos: the dataset authors distribute them through the request form
  at http://activity-net.org/download.html. Put the validation videos
  under ``<data_dir>/videos`` (or pass ``video_dir=``); files named
  ``v_<id>.<ext>`` or ``<id>.<ext>`` are both found.

References
----------
* Paper:   https://openaccess.thecvf.com/content_cvpr_2015/html/Heilbron_ActivityNet_A_Large-Scale_2015_CVPR_paper.html
* Project: http://activity-net.org/
* GitHub:  https://github.com/activitynet/ActivityNet
"""

from __future__ import annotations

import json
import logging
import os
import random
import urllib.request
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from .base import BaseEvaluator, Example
from .registry import register_evaluator

logger = logging.getLogger(__name__)

ANNOTATION_FILENAME = "activity_net.v1-3.min.json"
ANNOTATION_URL = os.environ.get(
    "FRAMESELECT_ACTIVITYNET_ANNOTATIONS",
    "https://raw.githubusercontent.com/activitynet/ActivityNet/master/"
    "Evaluation/data/activity_net.v1-3.min.json",
)
REQUEST_FORM_URL = "http://activity-net.org/download.html"

QUESTION = "Which activity is shown in the video?"
N_OPTIONS = 5

# The five top-level taxonomy categories of ActivityNet v1.3, as spelled
# in the annotation file.
ACTIVITYNET_CATEGORIES = (
    "Eating and drinking Activities",
    "Household Activities",
    "Personal Care",
    "Socializing, Relaxing, and Leisure",
    "Sports, Exercise, and Recreation",
)

_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load(
    split: str = "validation",
    data_root: Optional["str | Path"] = None,
    *,
    per_class: Optional[int] = 5,
    seed: int = 0,
) -> list[Example]:
    """Build the activity-recognition MCQ examples.

    Parameters
    ----------
    split : str
        ``"validation"`` (default) or ``"training"``; the test split's
        labels are withheld.
    data_root : str or Path, optional
        Directory holding ``activity_net.v1-3.min.json`` (or the file
        itself). A downloaded copy is saved there for later runs.
    per_class : int or None
        Videos drawn per class; ``None`` keeps every single-label video.
    seed : int
        Seed for the per-class draw and for each video's distractors and
        option order.
    """
    if split not in ("validation", "training"):
        raise ValueError(f"split must be 'validation' or 'training'; got {split!r}")
    database, taxonomy = _read_annotations(data_root)
    category_of = _top_level_categories(taxonomy)

    by_label: dict[str, list[str]] = defaultdict(list)
    for vid, entry in database.items():
        if entry.get("subset") != split:
            continue
        labels = {seg["label"] for seg in entry.get("annotations", [])}
        if len(labels) == 1:  # skip the few videos with two activities
            by_label[labels.pop()].append(vid)

    labels_by_category: dict[str, list[str]] = defaultdict(list)
    for label in sorted(category_of):
        labels_by_category[category_of[label]].append(label)

    examples = []
    for label in sorted(by_label):
        vids = sorted(by_label[label])
        if per_class is not None and len(vids) > per_class:
            vids = sorted(random.Random(f"{seed}:{label}").sample(vids, per_class))
        category = category_of[label]
        for vid in vids:
            examples.append(_to_example(vid, database[vid], label, category,
                                        labels_by_category[category], seed))
    return examples


def _to_example(vid: str, entry: dict, label: str, category: str,
                category_labels: list[str], seed: int) -> Example:
    rng = random.Random(f"{seed}:{vid}")
    distractors = rng.sample([l for l in category_labels if l != label], N_OPTIONS - 1)
    options = distractors + [label]
    rng.shuffle(options)
    answer_idx = options.index(label)
    return Example(
        video_path=f"v_{vid}.mp4",
        question=QUESTION,
        options=[f"{chr(ord('A') + i)}. {opt}." for i, opt in enumerate(options)],
        answer=chr(ord("A") + answer_idx),
        question_id=vid,
        duration=category,  # subgroup slot holds the top-level category
        metadata={
            "video_id": vid,
            "label": label,
            "category": category,
            "video_duration_s": entry.get("duration"),
            "segments": [seg["segment"] for seg in entry.get("annotations", [])],
        },
    )


def _read_annotations(data_root: Optional["str | Path"]) -> tuple[dict, list]:
    """Return ``(database, taxonomy)`` from the v1.3 annotation file."""
    path = None
    if data_root is not None:
        root = Path(data_root).expanduser()
        path = root if root.is_file() else root / ANNOTATION_FILENAME
    if path is not None and path.exists():
        raw = json.loads(path.read_text())
    else:
        with urllib.request.urlopen(ANNOTATION_URL, timeout=60) as resp:
            raw = json.loads(resp.read())
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw))
    return raw["database"], raw["taxonomy"]


def _top_level_categories(taxonomy: list[dict]) -> dict[str, str]:
    """Map every leaf label to its top-level taxonomy category."""
    parent = {node["nodeName"]: node.get("parentName") for node in taxonomy}
    children = defaultdict(list)
    for name, par in parent.items():
        children[par].append(name)

    def top_of(name: str) -> str:
        while parent.get(name) not in (None, "Root"):
            name = parent[name]
        return name

    return {name: top_of(name) for name in parent
            if name != "Root" and not children.get(name)}


def _resolve_video_files(examples: list[Example], video_dir: Path, *,
                         skip_missing: bool) -> list[Example]:
    """Point each example at the file that exists for its video id."""
    found: dict[str, str] = {}
    if video_dir.is_dir():
        for p in video_dir.iterdir():
            if p.suffix.lower() in _VIDEO_EXTENSIONS:
                stem = p.stem[2:] if p.stem.startswith("v_") else p.stem
                found.setdefault(stem, p.name)
    resolved, missing = [], 0
    for ex in examples:
        name = found.get(ex.question_id)
        if name is None:
            missing += 1
            if skip_missing:
                continue
            resolved.append(ex)
        else:
            resolved.append(replace(ex, video_path=name))
    if missing:
        action = "skipped" if skip_missing else "will count as failed selections"
        logger.warning("ActivityNet: %d of %d videos not found in %s (%s)",
                       missing, len(examples), video_dir, action)
    return resolved


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@register_evaluator("activitynet")
class ActivityNetEvaluator(BaseEvaluator):
    """Evaluator for ActivityNet v1.3 activity recognition (5-way MCQ)."""

    subgroup_field = "duration"  # repurposed to hold the top-level category
    subgroup_values = ACTIVITYNET_CATEGORIES
    _default_data_dirname = "activitynet"

    def __init__(self, *, per_class: Optional[int] = 5, seed: int = 0,
                 skip_missing: bool = False, **kwargs):
        self._per_class = per_class
        self._seed = seed
        super().__init__(**kwargs)
        self.examples = _resolve_video_files(self.examples, self.video_dir,
                                             skip_missing=skip_missing)

    def load_annotations(self, path: str) -> Iterable[Example]:
        yield from load(data_root=path, per_class=self._per_class, seed=self._seed)

    def _load_default_annotations(self, data_dir: Path) -> list[Example]:
        return load(data_root=data_dir, per_class=self._per_class, seed=self._seed)

    def _download_videos(self, data_dir: Path) -> Path:
        video_dir = data_dir / "videos"
        if video_dir.is_dir() and any(video_dir.iterdir()):
            return video_dir
        raise RuntimeError(
            "ActivityNet videos are distributed by the dataset authors through "
            f"the request form at {REQUEST_FORM_URL}. Put the validation videos "
            f"under {video_dir} or pass video_dir=."
        )
