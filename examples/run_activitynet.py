# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Benchmark every registered selector on ActivityNet activity recognition.

Runs each selector on the 1,000-question ActivityNet v1.3 activity-recognition
set (5 single-label validation videos per class, seed 0; see
:mod:`frameselect.evaluations.activitynet`) under one shared configuration,
with every axis pinned explicitly (the answering VLM included) and every
result saved. Existing result files are skipped.

ActivityNet's videos come from the dataset authors' request form
(http://activity-net.org/download.html): put the validation videos under
``$FRAMESELECT_ROOT/data/activitynet/videos``. The annotation file is
downloaded on first use. The SigLIP2 embedding cache is built first, on
the GPUs in ``CUDA_VISIBLE_DEVICES``.

Usage::

    CUDA_VISIBLE_DEVICES=0 python examples/run_activitynet.py
    CUDA_VISIBLE_DEVICES=0 python examples/run_activitynet.py efs qav

Naming selectors runs only those.

Data and caches live under ``$FRAMESELECT_ROOT/data`` (default: this
checkout); results go to ``benchmarks/`` in the checkout. The answering VLM
is an OpenAI-compatible server (e.g. vLLM) at ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from frameselect import Evaluator, Selector, list_methods

ROOT = Path(os.environ.get("FRAMESELECT_ROOT", REPO))
DATA = ROOT / "data" / "activitynet"
BENCH = REPO / "benchmarks"

ANSWERING_VLM = "openai:Qwen/Qwen2.5-VL-7B-Instruct"

# Fail fast if the answering server is not up, before creating any
# directories, downloading annotations or encoding anything.
if ANSWERING_VLM.startswith("openai:"):
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    try:
        urllib.request.urlopen(f"{BASE_URL}/models", timeout=5).close()
    except Exception as e:
        sys.exit(f"No OpenAI-compatible server at {BASE_URL} ({type(e).__name__}). Start one first, e.g.\n"
                 f"  vllm serve {ANSWERING_VLM.split(':', 1)[1]} --host 127.0.0.1 --port 8000\n"
                 f"or point OPENAI_BASE_URL at a running server.")

OUTPUT_DIR = BENCH / "activitynet__siglip2__128cand__16f__qwen25vl7b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER, ENCODER_PRETRAINED = "ViT-SO400M-16-SigLIP2-384", "webli"

ev = Evaluator(
    "activitynet",
    vlm=ANSWERING_VLM,
    data_dir=str(DATA),
    video_dir=str(DATA / "videos"),
    cache_dir=str(DATA / "cache_siglip2"),
    per_class=5,
    seed=0,
)

METHODS = list_methods()

SELECTOR_DEFAULTS = dict(
    n_frames=16,
    candidate_frames=128,
    encoder=ENCODER,
    encoder_pretrained=ENCODER_PRETRAINED,
)

N_WORKERS = 32

print(f"answering VLM: {ANSWERING_VLM}")
print(f"questions:     {len(ev.examples)}")
print(f"output dir:    {OUTPUT_DIR}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

# Shards over every GPU in CUDA_VISIBLE_DEVICES (the library default).
n_new = ev.precompute_cache(encoder=ENCODER, encoder_pretrained=ENCODER_PRETRAINED,
                            candidate_frames=128)
print(f"embedding cache: {n_new} videos encoded", flush=True)

requested = sys.argv[1:] or METHODS
unknown = sorted(set(requested) - set(METHODS))
if unknown:
    sys.exit(f"unknown selector(s): {', '.join(unknown)}")

for method in requested:
    out_path = OUTPUT_DIR / f"{method}.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        print(f"SKIP {method:10s} — {data.get('accuracy', '?')}%", flush=True)
        continue

    print(f"\n{'=' * 60}\n{method}\n{'=' * 60}", flush=True)
    try:
        sel = Selector(method, device="cuda", **SELECTOR_DEFAULTS)
        t0 = time.time()
        result = ev.run(sel, n_workers=N_WORKERS)
        saved = result.save(OUTPUT_DIR)
        print(result, flush=True)
        print(f"  -> {saved}  ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:  # keep going so one bad selector doesn't stall the sweep
        print(f"  FAILED {method}: {type(e).__name__}: {e}", flush=True)

summary = {}
for method in METHODS:
    path = OUTPUT_DIR / f"{method}.json"
    if path.exists():
        r = json.loads(path.read_text())
        summary[method] = {k: r.get(k) for k in
                           ("accuracy", "correct", "total", "n_errors", "subgroup_accuracy")}
(OUTPUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {OUTPUT_DIR / '_summary.json'}")
