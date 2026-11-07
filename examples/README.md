# Examples

This directory holds a notebook that tours the library and a script that
benchmarks every registered selector on ActivityNet.

| Notebook | What it shows | Needs |
|---|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | all six selectors on one video of your choice, query switching, debug traces | CPU |

## Setup

Install the library from this checkout (see the top-level README), then:

* **Data root.** The script reads data and embedding caches from
  `$FRAMESELECT_ROOT/data/` and writes results to `benchmarks/` inside the
  checkout. `FRAMESELECT_ROOT` defaults to the checkout itself, i.e.
  `data/` in the repository root (ignored by git).
* **GPUs.** The script constructs its selectors with `device="cuda"`. Before
  any selector runs, it encodes every video missing from the SigLIP2
  embedding cache, sharded over the GPUs in `CUDA_VISIBLE_DEVICES`.
* **Answering VLM.** The script answers with `openai:Qwen/Qwen2.5-VL-7B-Instruct`,
  served by an OpenAI-compatible server at `OPENAI_BASE_URL` (probed on
  `localhost:8000` when unset). vLLM is not a dependency of the library:
  install it separately (`pip install vllm`) and start one server before the
  run, e.g. `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port 8000`, so every
  selector reuses it. Change `ANSWERING_VLM` in the script to use another
  backend.
* **Videos.** ActivityNet's videos come from the dataset authors' request form
  (http://activity-net.org/download.html); put the validation videos under
  `$FRAMESELECT_ROOT/data/activitynet/videos`. The annotation file is
  downloaded on first use.

## ActivityNet benchmark

[`run_activitynet.py`](run_activitynet.py) runs every registered selector on
the 1,000-question ActivityNet v1.3 activity-recognition set (5 single-label
validation videos per class, seed 0) under one shared configuration:
SigLIP2-SO400M-384, 128 candidates at 1 FPS, 16 output frames,
library-default seeds. Every query-aware selector gets the question; `qav`,
which reads answer choices, gets the five options as well, appended by the
evaluator one per line. Results go to
`benchmarks/activitynet__siglip2__128cand__16f__qwen25vl7b/`, one JSON per
selector plus a `_summary.json`. The run is resumable: a selector whose result
JSON already exists is skipped. The selection settings match the paper's
shared configuration, but the answering VLM differs from the paper's, so the
accuracies are not comparable with the paper's tables.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_activitynet.py            # every selector
CUDA_VISIBLE_DEVICES=0 python examples/run_activitynet.py efs qav    # just these
```
