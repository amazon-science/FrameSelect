<div align="center">

# FrameSelect

**One API for video frame selection: six selectors, one evaluator, one shared protocol.**

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-68448B.svg)](https://neurips.cc/Conferences/2026)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](LICENSE)
[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)](https://www.python.org/downloads/)

</div>

**FrameSelect** treats the frame selector as a standalone, interchangeable component, separate from the encoder that computes representations and the vision-language model (VLM) that answers. Candidate pool, encoder and frame budget are keyword arguments shared by every selector, and the answering VLM is one on the evaluator, so two methods compare under matched conditions with a one-line change.

This library accompanies our **NeurIPS 2026** paper (Evaluations & Datasets track). This release contains six selectors and one evaluator. Of the six, `uniform` and `kmeans` are among the selectors benchmarked in the paper; `efs`, `medr`, `agfs` and `qav` are additional re-implementations of published methods.

## Installation

Linux, Python 3.9 to 3.12. Install PyTorch and torchvision from the same index, matching your CUDA driver (skip that line on CPU-only machines):

```bash
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/amazon-science/FrameSelect.git && cd FrameSelect
pip install -e ".[viz]"         # or ".[all]" for every extra
```

Extras: `viz` (plots), `bedrock` (AWS Bedrock), `mclip` (multilingual text encoder), `cloud` (`s3://` / `gs://` URLs), `all`.

## Quickstart

```python
import frameselect as fs

sel = fs.Selector("efs", n_frames=16)
r = sel.select("video.mp4", query="What does the person assemble?")
r.frames, r.indices, r.all_scores   # picked frames, their frame indices, every candidate's score

# Benchmark it (videos from the ActivityNet authors; see below)
ev = fs.Evaluator("activitynet", video_dir="ActivityNet/videos",
                  vlm="openai:Qwen/Qwen2.5-VL-7B-Instruct")
res = ev.run(sel)
res.accuracy, res.subgroup_accuracy
res.save("results/")

# Replay the same picks with another answering VLM, without re-selecting
ev_qwen = fs.Evaluator("activitynet", video_dir="ActivityNet/videos",
                       vlm="hf:Qwen/Qwen2.5-VL-3B-Instruct")
ev_qwen.run(sel, selections_from="results/efs.json")  # written by res.save("results/")
```

`select()` accepts a path, a URL, a NumPy array, a torch tensor, a list of PIL images or a `decord.VideoReader`. The notebook [`examples/quickstart.ipynb`](examples/quickstart.ipynb) compares the six selectors on a video of your choice.

## Selectors

| Selector | Family | Query | Method |
|---|---|---|---|
| `uniform` | static | no | evenly spaced frames |
| `medr` | static | no | MEDR (Pu et al., 2026, [arXiv:2608.15058](https://arxiv.org/abs/2608.15058)) |
| `kmeans` | representation | no | K-Means over frame embeddings |
| `efs` | representation | yes | EFS (Chen et al., 2026, [arXiv:2603.00983](https://arxiv.org/abs/2603.00983)) |
| `agfs` | representation | yes | AGFS (Huang et al., 2026, [arXiv:2603.20180](https://arxiv.org/abs/2603.20180)) |
| `qav` | representation | yes, with answer choices | QAV (Islam et al., 2026, [arXiv:2609.31668](https://arxiv.org/abs/2609.31668)) |

The representation selectors encode with OpenCLIP `ViT-B-32` by default; the paper's encoder is `encoder="ViT-SO400M-16-SigLIP2-384", encoder_pretrained="webli"`. `qav` reads answer choices from the query, one per line, e.g. `query="What is the man holding?\nA. A cup.\nB. A phone."`. Each module's docstring lists where it differs from its paper. New selectors subclass `BaseSelector` and register with `@register("name")`.

## Evaluator

`activitynet` asks, for 1,000 ActivityNet v1.3 validation videos (5 per class), which of five activities from the same top-level category the video shows, and reports accuracy overall and per category. The annotation file is downloaded on first use; the videos come from the dataset authors' request form (http://activity-net.org/download.html), passed as `video_dir=`. [`examples/run_activitynet.py`](examples/run_activitynet.py) runs every selector under the paper's selection settings with a Qwen2.5-VL-7B answering model, so its accuracies are not comparable with the paper's tables; [`examples/README.md`](examples/README.md) covers the setup.

## Contributing

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citation

```bibtex
@inproceedings{jain2026frameselect,
  title     = {FrameSelect: A Unified Library for Video Frame Selection and Evaluation},
  author    = {Jain, Arihant and Dong, Chaosheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026},
}
```

## License

[CC BY-NC 4.0](LICENSE). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the methods implemented here. Datasets and model checkpoints keep their own licences and terms of use.
