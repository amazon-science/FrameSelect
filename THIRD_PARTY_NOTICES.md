# Third-party notices

FrameSelect re-implements published frame-selection methods. No module in
this repository adapts or copies code from a third-party repository. This
file lists the method each selector implements. To the best of our
knowledge, none of the four published methods (EFS, MEDR, AGFS, QAV) has a
public implementation from its authors. This was checked on 2026-10-14 in
five places: the paper's arXiv abstract page, every link in
the paper's PDF, its Hugging Face Papers entry
(`https://huggingface.co/api/papers/<arXiv id>`), the authors' project page
where the paper links one, and a GitHub repository search, READMEs included,
for the paper's title, method name and arXiv ID. The rest of this repository
is licensed as stated in `LICENSE`.

## Selectors

| Module (`frameselect/methods/`) | Method |
| --- | --- |
| `efs.py` | EFS, Event-Anchored Frame Selection (Chen et al., 2026) |
| `medr.py` | MEDR, Multi-Signal Event Modeling and Dynamic Rescoring (Pu et al., 2026) |
| `agfs.py` | AGFS, Adaptive Greedy Frame Selection (Huang et al., 2026) |
| `qav.py` | QAV, Query-Answer-Video aligned selection (Islam et al., 2026) |
| `uniform.py` | Evenly spaced sampling |
| `kmeans.py` | K-Means over frame embeddings |

## Evaluator

`frameselect/evaluations/activitynet.py` builds its multiple-choice questions
itself from the ActivityNet v1.3 annotations. It links the official
ActivityNet repository (https://github.com/activitynet/ActivityNet) only as
the source of the annotation file it downloads, and copies no code from it.

## Datasets and model weights

This repository contains no benchmark data and no model weights. The
ActivityNet evaluator downloads the v1.3 annotation file
(`activity_net.v1-3.min.json`) from the official ActivityNet GitHub
repository at runtime. It does not download videos: the dataset's authors
distribute them through the request form at
http://activity-net.org/download.html, and users pass their own copy as
`video_dir=`. Model checkpoints are downloaded at runtime: the OpenCLIP /
SigLIP encoders, the optional M-CLIP multilingual text tower, the
HuggingFace / vLLM answering models and, when `medr` runs with
`ocr="easyocr"`, EasyOCR's text detection and recognition models. Each
dataset and checkpoint keeps its own licence and terms of use; check them on
the source page before use.
