# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""MEDR: Multi-Signal Event Modeling and Dynamic Rescoring.

Training-free, query-independent selector: the frame set depends only on the
video, so one selection can be reused for every question about it. Two
stages:

1. **Multi-signal event modeling.** Three per-frame signals are turned into
   signal-specific temporal events ``e = (I_e, c_e, s_e)`` (interval,
   representative frame, strength):

   * *visual* — an adjacent-frame change signal ``d_i`` (colour-histogram
     distance, 1 − SSIM and perceptual-hash distance) separates stable
     intervals at strong local changes; each interval is represented by its
     most stable frame, ``argmax p_i`` with ``p_i = 1 − Norm(d_i)``;
   * *motion* — dense optical flow (Farnebäck) gives ``m_i``; runs of frames
     above a robust per-video threshold are motion events, represented by
     their strongest frame;
   * *text* (optional, needs an OCR backend) — OCR quality plus the
     appearance or change of recognised text gives ``x_i``; consecutive
     frames with similar text and an unchanged numerical state form one
     event.

   Event priorities are ``ℓ_e = log(1 + max(s_e, 0))`` and every frame gets
   a multi-signal score ``f_i = mean(R(p_i), R(m_i), R(x_i))`` with ``R``
   the within-video rank normalisation.
2. **Dynamic rescoring.** A candidate set (``B'`` uniform frames, then event
   representatives by priority, then frames by ``f_i``, up to
   ``max(candidate_limit, B')``) is filled greedily by the utility

   ``u_i = [ρ f_i + (1 − ρ) g_i] · [η + (1 − η)(1 − r_i)]``

   where ``g_i`` is the additional event coverage the candidate brings over
   the frames already chosen (event–frame affinities combine temporal
   proximity, lightweight visual or text compatibility and the event type's
   own signal) and ``r_i = max_j exp(−B'/D · |t_i − t_j|)`` its temporal
   redundancy with them. Coverage and redundancy are updated after every
   pick.

Reference
---------
Pu et al., "MEDR: Query-Independent Frame Selection via Multi-Signal Event
Modeling and Dynamic Rescoring", 2026.
Paper: https://arxiv.org/abs/2608.15058
Code: Not publicly available, to the best of our knowledge

Text signal
-----------
To the best of our knowledge, the paper does not name its OCR engine. ``ocr=None`` (the default) runs
MEDR on the visual and motion signals only, which the paper's ablation
(Table 3) reports as 1.37 points lower on Video-MME at a 64-frame budget.
Pass ``ocr="easyocr"`` (requires ``pip install easyocr``) or any callable
mapping an RGB ``uint8`` frame to ``[(text, confidence), ...]`` to add it.

Details the paper leaves open
-----------------------------
* ``d_i`` is the mean of the Hellinger distance between HSV histograms
  (8×4×4 bins), ``1 − SSIM`` (11×11 Gaussian window) and the pHash Hamming
  distance / 64, all on frames resized to a 160-pixel longer side;
  ``d_0 = 0``. ``m_i`` is the mean Farnebäck flow magnitude at that size
  (``m_0 = m_1``).
* Visual boundaries are local maxima of ``d`` above
  ``median + 3 · max(1.4826 · MAD, 0.02)``; motion events are runs above
  ``median + 3 · max(1.4826 · MAD, 0.05)``.
* Strengths: visual = interval length × mean ``p``; motion = summed
  ``m − median(m)``; text = the strongest ``x`` in the event.
* ``x_i = q_i · (1 + c_i) / 2`` with ``q_i`` the mean OCR confidence and
  ``c_i = 1`` when text appears or changes; text counts as similar at a
  ``difflib`` ratio of at least 0.8.
* Lightweight visual similarity is ``1 − (Hamming / 64 + Hellinger) / 2``;
  the text affinity's compatibility term is 1 when the candidate's text is
  similar to the event's, else 0, and its signal term is ``q_i``.
* Times are candidate positions, so the sampling interval is 1 and an
  event's ``σ_e`` is ``max(duration, 1)``.
* The paper tunes ``(ρ, η)`` for 32- and 64-frame budgets only (to the best of our knowledge), as
  ``(0.5, 0.1)`` and ``(0.8, 0.2)``; with ``rho``/``eta`` left at ``None``
  budgets below 48 frames use the first and the rest the second.
"""

from __future__ import annotations

import difflib
import re
import threading
from typing import Callable, Optional, Union

import numpy as np

from ..base import BaseSelector
from ..registry import register

# Longer side, in pixels, of the frames the pixel signals are computed on.
_WORK_SIDE = 160
# Text counts as "the same" at or above this difflib similarity ratio.
_TEXT_SIMILAR = 0.8

OCRFn = Callable[[np.ndarray], "list[tuple[str, float]]"]


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv is a core dependency
        raise ImportError(
            "MEDR needs OpenCV: pip install opencv-python-headless"
        ) from exc
    return cv2


# ---------------------------------------------------------------------------
# Per-frame pixel features
# ---------------------------------------------------------------------------

def _resize_long(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale >= 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return _cv2().resize(img, size, interpolation=_cv2().INTER_AREA)


def _phash(gray: np.ndarray) -> np.ndarray:
    """64-bit DCT perceptual hash (bool array of length 64)."""
    cv2 = _cv2()
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    return low > np.median(low[1:])


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM of two grey images (11×11 Gaussian window, σ = 1.5)."""
    cv2 = _cv2()
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def _pixel_features(frames: np.ndarray):
    """Grey working images, pHash bits ``(T, 64)`` and HSV histograms ``(T, 128)``."""
    cv2 = _cv2()
    grays, hashes, hists = [], [], []
    for frame in frames:
        small = _resize_long(np.ascontiguousarray(frame), _WORK_SIDE)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        grays.append(gray)
        hashes.append(_phash(gray))
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                            [0, 180, 0, 256, 0, 256]).ravel()
        hists.append(hist / (hist.sum() + 1e-12))
    return grays, np.asarray(hashes, dtype=bool), np.asarray(hists, dtype=np.float64)


def _hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hellinger distance between normalised histograms (broadcasts on rows)."""
    bc = np.sqrt(p * q).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))


def _visual_similarity(hashes: np.ndarray, hists: np.ndarray,
                       cands: np.ndarray) -> np.ndarray:
    """Lightweight similarity ``(|C|, T)``: 1 − (Hamming / 64 + Hellinger) / 2.

    Both terms are matrix products, so a long 1-fps pool stays cheap.
    """
    h = hashes.astype(np.float32)
    hc = h[cands]
    hamming = (hc @ (1.0 - h).T + (1.0 - hc) @ h.T) / 64.0
    root = np.sqrt(hists)
    hellinger = np.sqrt(np.clip(1.0 - root[cands] @ root.T, 0.0, 1.0))
    return 1.0 - (hamming + hellinger) / 2.0


def _visual_change(grays, hashes, hists) -> np.ndarray:
    T = len(grays)
    d = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        hist_d = float(_hellinger(hists[i - 1], hists[i]))
        ssim_d = float(np.clip(1.0 - _ssim(grays[i - 1], grays[i]), 0.0, 1.0))
        hash_d = float(np.count_nonzero(hashes[i - 1] != hashes[i])) / 64.0
        d[i] = (hist_d + ssim_d + hash_d) / 3.0
    return d


def _motion(grays) -> np.ndarray:
    cv2 = _cv2()
    T = len(grays)
    m = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        m[i] = float(np.linalg.norm(flow, axis=2).mean())
    if T > 1:
        m[0] = m[1]
    return m


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _rank01(x: np.ndarray) -> np.ndarray:
    """Rank normalisation to ``[0, 1]``; ties share their average rank."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    if T <= 1:
        return np.zeros(T, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(T, dtype=np.float64)
    ranks[order] = np.arange(T, dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ranks = np.bincount(inverse, weights=ranks)[inverse] / counts[inverse]
    return ranks / (T - 1)


def _robust_threshold(x: np.ndarray, floor: float, k: float = 3.0) -> float:
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    return med + k * max(mad, floor)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` of every run of True in ``mask``."""
    runs, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _normalise_text(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def _text_similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= _TEXT_SIMILAR


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class _Event:
    __slots__ = ("kind", "start", "end", "rep", "strength", "text")

    def __init__(self, kind, start, end, rep, strength, text=""):
        self.kind, self.start, self.end = kind, int(start), int(end)
        self.rep, self.strength, self.text = int(rep), float(strength), text

    @property
    def priority(self) -> float:
        return float(np.log1p(max(self.strength, 0.0)))


def _visual_events(d: np.ndarray, p: np.ndarray) -> list[_Event]:
    T = len(d)
    thr = _robust_threshold(d[1:] if T > 1 else d, floor=0.02)
    bounds = [
        i for i in range(1, T)
        if d[i] > thr and d[i] >= d[i - 1] and (i == T - 1 or d[i] >= d[i + 1])
    ]
    starts = [0] + bounds
    events = []
    for s, e in zip(starts, starts[1:] + [T]):
        rep = s + int(np.argmax(p[s:e]))
        events.append(_Event("visual", s, e - 1, rep, (e - s) * float(p[s:e].mean())))
    return events


def _motion_events(m: np.ndarray) -> list[_Event]:
    background = float(np.median(m))
    thr = _robust_threshold(m, floor=0.05)
    events = []
    for s, e in _runs(m > thr):
        rep = s + int(np.argmax(m[s:e + 1]))
        events.append(_Event("motion", s, e, rep, float((m[s:e + 1] - background).sum())))
    return events


def _text_signal(ocr_out: list) -> tuple:
    """``(x, q, texts, numbers)`` from per-frame OCR detections."""
    T = len(ocr_out)
    q = np.zeros(T, dtype=np.float64)
    texts, numbers = [""] * T, [()] * T
    for i, detections in enumerate(ocr_out):
        kept = [(str(t), float(c)) for t, c in detections if _normalise_text(str(t))]
        if not kept:
            continue
        q[i] = float(np.mean([c for _, c in kept]))
        raw = " ".join(t for t, _ in kept)
        texts[i] = _normalise_text(raw)
        numbers[i] = tuple(re.findall(r"\d+(?:[.,]\d+)?", raw))
    x = np.zeros(T, dtype=np.float64)
    for i in range(T):
        if not texts[i]:
            continue
        changed = (
            i == 0 or not texts[i - 1]
            or not _text_similar(texts[i - 1], texts[i])
            or numbers[i - 1] != numbers[i]
        )
        x[i] = q[i] * (1.0 + float(changed)) / 2.0
    return x, q, texts, numbers


def _text_events(x: np.ndarray, texts: list, numbers: list) -> list[_Event]:
    events, start = [], None

    def close(end):
        rep = start + int(np.argmax(x[start:end + 1]))
        events.append(_Event("text", start, end, rep, float(x[start:end + 1].max()),
                             text=texts[rep]))

    for i, text in enumerate(texts):
        if not text:
            if start is not None:
                close(i - 1)
                start = None
            continue
        if start is not None and (
            not _text_similar(texts[i - 1], text) or numbers[i - 1] != numbers[i]
        ):
            close(i - 1)
            start = None
        if start is None:
            start = i
    if start is not None:
        close(len(texts) - 1)
    return events


# ---------------------------------------------------------------------------
# Candidate set, affinities and dynamic rescoring
# ---------------------------------------------------------------------------

def _candidate_set(T: int, budget: int, limit: int, events: list[_Event],
                   f: np.ndarray) -> np.ndarray:
    L = min(T, max(limit, budget))
    if T <= L:
        return np.arange(T, dtype=np.int64)
    chosen: dict[int, None] = {}
    for i in np.linspace(0, T - 1, budget).round().astype(np.int64):
        chosen[int(i)] = None
    for ev in sorted(events, key=lambda e: -e.priority):
        if len(chosen) >= L:
            break
        chosen.setdefault(ev.rep, None)
    for i in np.argsort(-f, kind="stable"):
        if len(chosen) >= L:
            break
        chosen.setdefault(int(i), None)
    return np.sort(np.fromiter(chosen, dtype=np.int64))


def _affinities(events, cands, p, m_norm, q, texts, vis_sim):
    """Normalised event–candidate affinity matrix ``(E, |C|)`` and weights ``(E,)``."""
    E, C = len(events), len(cands)
    A = np.zeros((E, C), dtype=np.float64)
    t = cands.astype(np.float64)
    for k, ev in enumerate(events):
        dist = np.where(t < ev.start, ev.start - t, np.where(t > ev.end, t - ev.end, 0.0))
        sigma = max(float(ev.end - ev.start), 1.0)
        temporal = np.exp(-dist / sigma)
        if ev.kind == "text":
            content = np.array([1.0 if texts[c] and _text_similar(texts[c], ev.text) else 0.0
                                for c in cands])
            signal = q[cands]
        else:
            content = vis_sim[:, ev.start:ev.end + 1].max(axis=1)
            signal = p[cands] if ev.kind == "visual" else m_norm[cands]
        raw = temporal * content * signal
        top = float(raw.max())
        if top > 0:
            A[k] = raw / top
    active = A.max(axis=1) > 0
    priority = np.array([ev.priority for ev in events], dtype=np.float64)
    priority[~active] = 0.0
    total = float(priority.sum())
    weights = priority / total if total > 0 else priority
    return A, weights


def _dynamic_rescoring(cands, f_c, A, weights, budget, rho, eta) -> np.ndarray:
    t = cands.astype(np.float64)
    span = float(t.max() - t.min()) if len(t) else 0.0
    rate = budget / span if span > 0 else 0.0
    coverage = np.zeros(A.shape[0], dtype=np.float64)
    redundancy = np.zeros(len(cands), dtype=np.float64)
    available = np.ones(len(cands), dtype=bool)
    picked = []
    for _ in range(min(budget, len(cands))):
        gain = weights @ np.maximum(A - coverage[:, None], 0.0)
        utility = (rho * f_c + (1.0 - rho) * gain) * (eta + (1.0 - eta) * (1.0 - redundancy))
        utility[~available] = -np.inf
        k = int(np.argmax(utility))
        picked.append(k)
        available[k] = False
        coverage = np.maximum(coverage, A[:, k])
        redundancy = np.maximum(redundancy, np.exp(-rate * np.abs(t - t[k])))
    return np.sort(cands[picked])


def _budget_defaults(budget: int) -> tuple[float, float]:
    return (0.5, 0.1) if budget < 48 else (0.8, 0.2)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

@register("medr")
class MEDRSelector(BaseSelector):
    """MEDR: query-independent multi-signal events + dynamic rescoring.

    Parameters
    ----------
    rho : float, optional
        Weight ``ρ`` of the frame-level score against the additional event
        coverage. ``None`` (default) uses the paper's budget-specific value:
        ``0.5`` below 48 frames, ``0.8`` from 48 frames.
    eta : float, optional
        Lower bound ``η`` of the temporal-redundancy factor. ``None``
        (default) uses ``0.1`` below 48 frames and ``0.2`` from 48 frames.
    candidate_limit : int
        Maximum candidate-set size ``M``. Paper default ``512``. Pools no
        larger than ``max(candidate_limit, n_frames)`` are used whole.
    ocr : {None, "easyocr"} or callable, optional
        Backend for the text signal: ``None`` (default) disables it,
        ``"easyocr"`` uses EasyOCR (``pip install easyocr``), and a callable
        maps an RGB ``uint8`` frame to ``[(text, confidence), ...]``.
    ocr_languages : tuple of str
        Languages passed to EasyOCR. Default ``("en",)``.
    """

    query_aware = False

    def __init__(
        self,
        device: str = "cpu",
        rho: Optional[float] = None,
        eta: Optional[float] = None,
        candidate_limit: int = 512,
        ocr: Union[None, str, OCRFn] = None,
        ocr_languages: tuple = ("en",),
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        for name, value in (("rho", rho), ("eta", eta)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if candidate_limit < 1:
            raise ValueError(f"candidate_limit must be >= 1, got {candidate_limit}")
        if not (ocr is None or callable(ocr) or ocr == "easyocr"):
            raise ValueError(f"ocr must be None, 'easyocr' or a callable, got {ocr!r}")
        self.rho = rho
        self.eta = eta
        self.candidate_limit = candidate_limit
        self.ocr = ocr
        self.ocr_languages = tuple(ocr_languages)
        self._ocr_fn: Optional[OCRFn] = None
        self._ocr_lock = threading.Lock()

    def _hyperparams(self) -> dict:
        ocr = self.ocr if self.ocr is None or isinstance(self.ocr, str) else "callable"
        return {
            "rho": self.rho,
            "eta": self.eta,
            "candidate_limit": self.candidate_limit,
            "ocr": ocr,
        }

    def _get_ocr(self) -> Optional[OCRFn]:
        if self.ocr is None:
            return None
        if callable(self.ocr):
            return self.ocr
        with self._ocr_lock:
            if self._ocr_fn is None:
                try:
                    import easyocr
                except ImportError as exc:
                    raise ImportError(
                        "ocr='easyocr' needs EasyOCR: pip install easyocr"
                    ) from exc
                reader = easyocr.Reader(list(self.ocr_languages),
                                        gpu=str(self.device).startswith("cuda"))
                self._ocr_fn = lambda frame: [
                    (text, float(conf)) for _box, text, conf in reader.readtext(frame)
                ]
        return self._ocr_fn

    def _select_indices(
        self,
        frames: np.ndarray,
        n_select: int,
        query: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        del query  # query-independent by design
        T = len(frames)
        budget = min(n_select, T)
        rho, eta = _budget_defaults(budget)
        rho = rho if self.rho is None else self.rho
        eta = eta if self.eta is None else self.eta

        grays, hashes, hists = _pixel_features(frames)
        d = _visual_change(grays, hashes, hists)
        p = 1.0 - _minmax(d)
        m = _motion(grays)
        events = _visual_events(d, p) + _motion_events(m)
        signals = [p, m]
        q = np.zeros(T, dtype=np.float64)
        texts = [""] * T
        traces = {
            "visual_stability": p,
            "motion": m,
        }
        ocr = self._get_ocr()
        if ocr is not None:
            x, q, texts, numbers = _text_signal([ocr(np.ascontiguousarray(fr)) for fr in frames])
            events += _text_events(x, texts, numbers)
            signals.append(x)
            traces["text"] = x
        f = np.mean([_rank01(s) for s in signals], axis=0)

        cands = _candidate_set(T, budget, self.candidate_limit, events, f)
        vis_sim = _visual_similarity(hashes, hists, cands)
        A, weights = _affinities(events, cands, p, _minmax(m), q, texts, vis_sim)
        indices = _dynamic_rescoring(cands, f[cands], A, weights, budget, rho, eta)

        labels = {"visual_stability": "Visual stability p", "motion": "Motion m",
                  "text": "Text signal x"}
        self._pending_debug_traces = {
            name: {"values": np.asarray(v, dtype=np.float32), "label": labels[name],
                   "kind": "curve", "higher_is_better": True}
            for name, v in traces.items()
        }
        scores = f.astype(np.float32)
        return indices, scores[indices], scores, "custom"
