# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Visualization utilities for frame selection results.

Three public entry points, each answering a distinct question:

- :func:`score_timeline` — "why did this method pick here?" The
  full per-candidate score curve with picks overlaid.
- :func:`selection_strip` — "where on the timeline did each method
  pick?" One row per method on a shared time axis, with thumbnails
  under each pick and optional query/options/answer header.
- :func:`debug_trajectory` — "what internal signals drove those
  picks?" Renders whatever per-candidate traces a selector opts
  into via ``SelectionResult.metadata["debug_traces"]``.

Design note — **everything renders inline out of the box**:

- :func:`selection_strip` returns a :class:`PIL.Image.Image`, which
  Jupyter displays automatically via ``_repr_png_``.
- :func:`score_timeline` and :func:`debug_trajectory` return a
  :class:`matplotlib.figure.Figure`. The module does **not** force
  the Matplotlib Agg backend, so the figure renders inline in
  Jupyter and pops up interactively in a CLI session with no
  extra ``plt.show()`` needed.

All three take a ``save_path=`` kwarg for saving to disk.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import SelectionResult


# ---------------------------------------------------------------------------
# Matplotlib backend: only force Agg when we're genuinely headless.
# Forcing Agg unconditionally (as an earlier version did) broke inline
# rendering in Jupyter: the figure object was still returned correctly,
# but the cell produced no visible output until the caller wrapped it
# in ``fig.canvas.draw(); display(fig)``. Users expect
# ``fs.viz.score_timeline(result)`` to just render.
# ---------------------------------------------------------------------------

def _ensure_matplotlib():
    """Import pyplot, selecting a safe backend only when headless."""
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for frameselect.viz. "
            "Install it with: pip install matplotlib"
        ) from e
    # Only fall back to Agg if the process is headless (no DISPLAY,
    # no MPLBACKEND, no running ipykernel). Otherwise respect the
    # backend that pyplot will auto-select (inline for Jupyter, Qt/TK
    # for a GUI, etc.).
    if (
        "MPLBACKEND" not in os.environ
        and "DISPLAY" not in os.environ
        and "ipykernel" not in _sys_modules()
    ):
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass
    import matplotlib.pyplot as plt
    return plt


def _sys_modules():
    import sys
    return sys.modules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_font(size: int = 12):
    """Try to load a TTF font, fall back to default."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _frame_to_pil(frame: np.ndarray, width: int) -> Image.Image:
    """Resize a frame (H,W,3 uint8) to target width, preserving aspect."""
    img = Image.fromarray(frame)
    ratio = width / img.width
    new_h = int(img.height * ratio)
    return img.resize((width, new_h), Image.LANCZOS)


def _index_to_timestamp(index: int, fps: float) -> str:
    """Convert frame index to mm:ss string."""
    if fps <= 0:
        return str(index)
    secs = index / fps
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}"


def _chars_for_width(pixel_w: int, font) -> int:
    """Rough estimate of how many characters fit in ``pixel_w`` given ``font``.

    Uses the width of 'x' as a proxy for an average glyph width; good
    enough for plain ASCII wrapping on the selection-strip header.
    """
    try:
        bbox = font.getbbox("x")
        char_w = max(1, bbox[2] - bbox[0])
    except Exception:
        char_w = 7  # fallback for default bitmap font
    return max(10, pixel_w // char_w)


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Greedy whitespace wrap; preserves explicit newlines."""
    out: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split(" ")
        line = ""
        for w in words:
            cand = (line + " " + w).strip()
            if len(cand) <= max_chars:
                line = cand
            else:
                if line:
                    out.append(line)
                # Break an over-long word at max_chars
                while len(w) > max_chars:
                    out.append(w[:max_chars])
                    w = w[max_chars:]
                line = w
        if line:
            out.append(line)
    return out or [""]


def _match_answer_to_option(
    answer: Optional[str], options: Optional[Sequence[str]]
) -> int:
    """Return the index of the option that matches ``answer``, else -1.

    Accepts either:
    * a full option string (``"A. Backflip."``) — exact match;
    * just the letter (``"A"`` or ``"A."``) — matches any option that
      starts with the same leading letter-then-delimiter.
    """
    if not answer or not options:
        return -1
    ans = answer.strip()
    for i, opt in enumerate(options):
        if opt.strip() == ans:
            return i
    # Letter-only case: compare the leading letter of answer to each option.
    head = ans.rstrip(".").strip()
    if len(head) == 1 and head.isalpha():
        letter = head.upper()
        for i, opt in enumerate(options):
            o = opt.lstrip().upper()
            if o.startswith(letter + ".") or o.startswith(letter + ")") or o.startswith(letter + " "):
                return i
    return -1


# ---------------------------------------------------------------------------
# Score Timeline
# ---------------------------------------------------------------------------

def score_timeline(
    result: SelectionResult,
    all_scores: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 3.5),
) -> "matplotlib.figure.Figure":
    """Plot the similarity score curve with selected frames highlighted.

    Parameters
    ----------
    result : SelectionResult
        Output from a selector's ``.select()`` call.
    all_scores : np.ndarray, optional
        Full score array for all candidate frames (length = n_candidates).
        If ``None``, uses ``result.all_scores`` when available. If neither
        is set, only the selected-frame markers are shown.
    title : str, optional
        Plot title.
    save_path : str, optional
        If set, save the figure to this path.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Prefer an explicit arg, fall back to the new explainer field.
    if all_scores is None:
        all_scores = result.all_scores
    plt = _ensure_matplotlib()

    # ``native_fps``/``candidate_fps_used`` are present-but-``None`` for
    # in-memory inputs (the loader records ``None`` when it can't know the
    # rate), so ``.get(key, default)`` returns ``None`` there — and
    # ``None > 0`` raises. Coalesce to 1.0 (mirrors ``debug_trajectory``).
    known_fps = result.metadata.get("native_fps")
    fps = known_fps or 1.0
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot full score curve if available
    if all_scores is not None:
        candidate_fps = result.metadata.get("candidate_fps_used") or 1.0
        n_cand = len(all_scores)
        cand_idx = result.candidate_indices
        if cand_idx is not None and len(cand_idx) == n_cand:
            # Real candidate times. A pool capped at ``candidate_frames``
            # is spaced wider than 1 / candidate_fps, so spacing the curve
            # at that rate would misplace it against the picks below.
            t_cand = np.asarray(cand_idx, dtype=np.float64) / fps
        elif candidate_fps > 0:
            t_cand = np.arange(n_cand) * (1.0 / candidate_fps)
        else:
            t_cand = np.arange(n_cand)
        ax.plot(t_cand, all_scores, color="#90CAF9", linewidth=0.8, alpha=0.8, label="All candidates")
        ax.fill_between(t_cand, all_scores, alpha=0.15, color="#90CAF9")

    # Plot selected frames
    t_selected = result.indices / fps if fps > 0 else result.indices.astype(float)
    if result.scores is not None:
        ax.scatter(t_selected, result.scores, color="#E53935", s=60, zorder=5,
                   edgecolors="white", linewidths=0.8, label="Selected")
        # Vertical lines
        for t, s in zip(t_selected, result.scores):
            ax.axvline(t, color="#E53935", alpha=0.2, linewidth=0.8)
    else:
        # No scores — just mark positions on x-axis
        y_pos = np.zeros_like(t_selected)
        ax.scatter(t_selected, y_pos, color="#E53935", s=60, zorder=5,
                   edgecolors="white", linewidths=0.8, label="Selected")
        for t in t_selected:
            ax.axvline(t, color="#E53935", alpha=0.2, linewidth=0.8)

    # Without the video's frame rate the positions are frame indices.
    ax.set_xlabel("Time (seconds)" if known_fps else "Frame index")
    ax.set_ylabel("Score" if result.score_kind == "custom" else "Similarity score")
    ax.legend(loc="upper right", fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Selection strip — one row per method, shared time axis, thumbnails
# ---------------------------------------------------------------------------

def selection_strip(
    results: dict[str, SelectionResult],
    *,
    query: Optional[str] = None,
    options: Optional[Sequence[str]] = None,
    answer: Optional[str] = None,
    timeline: bool = False,
    thumb_width: int = 96,
    row_pad: int = 10,
    left_gutter: int = 140,
    right_gutter: int = 20,
    axis_height: int = 22,
    save_path: Optional[str] = None,
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (30, 30, 30),
    dot_color: tuple = (229, 57, 53),
    font_scale: float = 1.0,
    canvas_width: Optional[int] = None,
) -> Image.Image:
    """One row per method, with selected-frame thumbnails.

    Two layout modes, switched with ``timeline``:

    * ``timeline=False`` (default) — lays thumbnails out in a regular
      grid, one cell per pick, evenly spaced across the row. No
      temporal scaling, no overlap — the right choice when you want
      to inspect the selected frames themselves. The bottom time
      axis is hidden in this mode.
    * ``timeline=True`` — places every method's picks on a shared
      time axis so a reader sees *where* each method clustered its
      picks on the same video. Thumbnails sit below the dots at
      their temporal positions, so closely-spaced picks can overlap
      when the video is short or ``thumb_width`` is large. Best for
      studying how selectors cluster in time.

    Parameters
    ----------
    results : dict[str, SelectionResult]
        ``{method_name: result}``. All results must be from the same
        video (we don't enforce this, but the time axis assumes it).
    query : str, optional
        Query text to display as a header above the strip. Useful for
        query-aware methods so the reader sees what the selector was
        targeting.
    options : sequence of str, optional
        MCQ options to list under the query (e.g. ``["A. Backflip.",
        "B. Cartwheel.", ...]``). If ``answer`` is also provided, the
        matching option is highlighted in green.
    answer : str, optional
        Ground-truth or correct answer string. Used both to highlight
        the matching option and to draw a single "Answer:" line below
        the options block.
    timeline : bool
        ``False`` (default) tiles thumbnails in a no-overlap grid;
        ``True`` renders them on a shared time axis. See above.
    thumb_width : int
        Width of each thumbnail rendered under a selected-frame dot.
    row_pad : int
        Vertical padding between rows.
    left_gutter : int
        Reserved width on the left for method labels.
    right_gutter : int
        Reserved width on the right of the time axis.
    axis_height : int
        Height of the bottom time-axis strip (tick labels), scaled by
        ``font_scale``. Ignored when ``timeline=False``.
    bg_color, text_color, dot_color : RGB tuples
        Defaults are white background / dark text / red dots.
    font_scale : float
        Multiplier for every font — method labels, time-axis ticks, the
        query, the options and the answer line — and for the line
        heights and label strips sized from them. Raise it (e.g. ``1.5``)
        when the strip is shrunk to print width; widen ``left_gutter``
        if long method names then overrun it. Default ``1.0``.
    canvas_width : int, optional
        Canvas width in pixels. ``None`` keeps the defaults: 1200 px in
        timeline mode, and in grid mode the width that fits the widest
        row (capped at 1800 px). A narrower canvas makes the same fonts
        larger once the image is scaled to a fixed print width.

    Returns
    -------
    PIL.Image.Image
    """
    if not results:
        raise ValueError("selection_strip: results must be non-empty")
    if font_scale <= 0:
        raise ValueError(f"selection_strip: font_scale must be > 0, got {font_scale}")
    if canvas_width is not None and canvas_width <= left_gutter + right_gutter:
        raise ValueError(
            "selection_strip: canvas_width must exceed left_gutter + "
            f"right_gutter ({left_gutter + right_gutter}), got {canvas_width}"
        )

    def _px(n: int) -> int:
        """Scale a font-derived size by ``font_scale`` (identity at 1.0)."""
        return int(round(n * font_scale))

    def _font(size: int):
        return _get_font(max(1, _px(size)))

    # Derive a shared time extent from the union of (candidate_indices,
    # indices) across every method. Fall back to max(indices) if no
    # candidate_indices were recorded.
    max_idx = 0
    native_fps = None
    for res in results.values():
        if res.candidate_indices is not None and len(res.candidate_indices):
            max_idx = max(max_idx, int(res.candidate_indices.max()))
        if len(res.indices):
            max_idx = max(max_idx, int(res.indices.max()))
        if native_fps is None:
            native_fps = res.metadata.get("native_fps") or None
    if max_idx <= 0:
        raise ValueError("selection_strip: no valid indices across methods")
    if native_fps is None or native_fps <= 0:
        native_fps = 1.0  # falls back to "index units" rather than seconds

    duration_s = max_idx / native_fps

    # Per-row layout: [label gutter][axis track with dots][thumbnail strip]
    # Thumbnails go *below* the dot, so row_height = axis_track + thumb_h + pad
    thumb_h = None
    thumbs_per_row: list[list[Image.Image]] = []
    dot_xs_per_row: list[list[int]] = []

    # First pass: prepare thumbnails per row (we need their height to
    # size the canvas).
    font_label = _font(13)
    font_tick = _font(10)

    # In grid mode, pick a canvas width that fits the widest row without
    # scaling the thumbnails. In timeline mode, a fixed 1200 px is the
    # baseline resolution — the time axis compresses longer videos.
    # An explicit ``canvas_width`` overrides both.
    if canvas_width is not None:
        canvas_w = int(canvas_width)
    elif timeline:
        canvas_w = 1200
    else:
        max_picks = max(len(r.indices) for r in results.values())
        # cell = thumb + a few pixels of inter-cell padding; cap at 1800px
        cell_pad = 6
        canvas_w = min(
            left_gutter + max_picks * (thumb_width + cell_pad) + right_gutter,
            1800,
        )
    track_x0 = left_gutter
    track_x1 = canvas_w - right_gutter
    track_w = track_x1 - track_x0

    for name, res in results.items():
        thumbs = [_frame_to_pil(f, thumb_width) for f in res.frames]
        if thumb_h is None and thumbs:
            thumb_h = thumbs[0].height
        thumbs_per_row.append(thumbs)

        xs = []
        if timeline:
            for idx in res.indices:
                x = int(track_x0 + (int(idx) / max_idx) * track_w)
                xs.append(x)
        else:
            # Evenly-spaced cells across the row, one per pick.
            n = len(res.indices)
            if n == 0:
                pass
            elif n == 1:
                xs.append(track_x0 + track_w // 2)
            else:
                step = track_w / n
                for i in range(n):
                    xs.append(int(track_x0 + (i + 0.5) * step))
        dot_xs_per_row.append(xs)

    if thumb_h is None:
        thumb_h = 64  # shouldn't happen — every Result has ≥1 frame

    # Height of the dots-on-a-line strip; it also holds the method label,
    # so it grows with the label font.
    axis_track = _px(24)
    row_height = axis_track + thumb_h + row_pad
    total_rows = len(results)
    axis_h = _px(axis_height)  # tick-label strip, sized for the tick font

    # Optional top header (query + options + answer). Each is one line
    # ~18px tall with padding. Uses wrap for long strings.
    font_header = _font(14)
    font_answer = _font(12)
    font_option = _font(12)
    correct_color = (46, 125, 50)      # darker green, readable on white
    option_color = (70, 70, 70)

    header_lines: list[tuple[str, Any, tuple]] = []
    if query:
        for line in _wrap_text(f"Q: {query}", max_chars=_chars_for_width(
                canvas_w - 2 * row_pad, font_header)):
            header_lines.append((line, font_header, text_color))

    # Render each option on its own line; highlight whichever matches
    # ``answer``. Matching is generous: exact string equality, or a
    # leading-letter match ("A" or "A." matches "A. Backflip.").
    correct_idx = _match_answer_to_option(answer, options) if options else -1
    if options:
        for i, opt in enumerate(options):
            color = correct_color if i == correct_idx else option_color
            prefix = "✓ " if i == correct_idx else "  "
            for line in _wrap_text(
                f"{prefix}{opt}",
                max_chars=_chars_for_width(canvas_w - 2 * row_pad, font_option),
            ):
                header_lines.append((line, font_option, color))

    if answer:
        for line in _wrap_text(
            f"Answer: {answer}",
            max_chars=_chars_for_width(canvas_w - 2 * row_pad, font_answer),
        ):
            header_lines.append((line, font_answer, correct_color))

    line_h = _px(20)
    header_h = row_pad + line_h * len(header_lines) + row_pad if header_lines else 0

    canvas_h = (
        header_h
        + total_rows * row_height
        + (axis_h + row_pad if timeline else row_pad)
    )

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Draw header (query + answer)
    if header_lines:
        y = row_pad
        for text, font, color in header_lines:
            draw.text((row_pad, y), text, fill=color, font=font)
            y += line_h
        # thin divider under the header
        draw.line(
            [(row_pad, header_h - 2), (canvas_w - row_pad, header_h - 2)],
            fill=(200, 200, 200),
            width=1,
        )

    # Choose a baseline color that reads on either bg
    baseline_color = (160, 160, 160)

    for row_i, (name, res) in enumerate(results.items()):
        y0 = header_h + row_i * row_height
        # Method label
        draw.text((8, y0 + axis_track // 2 - _px(6)), name, fill=text_color, font=font_label)
        baseline_y = y0 + axis_track // 2
        # Horizontal baseline — only in timeline mode, where it
        # reinforces the shared time axis. In grid mode it's
        # misleading (no temporal meaning) so we skip it.
        if timeline:
            draw.line(
                [(track_x0, baseline_y), (track_x1, baseline_y)],
                fill=baseline_color,
                width=1,
            )
        # Dots above each thumbnail (useful anchor in both modes).
        for x in dot_xs_per_row[row_i]:
            draw.ellipse(
                [(x - 5, baseline_y - 5), (x + 5, baseline_y + 5)],
                fill=dot_color,
                outline=text_color,
                width=1,
            )
        # Thumbnails below each dot
        thumb_y = y0 + axis_track
        for x, thumb in zip(dot_xs_per_row[row_i], thumbs_per_row[row_i]):
            tx = int(x - thumb.width / 2)
            tx = max(track_x0 - thumb.width // 2, min(track_x1 - thumb.width // 2, tx))
            # Keep the whole thumbnail on the canvas: a pick at either end
            # of the track would otherwise be cut by the canvas edge.
            tx = max(0, min(canvas_w - thumb.width, tx))
            canvas.paste(thumb, (tx, thumb_y))

    # Time axis along the bottom (timeline mode only).
    if timeline:
        axis_y = header_h + total_rows * row_height + row_pad
        draw.line(
            [(track_x0, axis_y), (track_x1, axis_y)],
            fill=(120, 120, 120),
            width=1,
        )
        # 6 tick marks
        n_ticks = 6
        for i in range(n_ticks):
            tx = track_x0 + i * track_w / (n_ticks - 1)
            draw.line(
                [(int(tx), axis_y), (int(tx), axis_y + 4)],
                fill=(120, 120, 120),
                width=1,
            )
            t_s = duration_s * i / (n_ticks - 1)
            m, s = divmod(int(t_s), 60)
            label = f"{m}:{s:02d}" if native_fps >= 1.0 else f"{int(t_s)}"
            # Keep the last labels inside the canvas at large font scales.
            label_x = min(int(tx) - _px(12),
                          canvas_w - int(draw.textlength(label, font=font_tick)) - 2)
            draw.text(
                (label_x, axis_y + _px(6)),
                label,
                fill=text_color,
                font=font_tick,
            )

    if save_path:
        canvas.save(save_path)

    return canvas


# ---------------------------------------------------------------------------
# Debug trajectories — method-contributed traces from metadata
# ---------------------------------------------------------------------------

def debug_trajectory(
    result: SelectionResult,
    *,
    traces: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, None),
) -> "matplotlib.figure.Figure":
    """Plot a selector's method-specific debug traces, stacked in time.

    Methods opt in by writing to
    ``SelectionResult.metadata["debug_traces"]``. Each entry is a
    dict with keys ``values`` (1D array), ``label`` (str),
    ``kind`` (``"curve"`` or ``"bar"``), and
    ``higher_is_better`` (bool). See the ``SelectionResult`` docstring
    for the full protocol.

    If the result carries no debug_traces, this falls back to
    :func:`score_timeline` on ``result.all_scores``.

    Parameters
    ----------
    result : SelectionResult
    traces : list[str], optional
        Which trace keys to plot (default: all). Ordering preserved.
    title : str, optional
    save_path : str, optional
    figsize : (width, height_or_None)
        If ``height`` is ``None``, we auto-size based on number of
        subplots.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dbg = result.metadata.get("debug_traces") or {}
    if not dbg:
        return score_timeline(result, title=title, save_path=save_path,
                              figsize=(figsize[0], figsize[1] or 3.5))

    plt = _ensure_matplotlib()

    # Select which traces to render
    keys = list(traces) if traces is not None else list(dbg.keys())
    keys = [k for k in keys if k in dbg]
    if not keys:
        return score_timeline(result, title=title, save_path=save_path,
                              figsize=(figsize[0], figsize[1] or 3.5))

    # Time axis
    known_fps = result.metadata.get("native_fps")
    fps = known_fps or 1.0
    candidate_fps = result.metadata.get("candidate_fps_used", 1.0) or 1.0
    # Candidate times: prefer candidate_indices over uniform spacing
    cand_idx = result.candidate_indices
    if cand_idx is not None and len(cand_idx):
        t_cand = np.asarray(cand_idx, dtype=np.float64) / fps
    else:
        n_cand = len(dbg[keys[0]]["values"])
        t_cand = np.arange(n_cand, dtype=np.float64) / candidate_fps

    t_selected = np.asarray(result.indices, dtype=np.float64) / fps

    # Layout: one subplot per trace, stacked
    n = len(keys)
    h = figsize[1] if figsize[1] is not None else max(2.4, 1.6 * n + 0.6)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        entry = dbg[key]
        vals = np.asarray(entry["values"], dtype=np.float64)
        label = entry.get("label", key)
        kind = entry.get("kind", "curve")
        higher_better = entry.get("higher_is_better", True)

        # Align length with t_cand; if mismatch, regenerate t from len(vals)
        t = t_cand if len(vals) == len(t_cand) else np.arange(len(vals)) / candidate_fps

        if kind == "bar":
            # Thin bars; useful for per-cluster/per-segment scores
            ax.bar(t, vals, width=(t[1] - t[0]) if len(t) > 1 else 1.0,
                   color="#90CAF9", alpha=0.85)
        else:
            ax.plot(t, vals, color="#90CAF9", linewidth=0.9)
            fill_alpha = 0.15 if higher_better else 0.10
            ax.fill_between(t, vals, alpha=fill_alpha, color="#90CAF9")

        # Selected frame markers at the y-value of the trace
        if len(t_selected):
            # Find the nearest candidate bin for each selected time
            if len(t) > 0:
                nearest = np.clip(
                    np.searchsorted(t, t_selected), 0, len(vals) - 1
                )
                ys = vals[nearest]
                ax.scatter(t_selected, ys, color="#E53935", s=45, zorder=5,
                           edgecolors="white", linewidths=0.8)
                for x in t_selected:
                    ax.axvline(x, color="#E53935", alpha=0.15, linewidth=0.8)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.2)

    # Without the video's frame rate the positions are frame indices.
    axes[-1].set_xlabel("Time (seconds)" if known_fps else "Frame index")
    if title:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
