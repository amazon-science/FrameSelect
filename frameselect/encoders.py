# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Vision-language encoder wrapper (CLIP via open_clip)."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Thin wrapper around open_clip for frame and text encoding.

    Lazily loads the model on first use to avoid import-time GPU allocation.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import open_clip

        # The original OpenAI CLIP weights were trained with QuickGELU, but
        # open_clip's plain ``ViT-B-32``/``ViT-L-14`` configs default to
        # ``nn.GELU`` — loading the ``openai`` tag into them silently runs
        # the wrong activation (image embeddings diverge by cos≈0.91). Force
        # QuickGELU for the ``openai`` tag unless the model name already
        # encodes it. Other pretrained tags (LAION/webli/…) correctly report
        # their own activation, so leave them untouched.
        kwargs = {}
        if self.pretrained == "openai" and "quickgelu" not in self.model_name.lower():
            kwargs["force_quick_gelu"] = True

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device,
            **kwargs,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Encode frames to normalized embeddings.

        Parameters
        ----------
        frames : np.ndarray, shape (N, H, W, 3), uint8
        batch_size : int

        Returns
        -------
        embeddings : np.ndarray, shape (N, D), float32, L2-normalized
        """
        self._load()
        all_embs = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            imgs = torch.stack(
                [self._preprocess(Image.fromarray(f)) for f in batch]
            ).to(self.device)
            emb = self._model.encode_image(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized embedding.

        Returns
        -------
        embedding : np.ndarray, shape (D,), float32, L2-normalized
        """
        self._load()
        tokens = self._tokenizer([text]).to(self.device)
        emb = self._model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode multiple text strings to normalized embeddings.

        Returns
        -------
        embeddings : np.ndarray, shape (N, D), float32, L2-normalized
        """
        self._load()
        tokens = self._tokenizer(texts).to(self.device)
        emb = self._model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def similarity(self, frames: np.ndarray, text: str) -> np.ndarray:
        """Compute cosine similarity between each frame and a text query.

        Returns
        -------
        scores : np.ndarray, shape (N,), float32
        """
        frame_embs = self.encode_frames(frames)
        text_emb = self.encode_text(text)
        return (frame_embs @ text_emb).astype(np.float32)

    def release(self) -> None:
        """Free GPU memory held by this encoder.

        Sends the model to CPU, drops all references, and empties the
        CUDA cache. After this call the encoder is usable again — the
        next ``_load()`` call will re-allocate on ``self.device``.
        """
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self):
        # Best-effort release on garbage collection so stray references
        # in worker dict closures don't leak full encoder state.
        try:
            self.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Multilingual text tower on a FROZEN OpenAI CLIP image tower (M-CLIP).
# ---------------------------------------------------------------------------

# M-CLIP text-encoder ids paired with the OpenAI CLIP image tower they were
# aligned to (and the shared embedding dim). Only these two align to a frozen
# OpenAI tower; ``…-Vit-B-16Plus`` aligns to LAION ``ViT-B-16-plus-240`` (NOT
# OpenAI) so it is deliberately excluded from this OpenAI-frozen mapping.
_MCLIP_PAIRING: dict[str, tuple[str, str, int]] = {
    # text id                              -> (image model, image pretrained, dim)
    "M-CLIP/XLM-Roberta-Large-Vit-B-32": ("ViT-B-32", "openai", 512),
    "M-CLIP/XLM-Roberta-Large-Vit-L-14": ("ViT-L-14", "openai", 768),
}


def _ensure_mclip_safetensors(local_dir: str) -> None:
    """Convert M-CLIP's ``pytorch_model.bin`` to ``model.safetensors`` in place.

    M-CLIP ships only a pickle ``pytorch_model.bin``; transformers >= 4.53
    refuses ``torch.load`` on torch < 2.6 (CVE-2025-32434), which blocks
    ``from_pretrained``. The checkpoint is an official, self-downloaded XLM-R
    state dict (plain tensors), so a one-time ``weights_only=True`` load +
    re-save as safetensors is safe and
    lets ``from_pretrained`` use the safetensors path (guard not hit). The
    ``.bin`` is renamed aside so transformers prefers the safetensors file.
    """
    from safetensors.torch import save_file

    bin_path = os.path.join(local_dir, "pytorch_model.bin")
    st_path = os.path.join(local_dir, "model.safetensors")
    if os.path.exists(st_path) or not os.path.exists(bin_path):
        return
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    state_dict = {
        k: v.contiguous() for k, v in state_dict.items() if hasattr(v, "contiguous")
    }
    save_file(state_dict, st_path)
    os.rename(bin_path, bin_path + ".disabled")
    logger.info("M-CLIP: converted %s -> model.safetensors", bin_path)


class MultilingualCLIPEncoder:
    """Multilingual text tower (M-CLIP) on a FROZEN OpenAI CLIP image tower.

    Isolates the *text-encoder* variable in a multilingual frame-selection
    experiment: the image tower is an ordinary :class:`CLIPEncoder`
    (``ViT-B-32``/``openai`` or ``ViT-L-14``/``openai``) so frame embeddings
    are byte-identical to the English-CLIP baseline (and share its embedding
    cache), while text is encoded by an M-CLIP XLM-R multilingual text encoder
    aligned to that same frozen OpenAI image space. Swapping English-CLIP →
    this encoder therefore changes ONLY the query→embedding map.

    Interface-compatible with :class:`CLIPEncoder` (``encode_frames``,
    ``encode_text``, ``encode_texts``, ``similarity``, ``release``). Exposes
    ``model_name``/``pretrained`` = the IMAGE tower, so the frame-embedding
    cache key (which keys on the image encoder) is shared with the baseline.
    """

    def __init__(
        self,
        text_model: str = "M-CLIP/XLM-Roberta-Large-Vit-B-32",
        image_model: str = "ViT-B-32",
        image_pretrained: str = "openai",
        device: str = "cpu",
    ):
        # ``model_name``/``pretrained`` intentionally mirror the IMAGE tower so
        # cache.make_key / _cache_config key the frame cache on the image
        # encoder (shared with the English-CLIP baseline). The text tower
        # identity is tracked separately for the query-embedding cache.
        self.model_name = image_model
        self.pretrained = image_pretrained
        self.text_model = text_model
        self.device = device
        self._image = CLIPEncoder(image_model, image_pretrained, device)
        self._text_model = None
        self._tokenizer = None
        self._checked_dim = False

    def _load(self):
        if self._text_model is not None:
            return
        from huggingface_hub import snapshot_download
        from multilingual_clip import pt_multilingual_clip
        import transformers

        local_dir = snapshot_download(self.text_model)
        _ensure_mclip_safetensors(local_dir)
        self._text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
            local_dir
        ).to(self.device).eval()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(local_dir)

    def _assert_dim(self, text_dim: int) -> None:
        """Fail loudly if the text tower's dim != the image tower's dim.

        A mismatch means the M-CLIP variant is not aligned to the configured
        OpenAI image tower (e.g. pairing B-16Plus with an OpenAI tower), which
        would make cosine similarities meaningless.
        """
        if self._checked_dim:
            return
        expected = _MCLIP_PAIRING.get(self.text_model)
        img_dim = self._image.encode_text("dim probe").shape[0]
        if img_dim != text_dim:
            raise ValueError(
                f"M-CLIP text dim {text_dim} != image tower dim {img_dim} for "
                f"text={self.text_model!r} image={self.model_name!r}/"
                f"{self.pretrained!r}. The M-CLIP variant must align to this "
                f"image tower (expected pairing: {expected})."
            )
        self._checked_dim = True

    def encode_frames(self, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Encode frames via the frozen OpenAI image tower (delegated)."""
        return self._image.encode_frames(frames, batch_size=batch_size)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode text via the M-CLIP multilingual text tower.

        Returns L2-normalized embeddings, shape ``(N, D)``, float32 — matching
        :meth:`CLIPEncoder.encode_texts` (M-CLIP output is not unit-norm, so
        we normalize here).

        Replicates ``MultilingualCLIP.forward`` (tokenize → XLM-R transformer →
        attention-masked mean-pool → linear projection) but moves the tokenized
        tensors to the model's device — M-CLIP's own ``forward`` tokenizes on
        CPU and breaks when the model is on CUDA.
        """
        self._load()
        tok = self._tokenizer(list(texts), padding=True, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        embs = self._text_model.transformer(**tok)[0]
        att = tok["attention_mask"]
        pooled = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        emb = self._text_model.LinearTransformation(pooled)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out = emb.cpu().numpy().astype(np.float32)
        self._assert_dim(out.shape[1])
        return out

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string. Shape ``(D,)``, float32, L2-normalized."""
        return self.encode_texts([text])[0]

    def similarity(self, frames: np.ndarray, text: str) -> np.ndarray:
        """Cosine similarity between each frame (image tower) and a text query
        (M-CLIP text tower). Shape ``(N,)``, float32."""
        frame_embs = self.encode_frames(frames)
        text_emb = self.encode_text(text)
        return (frame_embs @ text_emb).astype(np.float32)

    def release(self) -> None:
        """Free GPU memory held by both towers (see :meth:`CLIPEncoder.release`)."""
        self._image.release()
        if self._text_model is not None:
            try:
                self._text_model.to("cpu")
            except Exception:
                pass
        self._text_model = None
        self._tokenizer = None
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
