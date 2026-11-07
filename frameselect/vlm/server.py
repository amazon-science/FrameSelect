# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""VLM server lifecycle management.

Provides a thin wrapper around ``vllm serve`` so scripts can start,
wait for, and stop a local vLLM server without writing subprocess
boilerplate. Designed as a context manager so the server is always
cleaned up even if the script crashes::

    from frameselect.vlm.server import VLMServer

    with VLMServer.start("Qwen/Qwen2.5-VL-7B-Instruct", port=8000) as server:
        result = ev.run(fs.create("efs"))
    # server stopped automatically

Or manually::

    server = VLMServer.start("Qwen/Qwen2.5-VL-7B-Instruct", port=8000)
    try:
        result = ev.run(fs.create("efs"))
    finally:
        server.stop()
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


def _visible_gpu_count() -> int:
    """Return the number of GPUs visible to this process.

    Honours ``CUDA_VISIBLE_DEVICES`` when set (comma-separated list of
    indices, possibly empty to force CPU). Falls back to ``nvidia-smi``
    otherwise. Returns ``0`` when no GPUs are available.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        stripped = visible.strip()
        if stripped == "":
            return 0
        return len([g for g in stripped.split(",") if g.strip()])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return len([line for line in out.strip().splitlines() if line.strip()])
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0


def _trust_remote_code() -> bool:
    """Whether to pass ``--trust-remote-code`` to ``vllm serve``.

    Opt-in via ``FRAMESELECT_TRUST_REMOTE_CODE`` (``1``/``true``/``yes``/
    ``on``). Defaults to ``False`` because the flag lets vLLM run arbitrary
    Python shipped in the model repo — only enable it for trusted model
    ids (InternVL / MiniCPM-V and similar need it).
    """
    val = os.environ.get("FRAMESELECT_TRUST_REMOTE_CODE", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _env_for_vllm() -> dict:
    """Return a subprocess env to hand to ``vllm serve``.

    Rule:

    * If the caller already set ``CUDA_VISIBLE_DEVICES`` (anything --
      including an explicit list like ``"1,3,5"`` or empty string to
      force CPU), pass it through unchanged. The caller's choice wins.
    * Otherwise inherit the current process env as-is and let vLLM see
      every GPU on the machine. That matches the "run end-to-end
      without setup" pattern: if the user didn't ask for a specific
      subset, use all of them.
    """
    return os.environ.copy()


class VLMServer:
    """Manages a local vLLM server process.

    Parameters
    ----------
    model : str
        HuggingFace model ID to serve.
    port : int
        Port to listen on.
    proc : subprocess.Popen
        The running server process.
    """

    def __init__(self, model: str, port: int, proc: subprocess.Popen):
        self.model = model
        self.port = port
        self._proc = proc

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        model: str,
        port: int = 8000,
        *,
        tensor_parallel_size: Optional[int] = None,
        data_parallel_size: int = 1,
        max_model_len: int = 16384,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float = 0.80,
        dtype: str = "auto",
        extra_args: Optional[list[str]] = None,
        startup_timeout: int = 300,
        verbose: bool = True,
    ) -> "VLMServer":
        """Start a vLLM server and wait until it is ready.

        Parameters
        ----------
        model : str
            HuggingFace model ID (e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``).
        port : int
            Port to listen on. Default 8000.
        tensor_parallel_size : int, optional
            Number of GPUs per replica (tensor parallelism). ``None``
            (the default) picks the number of currently visible GPUs:
            it honours ``CUDA_VISIBLE_DEVICES`` if the caller has
            pre-selected a subset (``CUDA_VISIBLE_DEVICES=1,3,5`` →
            TP=3), otherwise uses every GPU ``nvidia-smi`` reports.
        data_parallel_size : int
            Number of replicas (data parallelism). Default 1.
        max_model_len : int
            Maximum context length in tokens. Default 16384. With
            ``image_size=512`` (the evaluator's default) a 16-frame
            Qwen2.5-VL MCQ prompt lands at ~11-13 k tokens including
            question + options, so 16 k covers the library's common
            ``n_frames=8..16`` configurations with headroom. If you
            pass ``image_size=None`` or ``image_size>=768`` to the
            evaluator, raise this accordingly (native-res 1080p
            frames at 16x need ~30 k). Lower only when your model
            doesn't advertise that much rope-scaled context.
        max_num_batched_tokens : int
            Per-step prefill budget. Default 16384. vLLM's stock
            default (2048) serialises VLM prompts because one
            16-frame MCQ prompt is ~12 k tokens and can't share a
            prefill batch with anything else. 16384 lets a full
            prompt land in one prefill step which brings throughput
            to 4-5 req/s on Qwen2.5-VL-7B. Must be ≥ the largest
            single-prompt size you care about to stay in the fast
            path; dropping to 8192 cuts throughput ~3× when prompts
            exceed 8 k.
        max_num_seqs : int
            Maximum concurrent sequences. Default 32. Matches the
            thread-pool fan-out used by the OpenAI-path evaluator;
            raising beyond ~32 costs KV cache and rarely helps
            throughput on the VLM prompts the library emits. On
            24 GB-class cards with a 7-8 B VLM under
            TP=4 this is the safe ceiling — the ``masked_scatter``
            activation in VLM vision embedding scales with this knob
            times vision tokens, and going to 64 with ``image_size=
            None`` on long, high-resolution videos triggers OOM on the
            worst-case prompts.
        gpu_memory_utilization : float
            Fraction of GPU memory to use. Default 0.80. Slightly
            lower than vLLM's 0.90 to leave headroom for VLM vision
            embeddings, which allocate large transient tensors
            outside the KV cache budget. Raise on cards with more
            than 24 GB.
        dtype : str
            Model dtype. Default ``"auto"``.
        extra_args : list[str], optional
            Additional arguments forwarded to ``vllm serve``.
        startup_timeout : int
            Seconds to wait for the server to become ready. Default 300.
        verbose : bool
            Print startup messages. Default True.

        Returns
        -------
        VLMServer
            A running server instance. Call :meth:`stop` when done, or
            use as a context manager.

        Raises
        ------
        RuntimeError
            If no GPUs are visible, or the server does not become ready
            within ``startup_timeout``.
        """
        # Resolve tensor_parallel_size from visible GPUs when the
        # caller didn't pin one. The tricky constraint: vLLM requires
        # ``n_heads % tensor_parallel_size == 0``. Most modern VLMs
        # have head counts of 28 (Qwen2.5-VL-7B), 32 (Llama-family),
        # 40, 64, etc. — TP=8 breaks on a 28-head model. We therefore
        # default TP to the largest **power of two** ≤ visible GPU
        # count (1, 2, 4, 8, 16, ...). That covers every head count
        # we care about (28 is divisible by 1/2/4, 32/40/64 are by
        # 1/2/4/8), at the cost of sometimes under-using GPUs. The
        # caller can always pass ``tensor_parallel_size=8`` explicitly
        # to force an 8-way split when they know the model allows it.
        #
        # When extra GPUs remain (e.g. 8 visible but TP=4), they're
        # left idle by default — see the inline comment below for why
        # we stopped auto-populating DP.
        n_visible = _visible_gpu_count()
        if tensor_parallel_size is None:
            if n_visible == 0:
                raise RuntimeError(
                    "VLMServer.start(): no GPUs visible to the process. "
                    "Set CUDA_VISIBLE_DEVICES=<comma-list> or make sure "
                    "nvidia-smi reports at least one GPU."
                )
            # Largest power of two ≤ n_visible, capped at 4. Why 4:
            # Qwen2.5-VL-7B has 28 heads (divisible by 4 but not 8);
            # LLaVA-OV has 32 heads (divisible by both). Capping at 4
            # is the safe default that works for every VLM the library
            # targets. Callers who know their model supports TP=8
            # (Llama/Qwen3 64-head variants, 70B dense) pass
            # ``tensor_parallel_size=8`` explicitly.
            #
            # On a machine with more than 4 GPUs this leaves some idle.
            # That's a deliberate tradeoff: a single TP=4 engine consistently
            # hits 4-5 req/s on Qwen2.5-VL-7B at 16 frames, which is
            # the throughput ceiling the downstream base64 pipeline
            # can feed anyway. Doubling server capacity via
            # ``data_parallel_size=2`` costs 4 more GPUs for maybe
            # 30% more throughput in practice (the evaluator's HTTP
            # fan-out is the new bottleneck) and currently exposes a
            # vLLM DP-coordinator hang.
            pot = 1
            while pot * 2 <= min(n_visible, 4):
                pot *= 2
            tensor_parallel_size = pot
            # Leave data_parallel_size at the caller's value (default 1).
            # We used to auto-fill leftover GPUs with DP replicas, but
            # vLLM's DP coordinator is flaky on the v0.17 line — the
            # second engine can hang in "Waiting for READY message
            # from DP Coordinator..." and sit at 0% GPU utilisation
            # while the first takes the whole workload. Single-engine
            # TP=4 is stable and gets 4-5 req/s on Qwen2.5-VL-7B at
            # 16 frames, which matches the library's profiled
            # throughput. Users who know their setup tolerates DP can
            # still pass ``data_parallel_size=2`` explicitly.
        if tensor_parallel_size * data_parallel_size > max(1, n_visible):
            raise RuntimeError(
                f"VLMServer.start(): TP={tensor_parallel_size} × "
                f"DP={data_parallel_size} > visible GPUs ({n_visible}). "
                "Either lower the parallelism knobs or widen "
                "CUDA_VISIBLE_DEVICES."
            )

        # Resolve the vLLM entrypoint against *this* interpreter rather than
        # PATH. A bare "vllm" picks up whichever console script comes first on
        # PATH, which may belong to an unrelated interpreter with an
        # incompatible environment. Prefer a console
        # script sitting next to sys.executable, else run the CLI module with
        # the current interpreter.
        _vllm_script = Path(sys.executable).parent / "vllm"
        _launcher = (
            [str(_vllm_script)] if _vllm_script.exists()
            else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
        )
        cmd = [
            *_launcher, "serve", model,
            # Loopback only: the server has no API key, and the library
            # always talks to it on localhost.
            "--host", "127.0.0.1",
            "--port", str(port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-model-len", str(max_model_len),
            "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--max-num-seqs", str(max_num_seqs),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--dtype", dtype,
        ]
        # ``--trust-remote-code`` lets vLLM execute arbitrary Python from
        # the model repo (RCE if the model id is ever untrusted). It is
        # therefore opt-in: only passed when FRAMESELECT_TRUST_REMOTE_CODE
        # is set (needed for InternVL / MiniCPM etc. that ship custom
        # modules). Only enable it for model ids you trust.
        # --data-parallel-size only exists on newer vLLM (not 0.7.3), and
        # DP=1 is the default, so only pass it when it actually does something.
        if data_parallel_size and data_parallel_size > 1:
            cmd.extend(["--data-parallel-size", str(data_parallel_size)])
        if _trust_remote_code():
            cmd.append("--trust-remote-code")
        if extra_args:
            cmd.extend(extra_args)

        if verbose:
            print(f"Starting vLLM server: {model} on port {port} "
                  f"(TP={tensor_parallel_size}, DP={data_parallel_size})...")

        # Write stdout/stderr to a log file instead of /dev/null so we
        # can surface failures. Path is predictable for debugging.
        import tempfile
        log_path = Path(tempfile.gettempdir()) / f"vllm_server_{port}.log"
        log_fh = open(log_path, "w")
        if verbose:
            print(f"  Server logs → {log_path}")

        env = _env_for_vllm()
        cvd = env.get("CUDA_VISIBLE_DEVICES")
        if verbose:
            if cvd:
                print(f"  CUDA_VISIBLE_DEVICES={cvd} (from env)")
            else:
                print(f"  CUDA_VISIBLE_DEVICES unset → vLLM sees all "
                      f"{n_visible} GPUs")

        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            # Put the vLLM server into its own process group so we can
            # signal the whole tree (parent + EngineCore workers) with
            # a single ``killpg`` call in ``stop()``. Without this, the
            # engine workers linger after the parent dies and continue
            # holding GPU memory.
            start_new_session=True,
        )

        if verbose:
            print(f"  Waiting for server to be ready (up to {startup_timeout}s)...")

        if not _wait_for_server("localhost", port, timeout=startup_timeout, proc=proc):
            exit_code = proc.poll()  # before terminate(), which sets one
            proc.terminate()
            log_fh.close()
            # Surface the last chunk of the server log to help debug
            try:
                with open(log_path) as f:
                    tail = f.read()[-2000:]
            except OSError:
                tail = "(log unavailable)"
            reason = (
                f"exited with code {exit_code} before becoming ready"
                if exit_code is not None
                else f"did not become ready within {startup_timeout}s"
            )
            raise RuntimeError(
                f"vLLM server for {model} on port {port} {reason}.\n\n"
                f"--- Last 2KB of server log ({log_path}) ---\n{tail}\n"
                f"--- end log ---\n"
            )

        if verbose:
            print(f"  ✓ Server ready at http://localhost:{port}/v1")

        return cls(model=model, port=port, proc=proc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self, verbose: bool = True) -> None:
        """Terminate the server process tree.

        ``vllm serve`` spawns child processes (the EngineCore workers)
        that hold the GPU allocation. Simply calling ``proc.terminate()``
        on the parent is not sufficient — the children survive and
        continue to hold GPU memory. We therefore signal the whole
        process group so both parent and engine workers get SIGTERM
        (escalating to SIGKILL on timeout).
        """
        if self._proc is None:
            return
        if verbose:
            print(f"Stopping vLLM server ({self.model} on port {self.port})...")

        import os
        import signal

        pid = self._proc.pid
        # Prefer killing the whole process group so EngineCore children
        # die with the parent. ``start_new_session=True`` on Popen makes
        # the parent a session leader whose pgid == pid.
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # Fallback: terminate just the parent.
            try:
                self._proc.terminate()
            except Exception:
                pass

        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Escalate — SIGKILL the whole group.
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    self._proc.kill()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        self._proc = None
        if verbose:
            print("  ✓ Server stopped.")

        # Wait briefly for the OS / CUDA runtime to actually reclaim
        # the GPU memory the engine workers were holding. Polling
        # nvidia-smi is cheap and avoids the next method racing into
        # an allocation while the kernel is still tearing down the
        # vLLM process tree.
        self._wait_for_gpu_free(verbose=verbose)

    @staticmethod
    def _wait_for_gpu_free(timeout: float = 15.0, verbose: bool = False) -> None:
        """Block until ``nvidia-smi`` reports no compute apps, up to ``timeout``.

        Called after ``stop()``. Short of a hard OOM, the kernel frees
        the vLLM tree's GPU memory within 1-3 s after SIGKILL; we poll
        at 0.5 s intervals and give up after ``timeout``.
        """
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid",
                     "--format=csv,noheader"],
                    text=True, timeout=5,
                )
                if not out.strip():
                    if verbose:
                        print("  ✓ GPU memory reclaimed.")
                    return
            except (subprocess.SubprocessError, FileNotFoundError):
                return  # no nvidia-smi — nothing to check
            time.sleep(0.5)
        if verbose:
            print("  (GPU still shows active compute apps after "
                  f"{timeout}s — continuing anyway)")

    def is_running(self) -> bool:
        """Return True if the server process is still alive."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VLMServer":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return f"VLMServer(model={self.model!r}, port={self.port}, status={status})"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _wait_for_server(host: str, port: int, timeout: int = 300, proc=None) -> bool:
    """Poll ``http://<host>:<port>/v1/models`` until it answers 200.

    Gives up early when ``proc`` (the server process) has already exited,
    e.g. because vLLM is not installed or crashed on startup.
    """
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False
