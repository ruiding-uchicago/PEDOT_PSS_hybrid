"""Frozen-MatBERT text vectors for prediction, computed in an isolated torch-only process.

Default (``PEDOT_PSS_ENCODER=subprocess``): one persistent child interpreter per prediction
process runs ``_matbert_worker.py``. The child imports only the standard library, numpy,
torch and transformers, so PyTorch's OpenMP runtime is the only one in the process doing the
transformer forward pass, and the prediction process itself never imports torch. This removes
the macOS deadlock in which three different libomp copies (LightGBM/Homebrew, scikit-learn,
PyTorch) shared coalesced weak barrier symbols. The child uses a fixed CPU thread count
(``PEDOT_PSS_ENCODER_THREADS``, default 1) so results do not depend on the host.

Fallback (``PEDOT_PSS_ENCODER=inprocess``): the same code runs in this process with PyTorch
restricted to the same fixed thread count (default 1, i.e. no OpenMP parallel regions), for
hosts where starting a subprocess is not possible. Neither mode downloads anything; the
child is started with ``-I`` (isolated mode) and the parent's resolved ``sys.path``.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
import sys
import tempfile
import threading
import time
import warnings

import numpy as np

from ._assets import asset_directory

HIDDEN = 768
_HEADER = struct.Struct(">Q")
_WORKER_FILE = Path(__file__).with_name("_matbert_worker.py")
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
_BOOTSTRAP = (
    "import importlib.util, json, os, sys\n"
    "sys.path[:] = json.loads(os.environ.pop('PEDOT_PSS_WORKER_SYS_PATH'))\n"
    "spec = importlib.util.spec_from_file_location('pedot_pss_matbert_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module.serve()\n")
_LOCK = threading.RLock()
_WORKERS = {}
_IN_PROCESS = {}
_ERRORS = {"ValueError": ValueError, "FileNotFoundError": FileNotFoundError,
           "TypeError": TypeError, "KeyError": KeyError}


def settings():
    """Current encoder mode, thread count and timeout (environment controlled)."""
    mode = os.environ.get("PEDOT_PSS_ENCODER", "subprocess").strip().lower().replace("-", "")
    if mode not in {"subprocess", "inprocess"}:
        raise ValueError("PEDOT_PSS_ENCODER must be 'subprocess' (default) or 'inprocess'")
    threads = int(os.environ.get("PEDOT_PSS_ENCODER_THREADS", "1"))
    if threads < 1:
        raise ValueError("PEDOT_PSS_ENCODER_THREADS must be a positive integer")
    timeout = float(os.environ.get("PEDOT_PSS_ENCODER_TIMEOUT", "900"))
    return mode, threads, timeout


def _child_environment(threads):
    env = dict(os.environ)
    for name in _THREAD_VARIABLES:
        env[name] = str(threads)
    env.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1")
    paths = [os.fspath(entry) for entry in sys.path if isinstance(entry, (str, os.PathLike))]
    env["PEDOT_PSS_WORKER_SYS_PATH"] = json.dumps(paths)
    return env


class _Worker:
    """One persistent encoder child process; requests are serialised by the module lock."""

    def __init__(self, directory, threads, timeout):
        if not sys.executable:
            raise RuntimeError("No Python executable is known for the encoder worker; "
                               "set PEDOT_PSS_ENCODER=inprocess")
        self.directory, self.threads, self.timeout = str(directory), int(threads), float(timeout)
        self.owner = os.getpid()
        self.stderr = tempfile.TemporaryFile()
        command = [sys.executable, "-I", "-B", "-c", _BOOTSTRAP, str(_WORKER_FILE)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, env=_child_environment(self.threads),
                                        close_fds=True, bufsize=0)
        self.info = None
        try:
            self.info, _ = self._request({"op": "init", "asset_directory": self.directory,
                                          "threads": self.threads})
        except BaseException:
            self.close()                     # never leave an unregistered worker running
            raise

    # -- framing
    def _send(self, message):
        data = json.dumps(message).encode("utf-8")
        view = memoryview(_HEADER.pack(len(data)) + data)
        fd = self.process.stdin.fileno()
        try:
            while view:
                view = view[os.write(fd, view):]
        except OSError as error:
            raise self._failure(f"could not receive a request ({error})") from None

    def _read_exact(self, size, deadline):
        fd = self.process.stdout.fileno()
        chunks, remaining = [], size
        # selectors (kqueue/epoll/poll) rather than select(): no FD_SETSIZE limit in busy hosts.
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while remaining:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise self._failure(f"did not answer within {self.timeout:.0f} s", timeout=True)
                if not selector.select(min(left, 5.)):
                    if self.process.poll() is not None:
                        raise self._failure("exited unexpectedly")
                    continue
                chunk = os.read(fd, min(remaining, 1 << 20))
                if not chunk:
                    raise self._failure("closed its output unexpectedly")
                chunks.append(chunk)
                remaining -= len(chunk)
        return b"".join(chunks)

    def _read_frame(self, deadline):
        (size,) = _HEADER.unpack(self._read_exact(_HEADER.size, deadline))
        return self._read_exact(size, deadline)

    def _request(self, message):
        self._send(message)
        deadline = time.monotonic() + self.timeout
        header = json.loads(self._read_frame(deadline).decode("utf-8"))
        if not header.get("ok"):
            error = _ERRORS.get(header.get("type"), RuntimeError)(
                f"MatBERT encoder worker: {header.get('error')}")
            error.worker_reported = True     # protocol still consistent; worker stays usable
            raise error
        body = self._read_frame(deadline) if "nbytes" in header else b""
        return header, body

    def _failure(self, what, timeout=False):
        self.kill()
        tail = ""
        try:
            self.stderr.seek(0)
            tail = self.stderr.read()[-4000:].decode("utf-8", "replace")
        except (OSError, ValueError):
            pass
        kind = TimeoutError if timeout else RuntimeError
        return kind(f"The MatBERT encoder worker {what}." + (f"\nWorker stderr (tail):\n{tail}" if tail else ""))

    # -- public
    def encode(self, texts):
        header, body = self._request({"op": "encode", "texts": list(texts)})
        shape = tuple(header["shape"])
        if shape != (len(texts), HIDDEN) or header.get("dtype") != "<f4" or len(body) != header["nbytes"]:
            raise self._failure("returned an inconsistent array")
        return np.frombuffer(body, dtype="<f4").reshape(shape).astype(np.float32)

    def alive(self):
        return self.owner == os.getpid() and self.process.poll() is None

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        if self.owner != os.getpid():
            return
        if self.process.poll() is None:
            try:
                self._send({"op": "close"})
                self.process.wait(timeout=10)
            except Exception:
                pass
        self.kill()
        for stream in (self.process.stdin, self.process.stdout, self.stderr):
            try:
                stream.close()
            except Exception:
                pass


def _worker(directory, threads, timeout):
    key = (str(directory), int(threads))
    worker = _WORKERS.get(key)
    if worker is not None and not worker.alive():
        _WORKERS.pop(key, None)
        worker.close()
        worker = None
    if worker is None:
        worker = _Worker(directory, threads, timeout)
        _WORKERS[key] = worker
    worker.timeout = timeout
    return worker


def _encode_in_process(texts, directory, threads):
    from . import _matbert_worker as core
    if threads > 1 and sys.platform == "darwin":
        warnings.warn("PEDOT_PSS_ENCODER=inprocess with more than one PyTorch thread can deadlock on macOS "
                      "when LightGBM/scikit-learn OpenMP runtimes are loaded; use the default subprocess "
                      "encoder or PEDOT_PSS_ENCODER_THREADS=1", RuntimeWarning, stacklevel=3)
    import torch
    key = (str(directory), int(threads))
    previous = torch.get_num_threads()
    try:
        if key not in _IN_PROCESS:
            _IN_PROCESS[key] = core.load(directory, threads)
        tokenizer, encoder = _IN_PROCESS[key]
        return core.encode_unique(texts, tokenizer, encoder, threads)
    finally:
        torch.set_num_threads(previous)


def encode_texts(texts, configured_directory=None):
    """Float32 array (len(texts), 768) of frozen MatBERT vectors for unique strings."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, HIDDEN), dtype=np.float32)
    if len(set(texts)) != len(texts) or not all(isinstance(text, str) for text in texts):
        raise ValueError("Encoder input must be unique strings")
    mode, threads, timeout = settings()
    directory = asset_directory(configured_directory)
    with _LOCK:
        if mode == "inprocess":
            return _encode_in_process(texts, directory, threads)
        worker = _worker(directory, threads, timeout)
        try:
            return worker.encode(texts)
        except BaseException as error:
            if not getattr(error, "worker_reported", False):
                _WORKERS.pop((str(directory), int(threads)), None)
                worker.close()
            raise


def worker_info():
    """Diagnostics of live encoder workers owned by this process (pid, versions, OpenMP images)."""
    with _LOCK:
        return [dict(worker.info or {}, asset_directory=key[0], threads=key[1])
                for key, worker in _WORKERS.items() if worker.alive()]


def shutdown():
    """Stop all encoder workers started by this process (also registered with atexit)."""
    with _LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.close()


def _after_fork_in_child():
    # A forked child must not talk to its parent's worker; drop its inherited pipe copies.
    global _LOCK
    _LOCK = threading.RLock()
    for worker in list(_WORKERS.values()):
        for stream in (worker.process.stdin, worker.process.stdout):
            try:
                stream.close()
            except Exception:
                pass
    _WORKERS.clear()


atexit.register(shutdown)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
