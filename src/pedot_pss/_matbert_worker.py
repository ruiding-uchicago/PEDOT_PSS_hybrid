"""Frozen cased MatBERT text encoder, runnable as an isolated torch-only worker process.

This module deliberately imports only the standard library at top level. ``load`` and
``encode_unique`` need numpy, torch and transformers; nothing here imports scikit-learn,
LightGBM or CatBoost. The package runs this file in a separate interpreter (see
``pedot_pss._encoder``) so that exactly one OpenMP runtime (PyTorch's) is present in the
process that executes the transformer forward pass.

Why: on macOS the LightGBM wheel uses Homebrew's libomp, scikit-learn bundles another and
PyTorch bundles a third. libomp exports C++ template instantiations such as
``__kmp_suspend_64<false,true>`` / ``__kmp_resume_64`` as weak symbols, and dyld coalesces
weak definitions to the first-loaded image. PyTorch's OpenMP barrier therefore slept and
woke threads through another runtime's thread table, which deadlocked the softmax parallel
region. One runtime per process removes the mechanism instead of relying on timing.

The numerical procedure is the original one: 12 projected fields, 128-token right
truncation, unique strings ordered by (min(token count, 126), text), batches of four,
nonspecial-token mean pooling in float64, stored as float32. The worker additionally fixes
the CPU thread count (default 1) so results do not depend on the host's core count.
"""
from __future__ import annotations

import json
import os
import struct
import sys

HIDDEN = 768
_HEADER = struct.Struct(">Q")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _disable_optional_transformers_backends():
    # BERT text encoding needs none of these; importing scikit-learn here would load a second
    # OpenMP runtime next to PyTorch's, and the vision/audio stacks are irrelevant.
    from transformers.utils import import_utils
    for flag in ("_sklearn_available", "_torchvision_available", "_librosa_available"):
        if hasattr(import_utils, flag):
            setattr(import_utils, flag, False)


def load(directory, threads=1):
    """Load the original author tokenizer and encoder from local files only."""
    from pathlib import Path
    import torch
    torch.set_num_threads(int(threads))
    _disable_optional_transformers_backends()
    from transformers import BertForMaskedLM, BertTokenizer
    directory = Path(directory)
    if not (directory / "config.json").is_file() or not (directory / "vocab.txt").is_file():
        raise FileNotFoundError(f"MatBERT assets are missing in {directory}; run python -m pedot_pss setup-encoder")
    model = BertForMaskedLM.from_pretrained(str(directory), weights_only=True,
        trust_remote_code=False, local_files_only=True, use_safetensors=False,
        torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    tokenizer = BertTokenizer(vocab_file=str(directory / "vocab.txt"),
        do_lower_case=False, padding_side="right", truncation_side="right")
    if sorted(tokenizer.all_special_ids) != [0, 1, 2, 3, 4]:
        raise ValueError("Original MatBERT cased tokenizer vocabulary required")
    return tokenizer, model.bert


def encode_unique(texts, tokenizer, encoder, threads=1):
    """Return a float32 array (len(texts), 768) aligned with ``texts`` (unique strings)."""
    import numpy as np
    import torch
    texts = list(texts)
    if len(set(texts)) != len(texts):
        raise ValueError("encode_unique expects unique strings")
    # Re-assert the thread count in the calling thread (OpenMP ICVs are per thread).
    torch.set_num_threads(int(threads))
    ordered = sorted(texts, key=lambda value: (
        min(len(tokenizer(value, add_special_tokens=False)["input_ids"]), 126), value))
    vectors = {}
    with torch.no_grad():
        for start in range(0, len(ordered), 4):
            values = ordered[start:start + 4]
            inputs = tokenizer(values, padding=True, truncation=True, max_length=128,
                return_tensors="pt", return_special_tokens_mask=True)
            special = inputs.pop("special_tokens_mask").numpy().astype(bool)
            special |= np.isin(inputs["input_ids"].numpy(), tokenizer.all_special_ids)
            keep = inputs["attention_mask"].numpy().astype(bool) & ~special
            count = keep.sum(axis=1)
            if np.any(count == 0):
                raise ValueError("Experimental text contains no usable MatBERT content")
            hidden = encoder(**inputs, return_dict=True).last_hidden_state.numpy()
            pooled = np.einsum("bsh,bs->bh", hidden, keep,
                dtype=np.float64, optimize=False) / count[:, None]
            for value, vector in zip(values, pooled.astype(np.float32)):
                vectors[value] = vector
    result = np.stack([vectors[value] for value in texts]) if texts else np.zeros((0, HIDDEN), np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite MatBERT field vectors")
    return result


def loaded_openmp_images():
    """OpenMP runtime images mapped into this process (diagnostics only)."""
    names = []
    try:
        if sys.platform == "darwin":
            import ctypes
            libc = ctypes.CDLL(None)
            libc._dyld_image_count.restype = ctypes.c_uint32
            libc._dyld_get_image_name.restype = ctypes.c_char_p
            libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            names = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())
                     if libc._dyld_get_image_name(i)]
        elif os.path.exists("/proc/self/maps"):
            with open("/proc/self/maps") as handle:
                names = [line.split()[-1] for line in handle if len(line.split()) >= 6]
    except Exception:  # diagnostics must never break encoding
        return []
    return sorted({name for name in names
                   if any(key in os.path.basename(name) for key in ("libomp", "libgomp", "libiomp"))})


# ---------------------------------------------------------------- worker process protocol
def _read_frame(stream):
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        return None
    (size,) = _HEADER.unpack(header)
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("Truncated request from the parent process")
    return payload


def _write_frame(stream, payload):
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)


def serve():
    """Answer framed JSON requests on the original stdin/stdout until EOF or 'close'."""
    # Keep private copies of the protocol pipes, then point fd 0/1 elsewhere so that any
    # library output (Python or C level) cannot corrupt the binary protocol.
    protocol_in = os.fdopen(os.dup(0), "rb")    # buffered: read(n) returns n bytes or EOF
    protocol_out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    state = {}

    def reply(header, body=b""):
        _write_frame(protocol_out, json.dumps(header).encode("utf-8"))
        if header.get("ok") and "nbytes" in header:
            _write_frame(protocol_out, body)
        protocol_out.flush()

    while True:
        payload = _read_frame(protocol_in)
        if payload is None:
            return
        try:
            request = json.loads(payload.decode("utf-8"))
            operation = request.get("op")
            if operation == "close":
                reply({"ok": True})
                return
            if operation == "init":
                threads = int(request.get("threads", 1))
                state["tokenizer"], state["encoder"] = load(request["asset_directory"], threads)
                state["threads"] = threads
                import torch
                import transformers
                reply({"ok": True, "pid": os.getpid(), "python": sys.version.split()[0],
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "torch_threads": torch.get_num_threads(),
                       "openmp_images": loaded_openmp_images(),
                       "heavy_modules_loaded": sorted(name for name in ("sklearn", "lightgbm", "catboost")
                                                      if name in sys.modules)})
                continue
            if operation == "encode":
                if "encoder" not in state:
                    raise RuntimeError("Encoder worker used before initialisation")
                result = encode_unique(request["texts"], state["tokenizer"], state["encoder"], state["threads"])
                body = result.astype("<f4", copy=False).tobytes()
                reply({"ok": True, "shape": list(result.shape), "dtype": "<f4", "nbytes": len(body)}, body)
                continue
            raise ValueError(f"Unknown encoder worker operation {operation!r}")
        except Exception as error:  # report to the parent; the parent raises
            reply({"ok": False, "type": type(error).__name__, "error": str(error)})


if __name__ == "__main__":
    serve()
