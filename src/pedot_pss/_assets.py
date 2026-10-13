"""Explicit setup of the frozen author MatBERT assets; prediction stays offline."""
from pathlib import Path
import json
import os
import urllib.request

AUTHOR_BASE = "https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/"
AUTHOR_ASSETS = {"config.json": 507, "vocab.txt": 217022,
                 "pytorch_model.bin": 438147282}


def asset_directory(configured=None):
    """Resolve a relative package asset path or an explicit user override."""
    override = os.environ.get("PEDOT_PSS_MATBERT_ASSETS")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parent
    if configured is None:
        return root / "assets" / "matbert"
    directory = Path(configured).expanduser()
    return directory.resolve() if directory.is_absolute() else root / directory


def setup_encoder(destination=None):
    """Download the three original author files once, only on explicit request.

    Uses HTTPS plus the original declared file lengths; no hashing, model
    execution, corpus download or input transmission is performed.
    The checkpoint is excluded from Git and fetched directly from its author.
    """
    directory = asset_directory(destination)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected_bytes in AUTHOR_ASSETS.items():
        path = directory / name
        if path.is_file() and path.stat().st_size == expected_bytes:
            continue
        temporary = directory / (name + ".part")
        try:
            request = urllib.request.Request(AUTHOR_BASE + name,
                                             headers={"User-Agent": "PEDOT-PSS-surrogate-asset-setup/1"})
            received = 0
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if received > expected_bytes:
                        raise ValueError(f"Unexpected author asset size: {name}")
            if received != expected_bytes:
                raise ValueError(f"Incomplete author asset: {name}; expected {expected_bytes} bytes, received {received}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("hidden_size") != 768 or config.get("vocab_size") != 30522:
        raise ValueError("The author cased MatBERT configuration differs")
    notice = Path(__file__).with_name("assets") / "MatBERT_LICENSE.txt"
    (directory / "LICENSE").write_text(notice.read_text(encoding="utf-8"), encoding="utf-8")
    return directory


def setup_model(destination=None):
    """Fetch the pinned surrogate release asset when installed without weights.

    The packaged manifest supplies the actual byte count and the owned GitHub
    release URL. Prediction itself never downloads anything.
    """
    root = Path(__file__).resolve().parent
    manifest_path = root / "models" / "artifact.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("No published model artifact manifest is available")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    url, expected = manifest["url"], manifest["bytes"]
    prefix = "https://github.com/ruiding-uchicago/PEDOT_PSS_hybrid/releases/download/"
    if (not isinstance(url, str) or not url.startswith(prefix)
            or type(expected) is not int or expected <= 0):
        raise ValueError("The packaged model artifact manifest is invalid")
    configured = destination or os.environ.get("PEDOT_PSS_MODEL")
    path = Path(configured).expanduser().resolve() if configured else root / "models" / "model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == expected:
        return path
    temporary = path.with_name(path.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "PEDOT-PSS-surrogate-model-setup/1"})
        received = 0
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if received > expected:
                    raise ValueError("The model asset exceeds its published byte count")
        if received != expected:
            raise ValueError(f"Incomplete model download: expected {expected} bytes, received {received}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
