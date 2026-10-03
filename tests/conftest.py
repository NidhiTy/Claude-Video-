"""Shared pytest fixtures: ffmpeg-synthesized clips and scripts/ on sys.path."""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import os
import textwrap
from pathlib import Path

import pytest

# Make the bundled scripts importable (mirrors watch.py's sys.path insert).
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def isolated_user(monkeypatch, tmp_path):
    """No test inherits a developer's credentials, preferences, or project .env."""
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    for key in list(os.environ):
        if key.startswith("WATCH_") or key in {
            "GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "SETUP_COMPLETE", "XDG_CONFIG_HOME",
        }:
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.chdir(cwd)
    import config
    monkeypatch.setattr(config, "CONFIG_DIR", home / ".config" / "watch")
    monkeypatch.setattr(config, "CONFIG_FILE", config.CONFIG_DIR / ".env")
    if "setup" in sys.modules:
        monkeypatch.setattr(sys.modules["setup"], "CONFIG_DIR", config.CONFIG_DIR)
        monkeypatch.setattr(sys.modules["setup"], "CONFIG_FILE", config.CONFIG_FILE)
        sys.modules["setup"]._PERM_WARNED.clear()
    import whisper
    def no_upload(*args, **kwargs):
        raise AssertionError("Tests must mock cloud uploads")
    monkeypatch.setattr(whisper, "urlopen", no_upload)
    import gemini
    def no_gemini(*args, **kwargs):
        raise AssertionError("Tests must mock Gemini calls")
    monkeypatch.setattr(gemini, "urlopen", no_gemini)
    import vertex
    def no_vertex(*args, **kwargs):
        raise AssertionError("Tests must mock Vertex AI and Cloud Storage calls")
    monkeypatch.setattr(vertex, "urlopen", no_vertex)

# 14 visually distinct fills → 14 abrupt cuts → x264 emits a keyframe per cut.
COLORS = [
    "red", "green", "blue", "white", "black", "yellow", "cyan",
    "magenta", "gray", "orange", "purple", "brown", "navy", "olive",
]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def build_cut_clip(
    path: Path,
    n: int = 14,
    seg: float = 0.4,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """Concatenate ``n`` solid-color segments into one clip with ``n`` cuts.

    Each color change is a hard scene cut, so the scene selector finds ~n-1
    changes. x264's own scenecut detection is unreliable on flat fills, so we
    force a keyframe at every ``seg`` boundary — giving ~n real keyframes for
    the keyframe engine to find.
    """
    inputs: list[str] = []
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        inputs += ["-f", "lavfi", "-t", str(seg), "-i", f"color=c={color}:s={size}:r={fps}"]
    streams = "".join(f"[{i}:v]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=0[out]"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filt, "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
        str(path),
    ])


def build_static_clip(
    path: Path,
    duration: float = 3.0,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """One solid color: 1 keyframe, no scene changes → triggers both fallbacks."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=blue:s={size}:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "600",
        str(path),
    ])


# A throwaway RSA key from two well-known primes (2**255 - 19 and 2**256 - 189): large enough for
# a SHA-256 signature, generated nowhere, and not a credential for anything.
KEY_P, KEY_Q, KEY_E = 2**255 - 19, 2**256 - 189, 65537
KEY_N, KEY_D = KEY_P * KEY_Q, pow(KEY_E, -1, (KEY_P - 1) * (KEY_Q - 1))


def _der(tag: int, body: bytes) -> bytes:
    size = len(body).to_bytes((len(body).bit_length() + 7) // 8, "big")
    return bytes([tag]) + (bytes([len(body)]) if len(body) < 128 else bytes([0x80 | len(size)]) + size) + body


def _der_int(value: int) -> bytes:
    return _der(0x02, value.to_bytes(value.bit_length() // 8 + 1, "big"))


def build_service_account_key(path: Path, **overrides) -> Path:
    """Write a service-account JSON key shaped like a Google Cloud download (PKCS#8 PEM inside)."""
    numbers = (0, KEY_N, KEY_E, KEY_D, KEY_P, KEY_Q, KEY_D % (KEY_P - 1), KEY_D % (KEY_Q - 1), pow(KEY_Q, -1, KEY_P))
    rsa = _der(0x30, b"".join(_der_int(value) for value in numbers))
    algorithm = _der(0x30, bytes.fromhex("06092a864886f70d0101010500"))  # rsaEncryption, NULL
    info = _der(0x30, _der_int(0) + algorithm + _der(0x04, rsa))
    body = "\n".join(textwrap.wrap(base64.b64encode(info).decode("ascii"), 64))
    path.write_text(json.dumps({
        "type": "service_account", "project_id": "unit-project",
        "client_email": "watch@unit-project.iam.gserviceaccount.com",
        "private_key": f"-----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----\n", **overrides,
    }), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def cut_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "cuts.mp4"
    build_cut_clip(path)
    return path


@pytest.fixture(scope="session")
def static_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "static.mp4"
    build_static_clip(path)
    return path
