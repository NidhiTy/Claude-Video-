"""Subprocess-only Parakeet (parakeet-mlx) adapter for Apple Silicon. Never imports MLX into watch."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from config import get_config, positive_number
from runtime import diagnostic
from transcribe import normalize_segments

DEFAULT_MODEL = 'mlx-community/parakeet-tdt-0.6b-v3'
CHUNK_SECONDS = '120'
OVERLAP_SECONDS = '15'


def settings(cfg=None) -> dict:
    cfg = dict(get_config(backend_override='parakeet') if cfg is None else cfg)
    cfg['parakeet_timeout'] = (positive_number(cfg['parakeet_timeout'], 'WATCH_PARAKEET_TIMEOUT')
                               if cfg.get('parakeet_timeout') else None)
    return cfg


def sentinel_for(executable: Path) -> Path:
    return executable.parent.parent / '.deps-ok'


def executable_path(cfg: dict, *, require_ready=True) -> Path:
    value = cfg['parakeet_bin']
    if not value or not Path(value).is_absolute() or not Path(value).is_file():
        raise SystemExit('Parakeet executable missing. Run setup.py --backend parakeet.')
    executable = Path(value)
    if require_ready and not sentinel_for(executable).is_file():
        raise SystemExit('Parakeet venv is incomplete (no .deps-ok). Run setup.py --backend parakeet to rebuild it.')
    return executable


def command(executable: Path, audio: Path, output: Path, cfg: dict) -> list[str]:
    return [str(executable), str(audio.resolve()), '--model', cfg['parakeet_model'],
            '--output-dir', str(output.resolve()), '--output-format', 'json',
            '--chunk-duration', CHUNK_SECONDS, '--overlap-duration', OVERLAP_SECONDS]


def child_env(*, offline: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONIOENCODING='utf-8', PYTHONWARNINGS='ignore', HF_HUB_DISABLE_TELEMETRY='1')
    if offline:
        # Inference uses only the warmed cache; setup is the one step allowed to download.
        env.update(HF_HUB_OFFLINE='1')
    return env


def transcribe_audio(audio: Path, cfg=None, *, require_ready=True, offline=True):
    cfg = settings(cfg)
    executable = executable_path(cfg, require_ready=require_ready)
    print(f"[watch] local Parakeet ({cfg['parakeet_model']}) transcription on Apple Silicon…", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix='watch-parakeet-', dir=audio.parent) as temporary:
        output = Path(temporary)
        try:
            result = subprocess.run(command(executable, audio, output, cfg), capture_output=True,
                                    env=child_env(offline=offline), timeout=cfg['parakeet_timeout'])
        except subprocess.TimeoutExpired:
            raise SystemExit('Parakeet timed out and was stopped. Increase WATCH_PARAKEET_TIMEOUT or leave it unset.') from None
        except OSError as exc:
            raise SystemExit(f'Parakeet executable is not runnable ({type(exc).__name__}); rerun setup.py --backend parakeet.') from None
        detail = diagnostic(result.stdout + result.stderr, 1500)
        if result.returncode != 0:
            raise SystemExit(f'Parakeet failed (exit {result.returncode}). Rerun setup.py --backend parakeet if the model '
                             f'cache or dependencies are broken.\n{detail}')
        path = output / f'{audio.stem}.json'
        # parakeet-mlx reports per-file errors and still exits 0, so a missing file is a failure.
        if not path.is_file():
            raise SystemExit(f'Parakeet produced no transcript (exit 0).\n{detail}')
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            rows, floor = [], 0.0
            for s in data['sentences']:
                if not str(s.get('text', '')).strip():
                    continue
                # Chunk merges can emit a sentence a hair before the previous one;
                # clamp forward instead of rejecting the whole transcript.
                start = max(float(s['start']), floor)
                end = max(float(s['end']), start)
                floor = start
                rows.append({'start': start, 'end': end, 'text': s['text']})
            segments = normalize_segments({'segments': rows})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SystemExit(f'Parakeet output missing or malformed ({type(exc).__name__}).') from None
    if segments.no_speech:
        print('[watch] Parakeet detected no speech.', file=sys.stderr)
    return segments
