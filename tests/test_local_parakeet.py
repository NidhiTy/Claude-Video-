"""Local Parakeet (MLX) backend: real child processes, no model downloads."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import config
import local_parakeet
import setup
import watch
import whisper


@pytest.fixture
def fake_parakeet(tmp_path):
    root = tmp_path / 'venv with spaces'
    (root / 'bin').mkdir(parents=True)
    exe = root / 'bin' / 'parakeet-mlx'
    exe.write_text(f'#!{sys.executable}\n' + '''import json, os, pathlib, sys
mode = os.environ.get('WATCH_TEST_MODE', 'good')
assert os.environ.get('HF_HUB_OFFLINE') == '1', 'inference must not touch the network'
argv = sys.argv[1:]
assert argv[argv.index('--output-format') + 1] == 'json'
model = argv[argv.index('--model') + 1]
out = pathlib.Path(argv[argv.index('--output-dir') + 1])
audio = pathlib.Path(argv[0])
if mode == 'swallowed':  # parakeet-mlx prints the error and still exits 0
    print('Error transcribing file'); sys.exit(0)
if mode == 'crash':
    print('Error loading model'); sys.exit(1)
sentences = [] if mode == 'empty' else [
    {'text': ' Hello there.', 'start': 0.5, 'end': 1.25, 'tokens': []},
    {'text': 'Second line', 'start': 2.0, 'end': 3.0, 'tokens': []},
]
(out / (audio.stem + '.json')).write_text(json.dumps({'text': model, 'sentences': sentences}))
''', encoding='utf-8')
    exe.chmod(0o755)
    (root / '.deps-ok').write_text('0.5.2')
    config.write_settings({'WATCH_PARAKEET_BIN': str(exe), 'WATCH_WHISPER_BACKEND': 'parakeet'})
    audio = tmp_path / 'audio with spaces.mp3'
    audio.write_bytes(b'audio')
    return exe, audio


def test_sentences_become_segments_offline(fake_parakeet):
    _, audio = fake_parakeet
    assert local_parakeet.transcribe_audio(audio) == [
        {'start': 0.5, 'end': 1.25, 'text': 'Hello there.'},
        {'start': 2.0, 'end': 3.0, 'text': 'Second line'},
    ]
    assert not list(audio.parent.glob('watch-parakeet-*'))


def test_default_model_is_multilingual_v3(fake_parakeet, monkeypatch):
    seen = []
    real = local_parakeet.command
    monkeypatch.setattr(local_parakeet, 'command', lambda *a: seen.append(real(*a)) or seen[-1])
    local_parakeet.transcribe_audio(fake_parakeet[1])
    assert 'mlx-community/parakeet-tdt-0.6b-v3' in seen[0]


def test_empty_is_no_speech(fake_parakeet, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'empty')
    assert local_parakeet.transcribe_audio(fake_parakeet[1]).no_speech


def test_exit_zero_without_output_is_failure(fake_parakeet, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'swallowed')
    with pytest.raises(SystemExit, match='Parakeet produced no transcript'):
        local_parakeet.transcribe_audio(fake_parakeet[1])


def test_nonzero_exit_is_failure(fake_parakeet, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'crash')
    with pytest.raises(SystemExit, match='Parakeet failed'):
        local_parakeet.transcribe_audio(fake_parakeet[1])


def test_incomplete_install_is_refused(fake_parakeet):
    exe, audio = fake_parakeet
    (exe.parent.parent / '.deps-ok').unlink()
    with pytest.raises(SystemExit, match='setup.py --backend parakeet'):
        local_parakeet.transcribe_audio(audio)


def test_route_never_resolves_a_cloud_key(fake_parakeet, monkeypatch):
    _, audio = fake_parakeet
    monkeypatch.setattr(whisper, 'load_api_key', lambda *a, **k: pytest.fail('no cloud credentials for local inference'))
    monkeypatch.setattr(whisper, 'extract_audio', lambda *a: audio)
    assert whisper.transcribe_video('video.mp4', audio, backend='parakeet')[1] == 'parakeet'


def test_watch_routes_configured_parakeet_locally(monkeypatch, capsys):
    monkeypatch.setenv('WATCH_WHISPER_BACKEND', 'parakeet')
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: {'video_path': 'video.mp4'})
    monkeypatch.setattr(watch, 'get_metadata', lambda *a: {'duration_seconds': 30, 'has_audio': True, 'has_video': False})
    monkeypatch.setattr(watch, 'load_api_key', lambda *a: pytest.fail('no cloud fallback'))
    def fake(*a, **kw):
        assert kw['backend'] == 'parakeet' and kw['api_key'] is None
        return [{'start': 1, 'end': 2, 'text': 'hi'}], 'parakeet'
    monkeypatch.setattr(watch, 'transcribe_video', fake)
    monkeypatch.setattr(sys, 'argv', ['watch', 'video.mp4', '--detail', 'transcript'])
    assert watch.main() == 0
    assert 'parakeet' in capsys.readouterr().out


def test_cli_flag_accepts_parakeet(monkeypatch):
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: {'video_path': 'video.mp4'})
    monkeypatch.setattr(watch, 'get_metadata', lambda *a: {'duration_seconds': 30, 'has_audio': True, 'has_video': False})
    calls = []
    monkeypatch.setattr(watch, 'transcribe_video', lambda *a, **kw: calls.append(kw['backend']) or ([], 'parakeet'))
    monkeypatch.setattr(sys, 'argv', ['watch', 'video.mp4', '--detail', 'transcript', '--whisper', 'parakeet'])
    assert watch.main() == 0 and calls == ['parakeet']


def test_installer_requires_apple_silicon(monkeypatch):
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(setup.platform, 'machine', lambda: 'x86_64')
    with pytest.raises(SystemExit, match='Apple Silicon'):
        setup.install_parakeet()
    assert config.get_config()['whisper_backend'] == 'auto'


def test_installer_marks_ready_only_after_warmup(monkeypatch):
    venv = Path.home() / '.cache/watch/parakeet-venv'
    commands = []
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(setup.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(setup, '_ensure_uv', lambda: '/fake/uv')
    monkeypatch.setattr(setup, '_which', lambda name: '/fake/' + name)
    def step(cmd, label):
        commands.append(cmd)
        if 'venv' in cmd:
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin/python').touch()
            (venv / 'bin/parakeet-mlx').touch()
        if cmd[0] == 'ffmpeg':
            Path(cmd[-1]).write_bytes(b'audio')
    monkeypatch.setattr(setup, '_install_step', step)
    warm = []
    def warmup(audio, cfg, *, require_ready, offline):
        assert not (venv / '.deps-ok').exists() and offline is False
        warm.append(cfg['parakeet_bin'])
        return []
    monkeypatch.setattr(local_parakeet, 'transcribe_audio', warmup)
    monkeypatch.setattr(setup, 'run_text', lambda *a, **kw: subprocess.CompletedProcess([], 0, '[]', ''))
    assert setup.install_parakeet() == 0
    assert (venv / '.deps-ok').exists() and warm == [str(venv / 'bin/parakeet-mlx')]
    assert any(f'parakeet-mlx=={setup.PARAKEET_VERSION}' in map(str, c) for c in commands)
    cfg = config.get_config()
    assert cfg['whisper_backend'] == 'parakeet' and cfg['parakeet_bin'] == str(venv / 'bin/parakeet-mlx')
    assert not setup.is_first_run()


def test_failed_warmup_never_marks_ready(monkeypatch):
    venv = Path.home() / '.cache/watch/parakeet-venv'
    (venv / 'bin').mkdir(parents=True)
    for name in ['python', 'parakeet-mlx']:
        (venv / 'bin' / name).touch()
    (venv / '.deps-ok').write_text('old')
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(setup.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(setup, '_ensure_uv', lambda: '/fake/uv')
    monkeypatch.setattr(setup, '_which', lambda name: '/fake/' + name)
    monkeypatch.setattr(setup, '_install_step', lambda *a: None)
    def fail(*a, **kw):
        raise SystemExit('warmup failed')
    monkeypatch.setattr(local_parakeet, 'transcribe_audio', fail)
    with pytest.raises(SystemExit, match='warmup failed'):
        setup.install_parakeet()
    assert not (venv / '.deps-ok').exists()
    assert config.get_config()['whisper_backend'] == 'auto'
