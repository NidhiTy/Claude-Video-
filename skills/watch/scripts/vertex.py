#!/usr/bin/env python3
"""Vertex AI engine: Gemini watches the video through the user's own Google Cloud project.

Standard library only. A service-account key signs a short-lived OAuth token on this
machine; videos that are not YouTube URLs pass through the user's Cloud Storage bucket.
The key never leaves the machine and the token is scrubbed from every diagnostic.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from config import ConfigError
from gemini import _error_message, _processing, build_prompt
from runtime import diagnostic

TOKEN_URL = 'https://oauth2.googleapis.com/token'
SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
API = 'https://aiplatform.googleapis.com/v1beta1'  # Agentic video processing is not served on v1.
STORAGE = 'https://storage.googleapis.com'
RETRY_DELAYS = (10.0, 30.0)  # Vertex AI's shared quota answers 429 while a model is busy; Google advises backing off.
_SHA256_DIGEST_INFO = bytes.fromhex('3031300d060960864801650304020105000420')  # RFC 8017, section 9.2
_HINTS = {
    'auth': ('check the WATCH_VERTEX_CREDENTIALS key file, that the Vertex AI API (aiplatform.googleapis.com) is '
             'enabled, and that the service account has roles/aiplatform.user (plus roles/storage.objectUser on '
             'the bucket); a wrong system clock also invalidates the token'),
    'quota': 'the project is rate-limited or out of quota; wait or check its Vertex AI quota and billing',
    'rejected': ('Google refused this request; the video may be private, unsupported, or too long, or the model '
                 'or bucket name is wrong'),
    'service': 'Google had a server error; retry shortly',
    'network': 'could not reach Google Cloud; check network, proxy, or egress rules',
    'response': 'Vertex AI returned something this version cannot read',
    'upload': 'the video could not be uploaded to Cloud Storage',
}


def _der(data: bytes, at: int = 0) -> tuple[int, bytes, int]:
    """One DER element starting at `at`: (tag, value, offset of the next element)."""
    tag, length, at = data[at], data[at + 1], at + 2
    if length & 0x80:
        size = length & 0x7F
        length, at = int.from_bytes(data[at:at + size], 'big'), at + size
    return tag, data[at:at + length], at + length


def _rsa_key(pem: str) -> tuple[int, int, int]:
    """(n, e, d) from the PKCS#8 private key inside a service-account JSON key."""
    info = _der(base64.b64decode(''.join(line for line in pem.splitlines() if '-----' not in line)))[1]
    at = _der(info, _der(info)[2])[2]  # Skip the version and the algorithm identifier.
    key, numbers, at = _der(_der(info, at)[1])[1], [], 0
    for _ in range(4):  # RSAPrivateKey starts with version, n, e, d.
        _, value, at = _der(key, at)
        numbers.append(int.from_bytes(value, 'big'))
    return numbers[1], numbers[2], numbers[3]


def load_credentials(path: str, bucket: str = '') -> dict:
    """Read the service-account key. Diagnostics name the file, never its contents."""
    file = Path(path).expanduser()
    try:
        info = json.loads(file.read_text(encoding='utf-8'))
        if info['type'] != 'service_account':
            raise ValueError()
        n, e, d = _rsa_key(info['private_key'])
        if pow(pow(2, e, n), d, n) != 2:  # Also rejects a key this parser misread.
            raise ValueError()
        return {'email': str(info['client_email']), 'project': str(info['project_id']), 'n': n, 'd': d,
                'bucket': bucket.removeprefix('gs://').strip('/')}
    except OSError:
        raise ConfigError(f'Cannot read the service-account key {file}; check WATCH_VERTEX_CREDENTIALS.') from None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise ConfigError(f'{file} is not a usable service-account JSON key. Create a new JSON key for the '
                          'service account and point WATCH_VERTEX_CREDENTIALS at it.') from None


def _fail(category: str, detail: str, *secrets: str) -> SystemExit:
    for secret in filter(None, secrets):
        detail = detail.replace(secret, '[redacted]')
    return SystemExit(f'Vertex {category}: {_HINTS[category]}. Detail: {diagnostic(detail, 600) or "none"}. '
                      'No local fallback was attempted; rerun with --engine local to use frames + transcript.')


def _call(method: str, url: str, token: str, *, data=None, headers=None, timeout: float = 60.0, secret: str = '',
          retry=(), sleep=time.sleep):
    """One request. `retry` lists the waits before re-sending after a busy (429) or server (5xx) reply."""
    for delay in (*retry, None):
        request = Request(url, data=data, method=method,
                          headers={**({'Authorization': f'Bearer {token}'} if token else {}), **(headers or {})})
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            category = ('service' if exc.code >= 500 else 'quota' if exc.code == 429
                        else 'auth' if exc.code in (401, 403) or url == TOKEN_URL else 'rejected')
            if delay is None or category not in ('quota', 'service'):
                raise _fail(category, f'HTTP {exc.code}: {_error_message(exc.read())}', token, secret) from None
            print(f'[watch] Vertex AI is busy (HTTP {exc.code}); retrying in {delay:g}s…', file=sys.stderr)
        except (URLError, TimeoutError, OSError) as exc:
            raise _fail('network', f'{type(exc).__name__}: {getattr(exc, "reason", exc)}', token, secret) from None
        sleep(delay)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def _sign(message: bytes, n: int, d: int) -> bytes:
    """RSASSA-PKCS1-v1_5 with SHA-256 (JWT RS256). Deterministic, so it needs no random source.

    Plain modular exponentiation without blinding: sound for signing our own token request on
    the user's machine, not for a service that signs attacker-chosen input.
    """
    size = (n.bit_length() + 7) // 8
    digest = _SHA256_DIGEST_INFO + hashlib.sha256(message).digest()
    block = b'\x00\x01' + b'\xff' * (size - len(digest) - 3) + b'\x00' + digest
    return pow(int.from_bytes(block, 'big'), d, n).to_bytes(size, 'big')


def _token(creds: dict) -> str:
    """An OAuth access token for the service account, renewed five minutes before it expires."""
    if creds.get('expires', 0) - 300 > time.time():
        return creds['token']
    issued = int(time.time())
    claims = {'iss': creds['email'], 'scope': SCOPE, 'aud': TOKEN_URL, 'iat': issued, 'exp': issued + 3600}
    unsigned = '.'.join(_b64(json.dumps(part).encode('utf-8')) for part in ({'alg': 'RS256', 'typ': 'JWT'}, claims))
    assertion = f"{unsigned}.{_b64(_sign(unsigned.encode('ascii'), creds['n'], creds['d']))}"
    body = _call('POST', TOKEN_URL, '', secret=assertion,
                 data=urlencode({'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                                 'assertion': assertion}).encode('ascii'),
                 headers={'Content-Type': 'application/x-www-form-urlencoded'})
    try:
        data = json.loads(body)
        creds['token'], creds['expires'] = str(data['access_token']), issued + int(data.get('expires_in', 3600))
    except (ValueError, KeyError, TypeError):
        raise _fail('auth', 'unreadable token response from Google') from None
    return creds['token']


def ask(video: dict, question: str | None, *, model: str, key: dict, clip=None, timeout: float = 600.0,
        sleep=time.sleep) -> dict:
    """Same contract as gemini.ask; `key` is the dict from load_credentials.

    A busy or failing model is asked again after each RETRY_DELAYS wait, so an upload is not wasted.
    """
    processing, label = _processing(clip)
    part = {'fileData': {'fileUri': video['uri'], 'mimeType': video.get('mime_type') or 'video/mp4'},
            'mediaProcessing': 'AGENTIC'}
    if processing != 'agentic':  # As in the gemini engine, a clip uses static processing with offsets.
        part.update(mediaProcessing='STATIC', videoMetadata={
            name: processing[field] for name, field in (('startOffset', 'start_offset'), ('endOffset', 'end_offset'))
            if field in processing})
    payload = {'contents': [{'role': 'user', 'parts': [part, {'text': build_prompt(question)}]}]}
    token = _token(key)
    body = _call('POST', f"{API}/projects/{quote(key['project'], safe='')}/locations/global/publishers/google/"
                         f"models/{quote(model, safe='')}:generateContent", token,
                 data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, timeout=timeout,
                 retry=RETRY_DELAYS, sleep=sleep)
    try:
        data = json.loads(body)
        candidate = (data.get('candidates') or [{}])[0]
        parts = [item['text'] for item in (candidate.get('content') or {}).get('parts') or []
                 if item.get('text') and not item.get('thought')]
        reason = candidate.get('finishReason') or (data.get('promptFeedback') or {}).get('blockReason')
        tokens = (data.get('usageMetadata') or {}).get('totalTokenCount')
    except (ValueError, KeyError, TypeError, AttributeError):
        raise _fail('response', body[:300].decode('utf-8', errors='replace'), token) from None
    if not parts:
        raise _fail('response', f'no answer text (reason {reason})', token)
    return {'text': ''.join(parts).strip(), 'model': model, 'processing': f'{label} via Vertex AI',
            'total_tokens': tokens if isinstance(tokens, int) else None}


def upload_file(path: Path, key: dict) -> dict:
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        raise _fail('upload', f'{path.name} is empty')
    guessed = mimetypes.guess_type(path.name)[0] or ''
    mime = guessed if guessed.startswith('video/') else 'video/mp4'
    name = f'watch/{uuid.uuid4().hex}{path.suffix.lower()}'
    with path.open('rb') as stream:  # Streamed in blocks; never held in memory.
        _call('POST', f"{STORAGE}/upload/storage/v1/b/{quote(key['bucket'], safe='')}/o"
                      f"?uploadType=media&name={quote(name, safe='')}", _token(key), data=stream,
              headers={'Content-Type': mime, 'Content-Length': str(size)}, timeout=3600.0)
    return {'name': name, 'uri': f"gs://{key['bucket']}/{name}", 'mime_type': mime,
            'sent': 'video uploaded to your Cloud Storage bucket, deleted after the answer'}


def delete_file(name: str, key: dict) -> str | None:
    try:
        _call('DELETE', f"{STORAGE}/storage/v1/b/{quote(key['bucket'], safe='')}/o/{quote(name, safe='')}", _token(key))
        return None
    except (SystemExit, Exception) as exc:  # Cleanup must never mask the real result or error.
        reason = str(exc).partition('Detail: ')[2].partition('. No local fallback')[0] or type(exc).__name__
        return (f"Could not delete gs://{key['bucket']}/{name} ({reason}); "
                'delete it yourself or let a bucket lifecycle rule expire it.')
