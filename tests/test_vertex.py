"""Vertex adapter: key parsing, token signing, payloads, and failures against a fake urlopen. No network."""
from __future__ import annotations

import base64
import hashlib
import json
from urllib.error import URLError
from urllib.parse import parse_qs, quote

import pytest

import config
import vertex
from conftest import KEY_D, KEY_E, KEY_N, build_service_account_key
from test_gemini import FakeResponse, http_error

TOKEN = 'unit-test-token-DO-NOT-LEAK'
SA_EMAIL = 'watch@unit-project.iam.gserviceaccount.com'


def answer(*texts):
    parts = [{'text': 'hidden reasoning', 'thought': True}, *({'text': text} for text in texts)]
    return json.dumps({'candidates': [{'content': {'role': 'model', 'parts': parts}, 'finishReason': 'STOP'}],
                       'usageMetadata': {'totalTokenCount': 4348}}).encode()


@pytest.fixture
def calls(monkeypatch):
    """Record every request; replies are popped from calls.replies in order."""
    class Calls(list):
        replies = []
    recorded = Calls()
    def fake(request, timeout=None):
        recorded.append(request)
        reply = recorded.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(vertex, 'urlopen', fake)
    return recorded


@pytest.fixture
def creds(tmp_path):
    """Credentials holding a fresh token, so each test sees only the call it is about."""
    loaded = vertex.load_credentials(str(build_service_account_key(tmp_path / 'key.json')), 'gs://unit-bucket/')
    loaded.update(token=TOKEN, expires=vertex.time.time() + 3600)
    return loaded


def test_key_file_is_parsed_and_bucket_normalized(creds):
    assert (creds['email'], creds['project'], creds['bucket']) == (SA_EMAIL, 'unit-project', 'unit-bucket')
    assert (creds['n'], creds['d']) == (KEY_N, KEY_D)


def test_signature_verifies_and_matches_openssl():
    signature = vertex._sign(b'header.claims', KEY_N, KEY_D)
    recovered = pow(int.from_bytes(signature, 'big'), KEY_E, KEY_N).to_bytes(64, 'big')
    assert recovered == (b'\x00\x01' + b'\xff' * 10 + b'\x00' + bytes.fromhex('3031300d060960864801650304020105000420')
                         + hashlib.sha256(b'header.claims').digest())
    # The bytes `openssl dgst -sha256 -sign` writes for this key and message.
    assert hashlib.sha256(signature).hexdigest() == 'ab30036f3cbe5a980ac6eab6497574f8ef51fb836e028bd404630998fcfe94a6'


@pytest.mark.parametrize('overrides,expected', [
    ({'type': 'authorized_user'}, 'not a usable service-account'),
    ({'private_key': 'private-secret'}, 'not a usable service-account'),
    ({'private_key': None}, 'not a usable service-account'),
])
def test_unusable_key_files_fail_without_echoing_contents(tmp_path, overrides, expected):
    path = build_service_account_key(tmp_path / 'key.json', **overrides)
    with pytest.raises(config.ConfigError, match=expected) as caught:
        vertex.load_credentials(str(path))
    assert 'private-secret' not in str(caught.value) and 'key.json' in str(caught.value)


def test_missing_or_non_json_key_file(tmp_path):
    with pytest.raises(config.ConfigError, match='Cannot read.*WATCH_VERTEX_CREDENTIALS'):
        vertex.load_credentials(str(tmp_path / 'absent.json'))
    (tmp_path / 'notes.txt').write_text('private-secret')
    with pytest.raises(config.ConfigError, match='not a usable service-account') as caught:
        vertex.load_credentials(str(tmp_path / 'notes.txt'))
    assert 'private-secret' not in str(caught.value)


def test_token_comes_from_a_signed_jwt_and_is_reused_until_it_nears_expiry(calls, tmp_path):
    creds = vertex.load_credentials(str(build_service_account_key(tmp_path / 'key.json')))
    calls.replies = [FakeResponse(json.dumps({'access_token': TOKEN, 'expires_in': 3599}).encode())]
    assert vertex._token(creds) == TOKEN and vertex._token(creds) == TOKEN
    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == 'https://oauth2.googleapis.com/token' and request.get_header('Authorization') is None
    form = parse_qs(request.data.decode())
    assert form['grant_type'] == ['urn:ietf:params:oauth:grant-type:jwt-bearer']
    header, claims, signature = form['assertion'][0].split('.')
    decode = lambda part: base64.urlsafe_b64decode(part + '=' * (-len(part) % 4))
    assert json.loads(decode(header)) == {'alg': 'RS256', 'typ': 'JWT'}
    payload = json.loads(decode(claims))
    assert payload['iss'] == SA_EMAIL and payload['aud'] == 'https://oauth2.googleapis.com/token'
    assert payload['scope'] == 'https://www.googleapis.com/auth/cloud-platform'
    assert payload['exp'] - payload['iat'] == 3600
    assert decode(signature) == vertex._sign(f'{header}.{claims}'.encode(), KEY_N, KEY_D)
    creds['expires'] = vertex.time.time() + 200  # Inside the five-minute renewal margin.
    calls.replies = [FakeResponse(json.dumps({'access_token': 'renewed'}).encode())]
    assert vertex._token(creds) == 'renewed' and len(calls) == 2


def test_token_endpoint_rejection_is_an_auth_failure(calls, tmp_path):
    creds = vertex.load_credentials(str(build_service_account_key(tmp_path / 'key.json')))
    calls.replies = [http_error(400, b'{"error":"invalid_grant","error_description":"Invalid JWT Signature."}')]
    with pytest.raises(SystemExit, match='^Vertex auth:.*invalid_grant'):
        vertex.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=creds)
    assert len(calls) == 1, 'no model call without a token'


def test_agentic_youtube_payload_on_v1beta1_with_a_bearer_token(calls, creds):
    calls.replies = [FakeResponse(answer('At 00:10 the screen ', 'turns blue.'))]
    result = vertex.ask({'uri': 'https://youtu.be/abc'}, 'When does it change?', model='gemini-3.7-flash', key=creds)
    request = calls[0]
    assert request.full_url == ('https://aiplatform.googleapis.com/v1beta1/projects/unit-project/locations/global/'
                                'publishers/google/models/gemini-3.7-flash:generateContent')
    assert request.get_header('Authorization') == f'Bearer {TOKEN}'
    video, text = json.loads(request.data)['contents'][0]['parts']
    assert video == {'fileData': {'fileUri': 'https://youtu.be/abc', 'mimeType': 'video/mp4'}, 'mediaProcessing': 'AGENTIC'}
    assert 'When does it change?' in text['text'] and 'MM:SS' in text['text']
    assert result == {'text': 'At 00:10 the screen turns blue.', 'model': 'gemini-3.7-flash',
                      'processing': 'agentic via Vertex AI', 'total_tokens': 4348}


def test_clip_uses_static_processing_with_offsets(calls, creds):
    calls.replies = [FakeResponse(answer('Blue.')), FakeResponse(answer('ok'))]
    video = {'uri': 'gs://unit-bucket/watch/x.mov', 'mime_type': 'video/quicktime'}
    result = vertex.ask(video, None, model='m', key=creds, clip=(1200.4, 1500.2))
    assert json.loads(calls[0].data)['contents'][0]['parts'][0] == {
        'fileData': {'fileUri': 'gs://unit-bucket/watch/x.mov', 'mimeType': 'video/quicktime'},
        'mediaProcessing': 'STATIC', 'videoMetadata': {'startOffset': '1200s', 'endOffset': '1501s'}}
    assert result['processing'] == 'static clip 20:00–25:01 via Vertex AI'
    vertex.ask(video, None, model='m', key=creds, clip=(90.0, None))
    assert json.loads(calls[1].data)['contents'][0]['parts'][0]['videoMetadata'] == {'startOffset': '90s'}


def test_blocked_or_empty_answer_is_a_failure(calls, creds):
    calls.replies = [FakeResponse(json.dumps({'promptFeedback': {'blockReason': 'PROHIBITED_CONTENT'}}).encode())]
    with pytest.raises(SystemExit, match='Vertex response:.*PROHIBITED_CONTENT'):
        vertex.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=creds)


@pytest.mark.parametrize('reply,category', [
    (http_error(401, b'{"error":{"code":401,"message":"Request had invalid authentication credentials.","status":"UNAUTHENTICATED"}}'), 'auth'),
    (http_error(403, b'{"error":{"code":403,"message":"Permission \'aiplatform.endpoints.predict\' denied.","status":"PERMISSION_DENIED"}}'), 'auth'),
    (http_error(429, b'{"error":{"code":429,"message":"Resource exhausted.","status":"RESOURCE_EXHAUSTED"}}'), 'quota'),
    (http_error(404, b'{"error":{"code":404,"message":"Publisher Model was not found.","status":"NOT_FOUND"}}'), 'rejected'),
    (http_error(503, b'upstream unavailable'), 'service'),
    (URLError('certificate verify failed'), 'network'),
    (TimeoutError('timed out'), 'network'),
    (FakeResponse(b'<html>not json</html>'), 'response'),
])
def test_failures_are_categorized_and_never_leak_the_token(calls, creds, reply, category):
    calls.replies = [reply] * 3
    with pytest.raises(SystemExit) as caught:
        vertex.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=creds, sleep=lambda seconds: None)
    message = str(caught.value)
    assert message.startswith(f'Vertex {category}:')
    assert TOKEN not in message
    assert '--engine local' in message


def test_busy_and_server_errors_are_retried_then_reported(calls, creds):
    busy = b'{"error":{"code":429,"message":"Resource exhausted. Please try again later."}}'
    calls.replies = [http_error(429, busy), http_error(503, b'unavailable'), FakeResponse(answer('Blue.'))]
    naps = []
    assert vertex.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=creds, sleep=naps.append)['text'] == 'Blue.'
    assert naps == [10.0, 30.0] and len(calls) == 3
    assert json.loads(calls[2].data) == json.loads(calls[0].data), 'the same request is sent again'
    calls.replies = [http_error(429, busy) for _ in range(3)]
    with pytest.raises(SystemExit, match='^Vertex quota:.*try again later'):
        vertex.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=creds, sleep=naps.append)
    assert len(calls) == 6 and naps == [10.0, 30.0, 10.0, 30.0]
    calls.replies = [http_error(403, b'{"error":{"message":"denied"}}')]
    with pytest.raises(SystemExit, match='^Vertex auth:'):
        vertex.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=creds, sleep=naps.append)
    assert len(calls) == 7, 'a refusal is not retried'


def test_token_echoed_by_the_server_is_redacted(calls, creds):
    calls.replies = [http_error(400, json.dumps({'error': {'message': f'bad token {TOKEN}'}}).encode())]
    with pytest.raises(SystemExit) as caught:
        vertex.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=creds)
    assert TOKEN not in str(caught.value) and '[redacted]' in str(caught.value)


def test_upload_streams_to_the_bucket_and_delete_removes_the_object(calls, creds, tmp_path):
    clip = tmp_path / 'clip.MP4'
    clip.write_bytes(b'x' * 1000)
    calls.replies = [FakeResponse(b'{}'), FakeResponse(b'')]
    uploaded = vertex.upload_file(clip, creds)
    name = uploaded['name']
    assert name.startswith('watch/') and name.endswith('.mp4')
    assert uploaded == {'name': name, 'uri': f'gs://unit-bucket/{name}', 'mime_type': 'video/mp4',
                        'sent': 'video uploaded to your Cloud Storage bucket, deleted after the answer'}
    upload = calls[0]
    assert upload.get_method() == 'POST' and upload.full_url == (
        f'https://storage.googleapis.com/upload/storage/v1/b/unit-bucket/o?uploadType=media&name={quote(name, safe="")}')
    assert upload.get_header('Authorization') == f'Bearer {TOKEN}'
    assert upload.get_header('Content-type') == 'video/mp4' and upload.get_header('Content-length') == '1000'
    assert hasattr(upload.data, 'read'), 'the body must be streamed, not loaded into memory'
    assert vertex.delete_file(name, creds) is None
    assert calls[1].get_method() == 'DELETE'
    assert calls[1].full_url == f'https://storage.googleapis.com/storage/v1/b/unit-bucket/o/{quote(name, safe="")}'


def test_empty_file_is_refused_and_delete_is_best_effort(calls, creds, tmp_path):
    empty = tmp_path / 'empty.mp4'
    empty.write_bytes(b'')
    with pytest.raises(SystemExit, match='Vertex upload:.*empty'):
        vertex.upload_file(empty, creds)
    calls.replies = [http_error(403, b'{"error":{"code":403,"message":"storage.objects.delete denied"}}')]
    warning = vertex.delete_file('watch/abc.mp4', creds)
    assert 'gs://unit-bucket/watch/abc.mp4' in warning and 'lifecycle' in warning and TOKEN not in warning
    assert 'storage.objects.delete denied' in warning, "the warning carries Google's reason"
