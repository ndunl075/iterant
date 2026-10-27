"""HTTP front end: ``POST /v1/messages`` in Anthropic format, nothing else.

Sandboxes set ``ANTHROPIC_BASE_URL`` to this server and ``ANTHROPIC_API_KEY`` to their
session token. The ``claude`` profile forwards the request body to the Anthropic Messages
API, streaming included. The ``free`` profile translates to an OpenAI-compatible
``/chat/completions`` upstream (see ``translate.py``). The real key is added here, on
the way out, and scrubbed from anything that comes back.

There are no retries and no fallbacks: an upstream failure goes straight back to the caller.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import IO, Any

from iterant.contracts import Profile
from iterant.proxy.core import Admission, LedgerError, Proxy
from iterant.proxy.jsonutil import Obj, obj
from iterant.proxy.ledger import Usage, estimate_tokens
from iterant.proxy.rules import NO_FALLBACK_ROLES, Rejection
from iterant.proxy.sse import StreamAccumulator, sse_event, usage_of
from iterant.proxy.translate import (
    OpenAIStreamTranslator,
    Unsupported,
    from_openai_response,
    to_openai_request,
)

log = logging.getLogger("iterant.proxy")

_CLIENT_DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
_FORWARD_RESPONSE_HEADERS = ("retry-after", "request-id", "anthropic-request-id")


def error_body(kind: str, message: str, code: str | None = None) -> bytes:
    err: Obj = {"type": kind, "message": message}
    if code:
        err["iterant_code"] = code
    return json.dumps({"type": "error", "error": err}).encode()


class _Upstream:
    """An open upstream response: status, headers and a readable body."""

    def __init__(self, status: int, headers: dict[str, str], body: IO[bytes]) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def close(self) -> None:
        self.body.close()


class ProxyServer:
    def __init__(self, proxy: Proxy, host: str = "127.0.0.1", port: int = 0) -> None:
        self.proxy = proxy
        handler = _make_handler(proxy)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host!s}:{port}"

    def start(self) -> ProxyServer:
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> ProxyServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _scrub(proxy: Proxy, data: bytes) -> bytes:
    for secret in proxy.config.secrets:
        data = data.replace(secret.encode(), b"[redacted]")
    return data


def _open_upstream(
    proxy: Proxy, url: str, headers: dict[str, str], data: bytes
) -> _Upstream | Rejection:
    cfg = proxy.config
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(
        *([] if cfg.trust_env_proxy else [urllib.request.ProxyHandler({})])
    )
    try:
        resp: Any = opener.open(req, timeout=cfg.upstream_timeout_s)  # noqa: S310
    except urllib.error.HTTPError as exc:
        return _Upstream(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        msg = _scrub(proxy, f"iterant proxy: upstream unreachable: {exc}".encode()).decode()
        return Rejection(502, "api_error", "upstream_unreachable", msg)
    return _Upstream(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp)


def _cutoff_usage(adm: Admission, seen: Usage | None, est_in: int) -> Usage:
    """Usage for a stream that never finished: the input already reported (or estimated)
    plus the most the call could still have produced. Never zero input."""
    base = seen if seen is not None and seen.in_total > 0 else Usage(input=est_in)
    if base.in_total == 0:
        base = Usage(input=estimate_tokens(adm.request_bytes))
    out = max(int(adm.body["max_tokens"]), base.output)
    return Usage(base.input, out, base.cache_write, base.cache_read)


def _parse(raw: bytes) -> Obj | None:
    try:
        return obj(json.loads(raw))
    except ValueError:
        return None


def _lines(body: IO[bytes]) -> Iterator[bytes]:
    while line := body.readline():
        yield line


def _make_handler(proxy: Proxy) -> type[BaseHTTPRequestHandler]:
    cfg = proxy.config
    slots = threading.BoundedSemaphore(cfg.max_concurrent)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # body ends at close: simplest correct SSE relay
        timeout = cfg.socket_timeout_s
        _slot = False
        _timer: threading.Timer | None = None

        def setup(self) -> None:
            super().setup()
            # One slot per connection from the first byte, so slow-drip clients count too.
            self._slot = slots.acquire(blocking=False)
            # Total deadline for headers + body (the socket timeout is per read only).
            self._timer = threading.Timer(cfg.request_deadline_s, self._expire)
            self._timer.daemon = True
            self._timer.start()

        def _expire(self) -> None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        def _body_received(self) -> None:
            if self._timer is not None:
                self._timer.cancel()

        def finish(self) -> None:
            self._body_received()
            if self._slot:
                self._slot = False
                slots.release()
            try:
                super().finish()
            except OSError:
                pass

        _streaming = False

        def log_message(self, format: str, *args: object) -> None:
            return  # never log request lines: headers and bodies may hold tokens

        # --- plumbing -----------------------------------------------------------

        def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject(self, r: Rejection) -> None:
            self._send(r.status, error_body(r.kind, r.message, r.code))

        def _token(self) -> str | None:
            key = self.headers.get("x-api-key")
            if key:
                return key.strip()
            auth = self.headers.get("authorization", "")
            return auth[7:].strip() if auth.lower().startswith("bearer ") else None

        def do_GET(self) -> None:
            self._send(
                404, error_body("not_found_error", "iterant proxy serves POST /v1/messages only")
            )

        do_PUT = do_DELETE = do_PATCH = do_GET

        def _drain(self, length: int) -> None:
            """Read a bounded part of an unwanted body so the client sees our reply, not a reset."""
            try:
                self.rfile.read(max(0, min(length, 1024 * 1024)))
            except OSError:
                pass

        def _length(self) -> int:
            try:
                return int(self.headers.get("content-length", ""))
            except ValueError:
                return -1

        def do_POST(self) -> None:
            if not self._slot:
                self._send(503, error_body("overloaded_error", "iterant proxy: too many requests"))
                self._drain(self._length())
                return
            try:
                self._post()
            except OSError:
                pass  # client gone, timed out or past the deadline

        def _post(self) -> None:
            length = self._length()
            token = self._token()
            if (denied := proxy.check_token(token)) is not None:  # before reading any body
                self._reject(denied)
                self._drain(length)
                return
            if length < 0:
                self._send(411, error_body("invalid_request_error", "content-length required"))
                return
            if length > cfg.max_body_bytes:
                self._send(413, error_body("request_too_large", "request body too large"))
                self._drain(length)
                return
            raw = self.rfile.read(length)
            self._body_received()
            if self.path.split("?")[0] != "/v1/messages":
                self._send(
                    404,
                    error_body("not_found_error", "iterant proxy serves POST /v1/messages only"),
                )
                return
            try:
                adm = proxy.admit(token, raw, self.headers.get("anthropic-beta", ""))
            except Exception as exc:  # a bug in admit must be a reply, never a dropped socket
                log.error("unexpected %s while admitting", type(exc).__name__)
                self._try_send(500, error_body("api_error", "iterant proxy: internal error"))
                return
            if isinstance(adm, Rejection):
                self._reject(adm)
                return
            try:
                self._forward(adm)
            except _CLIENT_DISCONNECT:
                pass
            except LedgerError:
                log.error("call sink failed; proxy now refuses new calls")
                if not self._streaming:
                    self._try_send(500, error_body("api_error", "iterant proxy: call log failed"))
            except Exception as exc:
                log.error("unexpected %s while forwarding", type(exc).__name__)
                if not self._streaming:
                    self._try_send(500, error_body("api_error", "iterant proxy: internal error"))

        def _try_send(self, status: int, body: bytes) -> None:
            try:
                self._send(status, body)
            except _CLIENT_DISCONNECT:
                pass

        # --- forwarding ---------------------------------------------------------

        def _forward(self, adm: Admission) -> None:
            settled = False
            try:
                if cfg.policy.profile is Profile.CLAUDE:
                    settled = self._forward_anthropic(adm)
                else:
                    settled = self._forward_openai(adm)
            finally:
                if not settled:
                    proxy.release(adm)

        def _relay_error(self, up: _Upstream, translate: bool) -> None:
            body = _scrub(proxy, up.body.read())
            up.close()
            if translate:  # OpenAI-style error -> Anthropic shape
                msg = body.decode(errors="replace")[:2000]
                kind = "api_error" if up.status >= 500 else "invalid_request_error"
                body = error_body(kind, f"iterant proxy: upstream returned {up.status}: {msg}")
            self.send_response(up.status)
            self.send_header("content-type", up.headers.get("content-type", "application/json"))
            for h in _FORWARD_RESPONSE_HEADERS:
                if h in up.headers:
                    self.send_header(h, up.headers[h])
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _start_stream(self) -> None:
            self._streaming = True
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()

        def _forward_anthropic(self, adm: Admission) -> bool:
            assert cfg.anthropic_base_url and cfg.anthropic_api_key
            headers = {
                "content-type": "application/json",
                "x-api-key": cfg.anthropic_api_key,
                "anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
            }
            beta = self.headers.get("anthropic-beta")
            if beta and adm.session.role not in NO_FALLBACK_ROLES:  # rules reject it earlier
                headers["anthropic-beta"] = beta
            data = json.dumps(adm.body, separators=(",", ":")).encode()
            up = _open_upstream(
                proxy, cfg.anthropic_base_url.rstrip("/") + "/v1/messages", headers, data
            )
            if isinstance(up, Rejection):
                self._reject(up)
                return False
            if up.status != 200:
                self._relay_error(up, translate=False)
                return False
            if adm.stream:
                return self._relay_anthropic_stream(adm, up)
            raw = up.body.read()
            up.close()
            message = _parse(raw)
            if message is None:
                self._send(502, error_body("api_error", "iterant proxy: bad upstream response"))
                return False
            usage, refusal = usage_of(message)
            proxy.settle(adm, usage, refusal, message)
            self._send(200, _scrub(proxy, raw))
            return True

        def _relay_anthropic_stream(self, adm: Admission, up: _Upstream) -> bool:
            acc = StreamAccumulator()
            self._start_stream()
            gone = False  # client left: keep reading upstream so we learn the real usage
            try:
                for line in _lines(up.body):
                    acc.feed_line(line)
                    if not gone:
                        try:
                            self.wfile.write(_scrub(proxy, line))
                            if line.strip() == b"":
                                self.wfile.flush()
                        except OSError:
                            gone = True
            except OSError:
                pass  # upstream cut: settle on what we saw
            finally:
                up.close()
            seen = acc.usage if acc.saw_usage else None
            usage = seen if acc.stopped and seen else _cutoff_usage(adm, seen, 0)
            proxy.settle(adm, usage, acc.refused, acc.message())
            return True

        def _forward_openai(self, adm: Admission) -> bool:
            assert cfg.openai_base_url
            try:
                payload = to_openai_request(adm.body)
            except Unsupported as exc:
                self._send(400, error_body("invalid_request_error", f"iterant proxy: {exc}"))
                return False
            headers = {"content-type": "application/json"}
            if cfg.openai_api_key:
                headers["authorization"] = f"Bearer {cfg.openai_api_key}"
            data = json.dumps(payload).encode()
            up = _open_upstream(
                proxy, cfg.openai_base_url.rstrip("/") + "/chat/completions", headers, data
            )
            if isinstance(up, Rejection):
                self._reject(up)
                return False
            if up.status != 200:
                self._relay_error(up, translate=True)
                return False
            est_in = estimate_tokens(adm.request_bytes)
            model = adm.cap.model
            if adm.stream:
                return self._relay_openai_stream(
                    adm, up, OpenAIStreamTranslator(model, est_in), est_in
                )
            raw = up.body.read()
            up.close()
            resp = _parse(raw)
            if resp is None:
                self._send(502, error_body("api_error", "iterant proxy: bad upstream response"))
                return False
            message = from_openai_response(resp, model)
            usage, refusal = usage_of(message)
            if "usage" not in resp:  # upstream gave no counts: estimate, never charge zero
                usage = Usage(input=est_in, output=estimate_tokens(len(raw)))
            proxy.settle(adm, usage, refusal, message)
            self._send(200, json.dumps(message).encode())
            return True

        def _relay_openai_stream(
            self, adm: Admission, up: _Upstream, tr: OpenAIStreamTranslator, est_in: int
        ) -> bool:
            acc = StreamAccumulator()
            self._start_stream()
            gone = False

            def emit(events: list[tuple[str, Obj]]) -> None:
                nonlocal gone
                for name, data in events:
                    acc.feed_event(data)
                    if not gone:
                        try:
                            self.wfile.write(sse_event(name, data))
                        except OSError:
                            gone = True
                if not gone:
                    try:
                        self.wfile.flush()
                    except OSError:
                        gone = True

            try:
                for line in _lines(up.body):
                    s = line.strip()
                    if not s.startswith(b"data:"):
                        continue
                    payload = s[5:].strip()
                    if payload == b"[DONE]":
                        break
                    chunk = obj(json.loads(payload))
                    if chunk is not None:
                        emit(tr.feed(chunk))
                else:
                    raise OSError("upstream closed before [DONE]")
                emit(tr.finish())
            except (OSError, ValueError):
                pass  # upstream cut or garbled: settle on what we saw
            finally:
                up.close()
            seen = Usage(input=tr.prompt_tokens_seen or est_in)
            usage = acc.usage if acc.stopped and acc.saw_usage else _cutoff_usage(adm, seen, est_in)
            proxy.settle(adm, usage, acc.refused, acc.message())
            return True

    return Handler
