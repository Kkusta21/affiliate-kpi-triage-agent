"""A real, read-only external integration: probe a landing page over HTTP.

This is the one place the agent talks to the outside world for real. Everything else
(tracker, offer API, traffic data) is still a fixture, and the README says so.

Select it with TRIAGE_PAGE_PROBE=http and TRIAGE_PAGE_ALLOWED_HOSTS=host1,host2.
The default stays the fixture probe, so the offline demo and the tests need no network.

Safety properties, each covered by a test with httpx.MockTransport:
- Only http and https URLs whose host is on the allowlist are requested.
- A host that resolves to a private, loopback or link-local address is refused (SSRF).
- DNS is resolved once per hop and the connection goes to that vetted address (DNS pinning),
  so a rebinding DNS server cannot answer "public" to the check and "private" to the connect.
  TLS still verifies the certificate against the host NAME (see `_send`).
- No proxy settings are taken from the environment.
- Redirects are followed by hand, at most `max_redirects`, and every hop is checked again.
- Connect and read timeouts. The body is read up to `max_bytes` and then cut.
- HTTP 429 and 503 with Retry-After become a retryable error that carries the wait time.
- The response is validated and reduced to four facts. The body never leaves this module.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .datasource import PermanentToolError, TransientToolError

Resolver = Callable[[str], list[str]]
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
USER_AGENT = "affiliate-kpi-triage-agent/0.1 (landing page probe)"


class RateLimitedError(TransientToolError):
    """The remote side asked us to slow down. `retry_after` is in seconds."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def resolve_host(host: str) -> list[str]:
    """Default resolver: every address the host name maps to."""
    try:
        return sorted({str(info[4][0]) for info in socket.getaddrinfo(host, None)})
    except socket.gaierror as exc:
        raise TransientToolError(f"DNS lookup failed for {host}: {exc}") from exc


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


@dataclass(frozen=True)
class PinnedTarget:
    """A vetted URL plus the one address it resolved to."""

    url: str
    scheme: str
    host: str
    port: int | None
    ip: str
    path: str
    query: str

    def host_header(self) -> str:
        return self.host if self.port is None else f"{self.host}:{self.port}"

    def pinned_url(self) -> str:
        ip = f"[{self.ip}]" if ":" in self.ip else self.ip
        port = f":{self.port}" if self.port is not None else ""
        return urlunsplit((self.scheme, f"{ip}{port}", self.path, self.query, ""))


def _retry_after(response: httpx.Response, default: float = 2.0) -> float:
    raw = response.headers.get("retry-after", "").strip()
    return float(raw) if raw.isdigit() else default


class HttpLandingPageProbe:
    def __init__(
        self,
        allowed_hosts: tuple[str, ...],
        timeout_seconds: float = 5.0,
        max_bytes: int = 512_000,
        pixel_marker: str = "data-tracking-pixel",
        max_redirects: int = 3,
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver = resolve_host,
    ) -> None:
        self._allowed_hosts = {h.lower() for h in allowed_hosts}
        self._max_bytes = max_bytes
        self._pixel_marker = pixel_marker.lower()
        self._max_redirects = max_redirects
        self._resolver = resolver
        # transport is injected in tests (httpx.MockTransport). follow_redirects stays off on purpose.
        # trust_env=False: no proxy from the environment, so the connection really goes to the pinned IP.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            transport=transport,
            headers={"User-Agent": USER_AGENT},
            trust_env=False,
        )

    def _vet(self, url: str) -> PinnedTarget:
        """Check the URL and resolve its host ONCE. Returns the address the request must use."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme not in ("http", "https") or not host:
            raise PermanentToolError(f"not an http(s) URL: {url[:120]}")
        if parts.username or parts.password:
            raise PermanentToolError("URLs with credentials are not probed")
        if host not in self._allowed_hosts:
            raise PermanentToolError(f"host is not on the probe allowlist: {host}")
        addresses = self._resolver(host)
        if not addresses or not all(_is_public(address) for address in addresses):
            raise PermanentToolError(f"host resolves to a non-public address and is not probed: {host}")
        return PinnedTarget(
            url=url,
            scheme=parts.scheme,
            host=host,
            port=parts.port,
            ip=addresses[0],
            path=parts.path or "/",
            query=parts.query,
        )

    def _send(self, target: PinnedTarget) -> httpx.Response:
        """Connect to the vetted IP, not to the name. This closes the DNS rebinding gap: the
        address that was checked is the address that is used, because nothing resolves again.

        The Host header carries the real name (virtual hosting), and the `sni_hostname`
        extension makes httpcore use the real name for TLS SNI AND for certificate checking
        (it is passed as `server_hostname`, and the default SSL context verifies hostnames).
        """
        request = self._client.build_request(
            "GET", target.pinned_url(), headers={"Host": target.host_header()}, extensions={"sni_hostname": target.host}
        )
        return self._client.send(request, stream=True)

    def fetch(self, url: str) -> dict[str, Any]:
        """Return {"status_code", "latency_ms", "title", "has_tracking_pixel", "truncated"}.

        Raises PermanentToolError for a URL that must not be probed, TransientToolError for
        timeouts and connection errors, and RateLimitedError when the site asks us to wait.
        A 4xx or 5xx page is a valid answer about the page and is returned, not raised.
        """
        started = time.perf_counter()
        for _ in range(self._max_redirects + 1):
            target = self._vet(url)
            try:
                response = self._send(target)
                try:
                    status = response.status_code
                    if status in (301, 302, 303, 307, 308) and "location" in response.headers:
                        url = urljoin(url, response.headers["location"])  # relative to the NAME, then vetted again
                        continue
                    if status == 429 or (status == 503 and "retry-after" in response.headers):
                        raise RateLimitedError(f"probe was rate limited (HTTP {status})", _retry_after(response))
                    body, truncated = b"", False
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) >= self._max_bytes:
                            body, truncated = body[: self._max_bytes], True
                            break
                finally:
                    response.close()
            except httpx.TimeoutException as exc:
                raise TransientToolError(f"probe timeout: {type(exc).__name__}") from exc
            except httpx.TransportError as exc:
                if "CERTIFICATE_VERIFY_FAILED" in str(exc):  # a bad certificate does not fix itself on retry
                    raise PermanentToolError("TLS certificate verification failed for the probed host") from exc
                raise TransientToolError(f"probe connection error: {type(exc).__name__}") from exc
            text = body.decode("utf-8", errors="replace")
            title = _TITLE_RE.search(text)
            return {
                "status_code": status,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "title": title.group(1).strip()[:200] if title else "",
                "has_tracking_pixel": self._pixel_marker in text.lower(),
                "truncated": truncated,
            }
        raise PermanentToolError(f"more than {self._max_redirects} redirects")

    def close(self) -> None:
        self._client.close()
