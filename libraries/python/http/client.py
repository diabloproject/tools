import json
import time
from typing import Any, Mapping, Optional, Union, Callable
from urllib import request, error, parse
from dataclasses import dataclass

@dataclass
class RequestHooks:
    on_request_before_send: Callable[["Request"], None] | None = None
    on_request_after_send: Callable[["Request"], None] | None = None
    on_response_before_receive: Callable[["Request", "Response"], None] | None = None
    on_response_after_headers: Callable[["Request", "Response"], None] | None = None
    on_response_chunk_received: Callable[["Request", "Response"], None] | None = None
    on_response_after_complete: Callable[["Request", "Response"], None] | None = None



class HTTPClientError(Exception):
    pass

class HTTPStatusError(HTTPClientError):
    def __init__(self, status: int, reason: str, body: bytes, headers: Mapping[str, str], url: str):
        super().__init__(f"{status} {reason} for URL: {url}")
        self.status = status
        self.reason = reason
        self.body = body
        self.headers = dict(headers)
        self.url = url

    def text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        return self.body.decode(encoding, errors)

    def json(self) -> Any:
        return json.loads(self.text())


class Response:
    content: bytes

    def __init__(self, status: int, headers: Mapping[str, str], content: bytes, url: str):
        self.status = status
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.content = content
        self.url = url

    @property
    def text(self) -> str:
        content_type = self.headers.get("content-type", "")
        charset = "utf-8"
        if "charset=" in content_type:
            try:
                charset = content_type.split("charset=")[1].split(";")[0].strip()
            except Exception:
                pass
        return self.content.decode(charset, errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class HttpClient:
    def __init__(
        self,
        user_agent: Optional[str] = None,
        max_redirects: int = 10,
        retries: int = 3,
        retry_backoff: float = 0.5,
        timeout: Optional[float] = None,
    ):
        self.user_agent = user_agent or "python-http-client/2.0"
        self.max_redirects = max_redirects
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self._opener = request.build_opener()

    def _build_url(self, url: str, params: Optional[Mapping[str, Any]]) -> str:
        if not params:
            return url
        parsed = parse.urlparse(url)
        existing = dict(parse.parse_qsl(parsed.query))
        merged = {**existing, **{k: str(v) for k, v in params.items()}}
        query = parse.urlencode(merged)
        return parse.urlunparse(parsed._replace(query=query))

    def _prepare_data_and_headers(
        self,
        data: Optional[Union[bytes, Mapping[str, Any]]],
        headers: Mapping[str, str],
        json_data: Optional[Any],
    ):
        # Choose between raw bytes or JSON body
        if json_data is not None:
            body = json.dumps(json_data).encode("utf-8")
            headers = {**headers, "Content-Type": "application/json"}
        elif isinstance(data, Mapping):
            body = parse.urlencode(data).encode("utf-8")
            headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}
        else:
            body = data
        return body, headers

    def _do_request(self, method: str, url: str, headers: Mapping[str, str], body: Optional[bytes]) -> Response:
        req = request.Request(url, data=body)
        try:
            req.method = method.upper()
        except Exception:
            req.get_method = lambda: method.upper()

        req.add_header("User-Agent", self.user_agent)
        for k, v in headers.items():
            req.add_header(k, v)

        try:
            resp = self._opener.open(req, timeout=self.timeout)
            status = getattr(resp, "getcode", lambda: resp.status)()
            body_response = resp.read() or b""
            hdrs = resp.headers
            if 400 <= status <= 599:
                raise HTTPStatusError(status, getattr(resp, "reason", ""), body_response, hdrs, url)
            return Response(status, dict(hdrs.items()), body_response, resp.geturl())
        except error.HTTPError as he:
            body = he.read() if hasattr(he, "read") else b""
            raise HTTPStatusError(he.code, he.reason or str(he.code), body, dict(he.headers.items()), url)
        except error.URLError as ue:
            raise HTTPClientError(f"Network error connecting to {url}: {ue}") from ue

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Union[bytes, Mapping[str, Any]]] = None,
        json: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> Response:
        url = self._build_url(url, params)
        hdrs = {**(headers or {})}
        body, hdrs = self._prepare_data_and_headers(data, hdrs, json)
        attempts = retries if retries is not None else self.retries
        delay = self.retry_backoff

        for attempt in range(attempts):
            try:
                return self._do_request(method, url, hdrs, body)
            except (HTTPClientError, HTTPStatusError) as e:
                if attempt < attempts - 1:
                    # Retry only on transient network errors or 5xx
                    if isinstance(e, HTTPStatusError) and not (500 <= e.status < 600):
                        raise
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise HTTPClientError("All attempts failed")

    def get(self, url: str, **kwargs) -> Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> Response:
        return self.request("DELETE", url, **kwargs)
