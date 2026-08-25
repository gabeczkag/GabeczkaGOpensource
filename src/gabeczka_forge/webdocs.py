import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.ignored_depth += 1
        elif self.ignored_depth == 0 and tag.lower() in {"p", "div", "li", "pre", "h1", "h2", "h3", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif self.ignored_depth == 0 and tag.lower() in {"p", "div", "li", "pre", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth == 0:
            self.parts.append(data)


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use http:// or https://")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as error:
        raise ValueError(f"Cannot resolve host: {parsed.hostname}") from error
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Local and private network URLs are not allowed")


def fetch_html_text(url: str, max_bytes: int = 1_000_000, timeout: int = 15) -> str:
    _validate_url(url)
    request = Request(url, headers={"User-Agent": "GabeczkaForge/0.1 documentation reader"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError(f"Expected HTML, received {content_type}")
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"HTML document exceeds the {max_bytes:,}-byte limit")
    parser = _TextExtractor()
    parser.feed(content.decode("utf-8", errors="ignore"))
    text = " ".join(" ".join(parser.parts).split())
    if not text:
        raise ValueError("The HTML page contains no readable text")
    return text
