"""Bounded inert HTML text projection; preparation only, never a browser DOM.

Original token offsets are Unicode codepoints and byte intervals, not geometry.
Inline text is concatenated. Block boundaries insert LF, edge whitespace is
trimmed with Python Unicode whitespace semantics. No attributes are projected.
"""

import codecs
import hashlib
import html
from html.parser import HTMLParser
import json
import re

VERSION = "original-html-text/1"
BLOCKS = frozenset(
    "p div section article li ul ol h1 h2 h3 h4 h5 h6 tr td th br hr pre blockquote".split()
)
EXCLUDED = frozenset(
    "head script style template noscript iframe object embed svg math form button input select textarea".split()
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def decode(raw):
    if len(raw) > 8_000_000:
        raise ValueError("HTML_SOURCE_TOO_LARGE")
    labels = re.findall(
        rb"<meta\b[^>]*charset\s*=\s*[\x22\x27]?([a-zA-Z0-9_-]+)", raw[:4096], re.I
    )
    names = {codecs.lookup(label.decode("ascii")).name for label in labels}
    if len(names) > 1:
        raise ValueError("HTML_CHARSET_AMBIGUOUS")
    encoding = next(iter(names), "utf-8")
    if encoding not in {"utf-8", "gb18030", "gbk", "iso8859-1", "cp1252"}:
        raise ValueError("HTML_CHARSET_UNSUPPORTED")
    text = raw.decode(encoding, errors="strict")
    if text.encode(encoding) != raw:
        raise ValueError("HTML_DECODING_NOT_REVERSIBLE")
    return text, encoding


class _Projection(HTMLParser):
    def __init__(self, text, encoding):
        super().__init__(convert_charrefs=False)
        self.original, self.encoding = text, encoding
        self.lines = [0]
        for m in re.finditer("\n", text):
            self.lines.append(m.end())
        self.byte_offsets = [0]
        for c in text:
            self.byte_offsets.append(self.byte_offsets[-1] + len(c.encode(encoding)))
        self.excluded = []
        self.current, self.segments, self.removed = [], [], []
        self.token = 0

    def source_offset(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def flush(self):
        while self.current and not self.current[0]["text"].strip():
            self.current.pop(0)
        while self.current and not self.current[-1]["text"].strip():
            self.current.pop()
        if not self.current:
            return
        first, last = self.current[0], self.current[-1]
        trim = len(first["text"]) - len(first["text"].lstrip())
        first["decoded_start"] += trim
        first["text"] = first["text"].lstrip()
        last["text"] = last["text"].rstrip()
        last["decoded_end"] = last["decoded_start"] + len(last["text"])
        self.segments.extend(self.current)
        self.segments.append(
            {
                "kind": "separator",
                "text": "\n",
                "rule": "block-end-LF",
                "token": self.token,
            }
        )
        self.current = []

    def handle_starttag(self, tag, attrs):
        self.token += 1
        if self.excluded:
            if tag in EXCLUDED and tag not in {"input", "embed"}:
                self.excluded.append(tag)
            return
        if tag in EXCLUDED:
            self.removed.append({"tag": tag, "raw_start": self.source_offset()})
            if tag not in {"input", "embed"}:
                self.excluded.append(tag)
            return
        if tag in BLOCKS:
            self.flush()

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self.token += 1
        if self.excluded:
            if tag == self.excluded[-1]:
                self.excluded.pop()
            return
        if tag in BLOCKS:
            self.flush()

    def emit(self, raw_text):
        self.token += 1
        if self.excluded:
            return
        start = self.source_offset()
        end = start + len(raw_text)
        assert self.original[start:end] == raw_text
        text = html.unescape(raw_text).replace("\r\n", "\n").replace("\r", "\n")
        self.current.append(
            {
                "kind": "text",
                "token": self.token,
                "raw_start": start,
                "raw_end": end,
                "byte_start": self.byte_offsets[start],
                "byte_end": self.byte_offsets[end],
                "decoded_start": 0,
                "decoded_end": len(text),
                "text": text,
            }
        )

    def handle_data(self, data):
        self.emit(data)

    def handle_entityref(self, name):
        start = self.source_offset()
        length = len(name) + 1
        if self.original[start + length : start + length + 1] == ";":
            length += 1
        self.emit(self.original[start : start + length])

    def handle_charref(self, name):
        self.handle_entityref("#" + name)


def prepare_html(raw):
    try:
        text, encoding = decode(raw)
    except (UnicodeError, LookupError) as exc:
        raise ValueError("HTML_CHARSET_INVALID") from exc
    parser = _Projection(text, encoding)
    parser.feed(text)
    parser.close()
    parser.flush()
    position = 0
    for segment in parser.segments:
        segment["start"] = position
        position += len(segment["text"])
        segment["end"] = position
    packet = {
        "version": VERSION,
        "projection": "inert-text-segments/1",
        "raw_sha256": sha(raw),
        "charset": encoding,
        "offset_unit": "Unicode-codepoint",
        "text": "".join(s["text"] for s in parser.segments),
        "segments": parser.segments,
        "removed_subtrees": parser.removed,
    }
    packet["sha256"] = sha(canonical(packet))
    return packet


def verify_projection(raw, packet, expected_sha256=None):
    """Offline verify retained segments against raw slices, without parsing HTML."""
    try:
        value = dict(packet)
        claimed = value.pop("sha256")
        if sha(canonical(value)) != claimed or (
            expected_sha256 is not None and claimed != expected_sha256
        ):
            raise ValueError("HTML_PROJECTION_PIN_MISMATCH")
        if (
            packet["version"] != VERSION
            or packet["projection"] != "inert-text-segments/1"
            or sha(raw) != packet["raw_sha256"]
        ):
            raise ValueError("HTML_SOURCE_PIN_MISMATCH")
        original = raw.decode(packet["charset"], errors="strict")
        position, chunks, previous_raw_end = 0, [], 0
        for s in packet["segments"]:
            if s["kind"] == "text":
                a, b = s["raw_start"], s["raw_end"]
                if not 0 <= previous_raw_end <= a < b <= len(original):
                    raise ValueError("HTML_SEGMENT_ORDER")
                previous_raw_end = b
                fragment = original[a:b]
                if raw[s["byte_start"] : s["byte_end"]] != fragment.encode(
                    packet["charset"]
                ):
                    raise ValueError("HTML_BYTE_MAPPING_MISMATCH")
                decoded = (
                    html.unescape(fragment).replace("\r\n", "\n").replace("\r", "\n")
                )
                if not 0 <= s["decoded_start"] <= s["decoded_end"] <= len(decoded):
                    raise ValueError("HTML_DECODED_RANGE")
                expected = decoded[s["decoded_start"] : s["decoded_end"]]
            elif s["kind"] == "separator" and s["rule"] == "block-end-LF":
                expected = "\n"
            else:
                raise ValueError("HTML_SEGMENT_KIND")
            if (
                s["text"] != expected
                or s["start"] != position
                or s["end"] != position + len(expected)
            ):
                raise ValueError("HTML_SEGMENT_MISMATCH")
            position = s["end"]
            chunks.append(expected)
        if "".join(chunks) != packet["text"]:
            raise ValueError("HTML_TEXT_MISMATCH")
    except (KeyError, TypeError, UnicodeError, LookupError) as exc:
        raise ValueError("HTML_MAPPING_INVALID") from exc
    return True


def select_occurrence(packet, quote, *, start=None):
    if not quote:
        return {"state": "missing"}
    positions = [
        m.start() for m in re.finditer("(?=" + re.escape(quote) + ")", packet["text"])
    ]
    if start is not None:
        positions = [p for p in positions if p == start]
    if len(positions) != 1:
        return {"state": "ambiguous" if positions else "missing"}
    a = positions[0]
    b = a + len(quote)
    return {
        "state": "exact",
        "start": a,
        "end": b,
        "quote": quote,
        "quote_sha256": sha(quote.encode()),
        "projection_sha256": packet["sha256"],
        "segments": [s for s in packet["segments"] if s["start"] < b and s["end"] > a],
    }


def adapt_html_bytes(raw, *, source_identity_ref):
    from contracts.canonical_source_ir import CanonicalSourceDocument, content_hash

    packet = prepare_html(raw)
    return CanonicalSourceDocument.model_validate(
        {
            "sourceIdentityRef": source_identity_ref,
            "blocks": [
                {
                    "blockId": "html:" + packet["sha256"],
                    "blockType": "paragraph",
                    "content": packet["text"],
                    "ordinal": 0,
                    "contentHash": content_hash(packet["text"]),
                }
            ],
            "sourceMap": {"mappings": [{
                "sourceLocators": [{"sourceIdentityRef": source_identity_ref,
                    "sourceHash": "sha256:" + sha(raw), "mediaType": "text/html",
                    "nativeProvider": "python.html.parser", "format": "html",
                    "nativeId": "html:" + packet["sha256"], "ordinalPath": ["html", "text"],
                    "nativeMetadata": {"projection_hash": packet["sha256"]}}],
                "blockIds": ["html:" + packet["sha256"]],
                "transformation": {"kind": "structural_normalization", "transformRef": VERSION}
            }]},
            "assets": [{"kind": "original_html_projection", "packet": packet}],
            "parserTrace": {
                "provider": "python.html.parser",
                "adapterVersion": "semantier.canonical_source_ir.native.html.v2",
                "canonicalIrVersion": "canonical_source_ir.v1",
                "warnings": [
                    "Inert text projection; styles/attributes/active subtrees excluded; not pixel-identical."
                ],
            },
        }
    )
