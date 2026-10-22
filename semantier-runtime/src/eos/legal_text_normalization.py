"""Deterministic normalization, anchoring, and diff helpers for legal text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from eos.artifact_hashing import content_hash


_ARTICLE_RE = re.compile(r"^第([一二三四五六七八九十百千万零〇两\d]+)条(?:\s*(.*))?$")
_ITEM_RE = re.compile(r"^[（(]([一二三四五六七八九十百千万零〇两\d]+)[）)]")


@dataclass(frozen=True)
class LegalSectionAnchor:
    stable_locator: str
    section_type: str
    ordinal: str
    title: str | None
    text: str
    text_hash: str
    local_hash: str
    anchor: dict[str, Any]


@dataclass(frozen=True)
class NormalizedPdfPage:
    page_number: int
    text: str
    text_hash: str
    coordinate_refs: list[dict[str, Any]]
    extraction_method: str
    probabilistic: bool
    requires_review: bool


def normalize_legal_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t\f\v \u00a0]+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def normalize_legal_html(html: str) -> str:
    parser = _LegalTextHTMLParser()
    parser.feed(str(html or ""))
    parser.close()
    return normalize_legal_text("".join(parser.parts))


def normalize_authoritative_pdf_pages(
    pages: list[dict[str, Any]],
    *,
    ocr_used: bool = False,
) -> tuple[str, list[NormalizedPdfPage]]:
    if ocr_used:
        return normalize_ocr_pdf_pages(pages)
    normalized_pages: list[NormalizedPdfPage] = []
    text_parts: list[str] = []
    for index, page in enumerate(pages, start=1):
        raw_text = str(page.get("text") or "")
        text = normalize_legal_text(raw_text)
        if not text:
            raise ValueError("PDF page text layer is required unless OCR is explicitly used")
        coordinate_refs = _normalize_coordinate_refs(page.get("coordinates") or [])
        text_parts.append(text)
        normalized_pages.append(
            NormalizedPdfPage(
                page_number=int(page.get("page_number") or index),
                text=text,
                text_hash=content_hash({"text": text}),
                coordinate_refs=coordinate_refs,
                extraction_method="pdf_text_layer",
                probabilistic=False,
                requires_review=False,
            )
        )
    return normalize_legal_text("\n".join(text_parts)), normalized_pages


def normalize_ocr_pdf_pages(pages: list[dict[str, Any]]) -> tuple[str, list[NormalizedPdfPage]]:
    normalized_pages: list[NormalizedPdfPage] = []
    text_parts: list[str] = []
    for index, page in enumerate(pages, start=1):
        text = normalize_legal_text(str(page.get("ocr_text") or page.get("text") or ""))
        if not text:
            raise ValueError("OCR text is required when no PDF text layer exists")
        coordinate_refs = _normalize_coordinate_refs(page.get("coordinates") or [])
        text_parts.append(text)
        normalized_pages.append(
            NormalizedPdfPage(
                page_number=int(page.get("page_number") or index),
                text=text,
                text_hash=content_hash({"text": text}),
                coordinate_refs=coordinate_refs,
                extraction_method="ocr_probabilistic",
                probabilistic=True,
                requires_review=True,
            )
        )
    return normalize_legal_text("\n".join(text_parts)), normalized_pages


def generate_legal_section_anchors(text: str) -> list[LegalSectionAnchor]:
    normalized = normalize_legal_text(text)
    if not normalized:
        return []
    lines = normalized.splitlines()
    sections: list[LegalSectionAnchor] = []
    current_article: dict[str, Any] | None = None
    current_paragraphs: list[str] = []
    article_index = 0

    def flush_article() -> None:
        nonlocal current_article, current_paragraphs
        if current_article is None:
            return
        article_locator = current_article["locator"]
        article_text = "\n".join([current_article["heading"], *current_paragraphs]).strip()
        sections.append(
            _section(
                stable_locator=article_locator,
                section_type="article",
                ordinal=current_article["ordinal"],
                title=current_article["title"],
                text=article_text,
                anchor={
                    "article_index": current_article["index"],
                    "article_marker": current_article["marker"],
                },
            )
        )
        for paragraph_index, paragraph in enumerate(current_paragraphs, start=1):
            paragraph_locator = f"{article_locator}.paragraph_{paragraph_index:03d}"
            sections.append(
                _section(
                    stable_locator=paragraph_locator,
                    section_type="paragraph",
                    ordinal=str(paragraph_index),
                    title=None,
                    text=paragraph,
                    anchor={
                        "article_index": current_article["index"],
                        "paragraph_index": paragraph_index,
                    },
                )
            )
            item_match = _ITEM_RE.match(paragraph)
            if item_match:
                item_locator = f"{paragraph_locator}.item_001"
                sections.append(
                    _section(
                        stable_locator=item_locator,
                        section_type="item",
                        ordinal=item_match.group(1),
                        title=None,
                        text=paragraph,
                        anchor={
                            "article_index": current_article["index"],
                            "paragraph_index": paragraph_index,
                            "item_index": 1,
                            "item_marker": item_match.group(1),
                        },
                    )
                )
        current_article = None
        current_paragraphs = []

    preamble: list[str] = []
    for line in lines:
        article_match = _ARTICLE_RE.match(line)
        if article_match:
            flush_article()
            article_index += 1
            marker = article_match.group(1)
            article_body = article_match.group(2) or None
            current_article = {
                "index": article_index,
                "marker": marker,
                "ordinal": marker,
                "locator": f"article_{article_index:03d}",
                "title": None,
                "heading": f"第{marker}条",
            }
            current_paragraphs = [article_body] if article_body else []
            continue
        if current_article is None:
            preamble.append(line)
        else:
            current_paragraphs.append(line)
    flush_article()

    if preamble:
        preamble_text = "\n".join(preamble)
        sections.insert(
            0,
            _section(
                stable_locator="preamble",
                section_type="preamble",
                ordinal="0",
                title=None,
                text=preamble_text,
                anchor={"line_start": 1, "line_end": len(preamble)},
            ),
        )
    return sections


def diff_legal_sections(
    before: list[LegalSectionAnchor],
    after: list[LegalSectionAnchor],
) -> dict[str, list[dict[str, Any]]]:
    before_by_locator = {section.stable_locator: section for section in before}
    after_by_locator = {section.stable_locator: section for section in after}
    added = [
        _diff_row(section)
        for locator, section in sorted(after_by_locator.items())
        if locator not in before_by_locator
    ]
    removed = [
        _diff_row(section)
        for locator, section in sorted(before_by_locator.items())
        if locator not in after_by_locator
    ]
    modified = [
        {
            "stable_locator": locator,
            "before_hash": before_by_locator[locator].local_hash,
            "after_hash": after_by_locator[locator].local_hash,
            "section_type": after_by_locator[locator].section_type,
        }
        for locator in sorted(set(before_by_locator) & set(after_by_locator))
        if before_by_locator[locator].local_hash != after_by_locator[locator].local_hash
    ]
    return {"added": added, "removed": removed, "modified": modified}


def _section(
    *,
    stable_locator: str,
    section_type: str,
    ordinal: str,
    title: str | None,
    text: str,
    anchor: dict[str, Any],
) -> LegalSectionAnchor:
    text_hash = content_hash({"text": text})
    local_hash = content_hash(
        {
            "stable_locator": stable_locator,
            "section_type": section_type,
            "ordinal": ordinal,
            "title": title,
            "text_hash": text_hash,
        }
    )
    return LegalSectionAnchor(
        stable_locator=stable_locator,
        section_type=section_type,
        ordinal=ordinal,
        title=title,
        text=text,
        text_hash=text_hash,
        local_hash=local_hash,
        anchor=anchor,
    )


def _diff_row(section: LegalSectionAnchor) -> dict[str, Any]:
    return {
        "stable_locator": section.stable_locator,
        "section_type": section.section_type,
        "local_hash": section.local_hash,
    }


def _normalize_coordinate_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("PDF coordinate refs must be a list")
    refs: list[dict[str, Any]] = []
    for ref in value:
        if not isinstance(ref, dict):
            raise ValueError("PDF coordinate ref must be an object")
        normalized = {
            "locator": str(ref.get("locator") or "").strip(),
            "x0": float(ref.get("x0", 0)),
            "y0": float(ref.get("y0", 0)),
            "x1": float(ref.get("x1", 0)),
            "y1": float(ref.get("y1", 0)),
        }
        if not normalized["locator"]:
            raise ValueError("PDF coordinate ref locator is required")
        refs.append(normalized)
    return refs


class _LegalTextHTMLParser(HTMLParser):
    _BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "li",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        # Inline elements are one text flow; only block tags introduce lines.
        # Collapse HTML formatting whitespace without removing word separators.
        self.parts.append(re.sub(r"[\t\r\n\f ]+", " ", data))
