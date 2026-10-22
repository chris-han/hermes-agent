"""Canonical source-element projection over the native-evidence CanonicalSourceIR boundary."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from contracts.knowledge_feature_pack import content_hash
from contracts.knowledge_graph_phase1 import SourceAnchor
from semantier.knowledge_feature_packs.canonical_source_ir_native_adapter import adapt_docx_bytes
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import canonical_elements_from_ir

FAMILY_HEADING = re.compile(r"^\s*(1[0-5]|[1-9])[.、．]\s*(.+?)\s*$")


def document_hash(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_elements(path: str | Path) -> dict[str, Any]:
    p=Path(path); raw=p.read_bytes(); digest="sha256:"+hashlib.sha256(raw).hexdigest()
    document=adapt_docx_bytes(raw,source_identity_ref=f"source-file:{digest}")
    projected=canonical_elements_from_ir(document)
    elements=[];family_id=None;family="";paragraph_index=0;cell_index=0
    for item in projected:
        text=str(item["text"])
        match=FAMILY_HEADING.match(text)
        if match:
            family_id=int(match.group(1));family=match.group(2)
        entry=dict(item);entry["family_id"]=family_id;entry["family"]=family
        anchor=dict(entry["anchor"]);anchor["document_hash"]=digest
        if anchor.get("document_part") == "paragraph" and anchor.get("paragraph_index") is None:
            anchor["paragraph_index"] = paragraph_index; paragraph_index += 1
        elif anchor.get("document_part") == "table_cell":
            if anchor.get("table_index") is None: anchor["table_index"] = 0
            if anchor.get("row_index") is None: anchor["row_index"] = cell_index
            if anchor.get("cell_index") is None: anchor["cell_index"] = 0
            cell_index += 1
        entry["anchor"]=anchor
        elements.append(entry)
    return {"source_path":p.as_posix(),"document_hash":digest,"elements":elements,"element_manifest_hash":content_hash(elements)}


def anchor_span(element: dict[str, Any], start: int, end: int) -> SourceAnchor:
    text=str(element["text"])
    if start < 0 or end > len(text) or end <= start: raise ValueError("span is outside the document element")
    anchor=dict(element["anchor"]);anchor.update({"anchor_id":f"{anchor['anchor_id']}:span:{start}-{end}","start_offset":start,"end_offset":end,"exact_text":text[start:end],"local_text_hash":content_hash(text[start:end])})
    return SourceAnchor.model_validate(anchor)
