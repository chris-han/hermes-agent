from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
def _sha(data:bytes)->str:return "sha256:"+hashlib.sha256(data).hexdigest()
def parse_bound(path:str|Path,*,artifact_ref:str,source_sha256:str,media_type:str,byte_length:int,execution_scope_ref:str)->dict[str,Any]:
 return parse_bytes_bound(Path(path).read_bytes(),artifact_ref=artifact_ref,source_sha256=source_sha256,media_type=media_type,byte_length=byte_length,execution_scope_ref=execution_scope_ref)
def parse_bytes_bound(raw:bytes,*,artifact_ref:str,source_sha256:str,media_type:str,byte_length:int,execution_scope_ref:str)->dict[str,Any]:
 del execution_scope_ref
 import pymupdf
 if len(raw)!=byte_length or _sha(raw)!=source_sha256:raise ValueError("SOURCE_BINDING_MISMATCH")
 pages=[];blocks=[];elements=[];evidence=[];idx=0
 with pymupdf.open(stream=raw,filetype="pdf") as doc:
  for pn,page in enumerate(doc,1):
   rect=page.rect;pages.append({"page":pn,"width":float(rect.width),"height":float(rect.height)})
   rawd=page.get_text("rawdict",sort=False)
   for bi,b in enumerate(rawd.get("blocks",[]),1):
    if b.get("type")!=0:continue
    for li,line in enumerate(b.get("lines",[]),1):
     text="".join("".join(c.get("c","") for c in s.get("chars",[])) for s in line.get("spans",[]))
     if not text.strip():continue
     idx+=1;eid=f"pymupdf:p{pn}:b{bi}:l{li}";bbox=[float(x) for x in line["bbox"]];th=_sha(text.encode())
     evidence.append({"id":eid,"kind":"text","text":text,"text_sha256":th,"locator":{"kind":"pymupdf_rawdict","page":pn,"bbox":bbox,"provider_object_ref":eid,"provenance_class":"authored"}})
     chars=[{"text":c.get("c", ""),"bbox":[float(x) for x in c["bbox"]]} for span in line.get("spans", []) for c in span.get("chars", [])]
     item={"chars":chars,"id":eid,"type":"paragraph","kind":"paragraph","text":text,"page":pn,"bbox":bbox,"evidence_ids":[eid],"local_text_hash":th,"provenance_class":"authored"}
     blocks.append(item);elements.append(item)
 return {"schema":"canonical_document.v2","schema_version":"canonical_document.v2","parser":{"method":"pymupdf","version":pymupdf.__version__},"source":{"artifact_ref":artifact_ref,"source_sha256":source_sha256,"media_type":media_type,"byte_length":byte_length},"pages":pages,"blocks":blocks,"elements":elements,"tables":[],"key_value_candidates":[],"evidence":evidence,"provider_artifacts":[]}
def parse(path:str|Path,*,allow_ocr:bool=False,language_hints:list[str]|None=None):
 if allow_ocr: raise ValueError("OCR_NOT_QUALIFIED")
 del language_hints;p=Path(path);raw=p.read_bytes();return parse_bound(p,artifact_ref=f"local:{p.name}",source_sha256=_sha(raw),media_type="application/pdf",byte_length=len(raw),execution_scope_ref="local")
