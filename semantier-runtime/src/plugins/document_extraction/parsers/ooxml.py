from __future__ import annotations
import hashlib,io,zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

W="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
def paragraph_text(node):
 parts=[]
 for child in node.iter():
  if child.tag==W+"t":parts.append(child.text or "")
  elif child.tag in {W+"br",W+"cr"}:parts.append("\n")
  elif child.tag==W+"tab":parts.append("\t")
 return "".join(parts)
def _sha(data:bytes)->str:return "sha256:"+hashlib.sha256(data).hexdigest()
def _anchor(*,body:int,kind:str,text:str,paragraph:int|None=None,table:int|None=None,row:int|None=None,cell:int|None=None):
 return {"document_schema_version":"canonical_document.v2","normalized_document_artifact_ref":None,"source_hash":None,"element_kind":kind,"body_index":body,"paragraph_index":paragraph,"table_index":table,"row_index":row,"cell_index":cell,"local_text_hash":_sha(text.encode())}
def parse_bound(path:str|Path,*,artifact_ref:str,source_sha256:str,media_type:str,byte_length:int,execution_scope_ref:str)->dict[str,Any]:
 p=Path(path);raw=p.read_bytes()
 if len(raw)!=byte_length or _sha(raw)!=source_sha256:raise ValueError("SOURCE_BINDING_MISMATCH")
 if p.suffix.lower()!=".docx":raise ValueError("UNSUPPORTED_FORMAT")
 return parse_bytes_bound(raw,artifact_ref=artifact_ref,source_sha256=source_sha256,media_type=media_type,byte_length=byte_length,execution_scope_ref=execution_scope_ref,source_name=p.name)
def parse_bytes_bound(raw:bytes,*,artifact_ref:str,source_sha256:str,media_type:str,byte_length:int,execution_scope_ref:str,source_name:str="source.docx")->dict[str,Any]:
 del execution_scope_ref
 if len(raw)!=byte_length or _sha(raw)!=source_sha256:raise ValueError("SOURCE_BINDING_MISMATCH")
 with zipfile.ZipFile(io.BytesIO(raw)) as z:root=ET.fromstring(z.read("word/document.xml"))
 body=root.find(".//"+W+"body");elements=[];blocks=[];tables=[];evidence=[];pi=ti=0
 if body is None:raise ValueError("DOCX_BODY_MISSING")
 paragraphs=list(body.iter(W+"p"));first_visible=next((i for i,p in enumerate(paragraphs) if paragraph_text(p).strip()),0)
 paragraph_order={id(p):index-first_visible for index,p in enumerate(paragraphs)}
 for bi,node in enumerate(list(body),1):
  if node.tag==W+"p":
   text=paragraph_text(node)
   if not text:continue
   pi+=1;eid=f"ooxml:p{pi}";a=_anchor(body=bi,kind="paragraph",text=text,paragraph=pi);a["render_paragraph_index"]=paragraph_order[id(node)]
   ev={"id":eid,"source":source_name,"locator":{"kind":"ooxml","part":"word/document.xml","body_index":bi,"paragraph":pi},"anchor":a,"provenance_class":"native"}
   element={"id":f"ooxml.body.{bi}.p{pi}","kind":"paragraph","body_index":bi,"paragraph_index":pi,"text":text,"anchor":a,"local_text_hash":a["local_text_hash"],"evidence_ids":[eid],"provenance_class":"native"}
   elements.append(element);blocks.append({"id":f"block_{pi}","kind":"paragraph","text":text,"evidence_ids":[eid],"element_id":element["id"],"local_text_hash":a["local_text_hash"],"provenance_class":"native"});evidence.append(ev)
  elif node.tag==W+"tbl":
   ti+=1;rows=[];ids=[];parts=[]
   for ri,tr in enumerate(node.findall(W+"tr"),1):
    cells=[]
    for ci,tc in enumerate(tr.findall(W+"tc"),1):
     paragraphs=[paragraph_text(q) for q in tc.findall(W+"p")];text="\n".join(paragraphs)
     eid=f"ooxml:t{ti}.r{ri}.c{ci}";a=_anchor(body=bi,kind="table_cell",text=text,table=ti,row=ri,cell=ci);structure={}
     gs=tc.find(W+"tcPr/"+W+"gridSpan");vm=tc.find(W+"tcPr/"+W+"vMerge")
     if gs is not None:structure["gridSpan"]=gs.get(W+"val")
     if vm is not None:structure["vMerge"]=vm.get(W+"val") or "continue"
     cellobj={"paragraphs":paragraphs,"row":ri,"column":ci,"text":text,"evidence_ids":[eid],"element_id":f"ooxml.body.{bi}.t{ti}.r{ri}.c{ci}","local_text_hash":a["local_text_hash"],"structure":structure,"provenance_class":"native"}
     cells.append(cellobj);ids.append(eid);parts.append(text);evidence.append({"id":eid,"source":source_name,"locator":{"kind":"ooxml","part":"word/document.xml","body_index":bi,"table":ti,"row":ri,"cell":ci},"anchor":a,"structure":structure,"provenance_class":"native"})
    rows.append({"row":ri,"cells":cells})
   text="\n".join(parts);ta=_anchor(body=bi,kind="table",text=text,table=ti);table={"id":f"ooxml.table.{ti}","kind":"docx_table","body_index":bi,"rows":rows,"evidence_ids":ids,"element_id":f"ooxml.body.{bi}.t{ti}","local_text_hash":ta["local_text_hash"],"provenance_class":"native"}
   tables.append(table);elements.append({**table,"kind":"table","table_index":ti,"text":text,"anchor":ta})
 return {"schema":"canonical_document.v2","schema_version":"canonical_document.v2","parser":{"method":"ooxml","version":"semantier.ooxml.v1"},"source":{"artifact_ref":artifact_ref,"source_sha256":source_sha256,"media_type":media_type,"byte_length":byte_length},"pages":[],"blocks":blocks,"elements":elements,"tables":tables,"key_value_candidates":[],"evidence":evidence,"provider_artifacts":[]}
def parse(path:str|Path,*,allow_ocr:bool=False,language_hints:list[str]|None=None):
 del allow_ocr,language_hints;p=Path(path);raw=p.read_bytes();return parse_bound(p,artifact_ref=f"local:{p.name}",source_sha256=_sha(raw),media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",byte_length=len(raw),execution_scope_ref="local")
