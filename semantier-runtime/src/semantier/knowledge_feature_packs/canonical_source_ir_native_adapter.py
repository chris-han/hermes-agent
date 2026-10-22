from __future__ import annotations
import hashlib,io,zipfile
from pathlib import Path
from typing import Any
from contracts.canonical_source_ir import CanonicalSourceDocument,ParserTrace,SourceBlock,SourceLocator,SourceMap,SourceMapping,Transformation,content_hash,source_block_identity
from plugins.document_extraction.parsers import ooxml,pymupdf_native
DOCX="application/vnd.openxmlformats-officedocument.wordprocessingml.document";PDF="application/pdf"
def _sha(d:bytes):return "sha256:"+hashlib.sha256(d).hexdigest()
def adapt_source_bytes(data:bytes,*,source_identity_ref:str,media_type:str,adapter_version:str="semantier.canonical_source_ir.native.v1")->CanonicalSourceDocument:
 if media_type not in {DOCX,PDF}:raise ValueError("UNSUPPORTED_SOURCE_MEDIA_TYPE")
 digest=_sha(data);blocks=[];maps=[];assets=[]
 parser=ooxml if media_type==DOCX else pymupdf_native
 parsed=parser.parse_bytes_bound(data,artifact_ref=source_identity_ref,source_sha256=digest,media_type=media_type,byte_length=len(data),execution_scope_ref="canonical-source-ir")
 if media_type==DOCX:
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   for index,name in enumerate(sorted(n for n in z.namelist() if n.startswith("word/media/") and not n.endswith("/")),1):
    ext=Path(name).suffix.lower();mt={".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".gif":"image/gif",".svg":"image/svg+xml",".emf":"image/emf",".wmf":"image/wmf"}.get(ext,"application/octet-stream")
    assets.append({"packagePart":name,"mediaType":mt,"assetId":f"asset{index}"})
 def add(text,role,block_type,locator,parent=None,structure=None):
  bid=source_block_identity(digest,locator,role);blocks.append(SourceBlock(block_id=bid,block_type=block_type,content=text,parent_block_id=parent,ordinal=len(blocks),structure=structure or {},content_hash=content_hash(text)));maps.append(SourceMapping(source_locators=[locator],block_ids=[bid],transformation=Transformation(kind="one_to_one",transform_ref=locator.native_id or bid)));return bid
 if media_type==DOCX:
  for e in parsed["elements"]:
   bi=int(e["body_index"])
   if e["kind"]=="paragraph":
    loc=SourceLocator(source_identity_ref=source_identity_ref,source_hash=digest,media_type=media_type,native_provider="ooxml",format="docx",package_part="word/document.xml",native_id=e["id"],ordinal_path=[bi],paragraph_id=f"p{e['paragraph_index']}")
    add(e["text"],"paragraph","paragraph",loc,structure={"bodyIndex":bi})
   elif e["kind"]=="table":
    t=e["table_index"];tloc=SourceLocator(source_identity_ref=source_identity_ref,source_hash=digest,media_type=media_type,native_provider="ooxml",format="docx",package_part="word/document.xml",native_id=e["id"],ordinal_path=[bi,"table",t],table_id=f"t{t}")
    tbid=add({"rows":[]},"table","table",tloc,structure={"bodyIndex":bi,"tableOrdinal":t})
    for row in e["rows"]:
     for c in row["cells"]:
      loc=SourceLocator(source_identity_ref=source_identity_ref,source_hash=digest,media_type=media_type,native_provider="ooxml",format="docx",package_part="word/document.xml",native_id=c["element_id"],ordinal_path=[bi,"table",t,"row",c["row"],"cell",c["column"]],native_metadata=c.get("structure") or {},table_id=f"t{t}",row_id=f"r{c['row']}",column_id=f"c{c['column']}")
      add(c["text"],"cell","cell",loc,parent=tbid,structure={"tableOrdinal":t,"rowOrdinal":c["row"],"columnOrdinal":c["column"],**(c.get("structure") or {})})
  provider="ooxml";version="native"
 else:
  bypage={}
  for e in parsed["elements"]:bypage.setdefault(int(e["page"]),[]).append(e)
  for page,items in sorted(bypage.items()):
   # page aggregation is deterministic presentation normalization; line bbox provenance remains in native metadata.
   items=sorted(items,key=lambda x:(round(x["bbox"][1],3),round(x["bbox"][0],3),x["id"]))
   text="\n".join(x["text"] for x in items);bbox=[min(x["bbox"][0] for x in items),min(x["bbox"][1] for x in items),max(x["bbox"][2] for x in items),max(x["bbox"][3] for x in items)]
   metadata={"page":page,"bbox":bbox,"lineIds":[x["id"] for x in items]}
   if adapter_version=="semantier.canonical_source_ir.native.pdf.v2":
    offset=0;spans=[]
    for item in items:
     spans.append({"start_offset":offset,"end_offset":offset+len(item["text"]),"bbox":item["bbox"]})
     offset+=len(item["text"])+1
    metadata.update(kind="pdf_bbox",text_spans=spans)
   loc=SourceLocator(source_identity_ref=source_identity_ref,source_hash=digest,media_type=media_type,native_provider="pymupdf",format="pdf",native_id=f"pymupdf:page:{page}",ordinal_path=["page",page],native_metadata=metadata,paragraph_id=f"page:{page}")
   add(text,"paragraph","paragraph",loc,structure={"page":page,"bbox":bbox,"structureProvenance":"observation"})
  provider="pymupdf";version=parsed["parser"]["version"]
 if not blocks:raise ValueError("SOURCE_TEXT_UNAVAILABLE")
 return CanonicalSourceDocument(source_identity_ref=source_identity_ref,blocks=blocks,source_map=SourceMap(mappings=maps),assets=assets,parser_trace=ParserTrace(provider=provider,upstream_revision=None,upstream_package_version=version,semantier_patch_series_version=None,adapter_version=adapter_version,canonical_ir_version="canonical_source_ir.v1",warnings=[]))
def adapt_docx_bytes(data:bytes,*,source_identity_ref:str,media_type:str=DOCX,adapter_version:str="semantier.canonical_source_ir.native.v1",**_:Any):return adapt_source_bytes(data,source_identity_ref=source_identity_ref,media_type=media_type,adapter_version=adapter_version)
def adapt_pdf_bytes(data:bytes,*,source_identity_ref:str,media_type:str=PDF,adapter_version:str="semantier.canonical_source_ir.native.pdf.v2"):return adapt_source_bytes(data,source_identity_ref=source_identity_ref,media_type=media_type,adapter_version=adapter_version)
