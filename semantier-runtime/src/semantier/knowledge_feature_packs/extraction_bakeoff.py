"""Replay recorded provider objects through the production Phase 1 extraction seam."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

from contracts.knowledge_feature_pack import MachineProposal, content_hash
from knowledge_extraction.adapters.langextract_adapter import LangExtractAdapter
from knowledge_extraction.assertion_grouper import group_occurrences
from knowledge_extraction.contracts import SourceExtractionRequest
from semantier.knowledge_feature_packs.benchmark import run_deterministic_baseline
from semantier.knowledge_feature_packs.source_elements import source_elements
from plugins.document_extraction.parsers.pymupdf_native import parse as parse_pdf

METRIC_NAMES=("candidate_recall","candidate_precision","source_alignment_accuracy","duplicate_assertion_rate","condition_extraction_accuracy","exception_extraction_accuracy","hard_negative_proposal_rate","family_concept_grounding_quality","human_accept_rate","human_edit_rate","human_reject_rate","reviewer_minutes_per_accepted_assertion","extraction_runtime_ms","extraction_cost_usd","replay_provenance_completeness")

def sha256_file(path:str|Path)->str:return "sha256:"+hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _ratio(n:int,d:int)->dict[str,Any]:return {"value":n/d if d else None,"numerator":n,"denominator":d}
def _cases(root:Path,paths:list[str])->list[dict]:
    out=[]
    for p in paths:
        doc=json.loads((root/p).read_text(encoding="utf-8"));out += [{**c,"smoke_fixture_id":doc["fixture_id"]} for c in doc["cases"]]
    return sorted(out,key=lambda c:c["case_id"])
def _source_path(case:dict)->str:
    base="docs/招投标法规/test-cases/poc测试文件-ground/"
    if case["smoke_fixture_id"]=="UAT-smoke-001":return base+"货物/裸光专线与互联网专线.pdf"
    if case["smoke_fixture_id"]=="UAT-smoke-002":return base+"服务/2025-2027年公司内含价值审阅机构选聘项目.pdf"
    name=case.get("source_document")
    return base+"工程/"+name
def _norm(text:str)->str:return "".join(text.split())
def _verify_sources(root:Path,cases:list[dict],source_hashes:dict,extracted_override:dict|None=None)->list[dict]:
    extracted={} if extracted_override is None else extracted_override
    for relative in source_hashes:
        if relative in extracted:continue
        path=root/relative
        if path.suffix.lower()==".docx":
            parsed=source_elements(path);extracted[relative]=[(None,e["text"],e["anchor"]["document_part"]) for e in parsed["elements"]]
        else:
            parsed=parse_pdf(path,allow_ocr=False,language_hints=["zh"]);extracted[relative]=[(b.get("page"),b["text"],"pdf_block") for b in parsed["blocks"]]
    out=[]
    for c in cases:
        path=_source_path(c);needle=_norm(c["text"]);matches=[x for x in extracted[path] if needle in _norm(x[1]) and (Path(path).suffix.lower()==".docx" or x[0]==c["source_page"])]
        out.append({"case_id":c["case_id"],"source_path":path,"source_hash":source_hashes[path],"excerpt_hash":content_hash(c["text"]),"expected_page":c["source_page"],"verification_state":"verified" if matches else "unverified","verified_page":matches[0][0] if matches else None,"verified_part":matches[0][2] if matches else None,"unverified_reason":None if matches else "local document parser produced no normalized exact excerpt match"})
    return out

class _I:
    def __init__(self,s,e):self.start_pos=s;self.end_pos=e
class _E:
    def __init__(self,d,text):
        self.extraction_text=d["extraction_text"];self.extraction_class=d["extraction_class"];self.attributes=d["attributes"]
        interval=d["char_interval"];s=int(interval["start"]);e=int(interval["end"])
        if text[s:e]!=self.extraction_text:raise ValueError("recorded extraction interval does not reconstruct extraction_text")
        self.char_interval=_I(s,e);self.alignment_status=d["alignment_status"]
        self.group_index=d["group_index"];self.extraction_index=d["extraction_index"]
class _R:
    warnings=()
    def __init__(self,items,text):self.extractions=[_E(x,text) for x in items]

def _replay(case:dict,capture:dict,root:Path,metadata:dict|None=None)->tuple:
    metadata=metadata or {"provider_version":"langextract-1.6.0-recorded","model_ref":"recorded-provider-output:not-live","profile_ref":"tender_sensitive_v1","config":{"temperature":0}}
    artifacts={}
    adapter=LangExtractAdapter(artifact_writer=lambda raw,digest:artifacts.setdefault(digest,raw) and f"recorded:{digest}",extract_fn=lambda **_:_R(capture["extractions"],case["text"]),provider_version=metadata["provider_version"])
    req=SourceExtractionRequest(extraction_run_id=f"recorded::{case['case_id']}",document_id=case.get("source_document") or case["smoke_fixture_id"],canonical_document_ref=f"smoke:{case['case_id']}",canonical_document_hash=content_hash(case["text"]),text=case["text"],extraction_profile=metadata["profile_ref"],provider_options={"model_ref":metadata["model_ref"],"langextract_options":metadata["config"]})
    result=adapter.extract(req)
    assertions=group_occurrences(result.occurrences,candidate_graph_id=f"candidate::{case['case_id']}",source_document_hash=req.canonical_document_hash,machine_proposal=MachineProposal(proposal_id=f"proposal::{case['case_id']}",provider="langextract",model=result.model_ref,prompt_version="tender_sensitive_v1",prompt_hash=content_hash(result.prompt_ref),parameters={"temperature":0},run_id=req.extraction_run_id,raw_output_ref=result.raw_output_ref,raw_output_hash=result.raw_output_hash,confidence=0.9))
    return result,assertions

def build_phase1_extraction_bakeoff(*,root:str|Path,recorded_fixture:str|Path)->dict[str,Any]:
    root=Path(root).resolve();fixture=Path(recorded_fixture).resolve();p=json.loads(fixture.read_text(encoding="utf-8"));cases=_cases(root,p["corpus"]["fixture_paths"])
    by_id={c["case_id"]:c for c in cases}; positives={i for i,c in by_id.items() if c["expected_finding_count"]}; negatives=set(by_id)-positives
    for path,digest in p["corpus"]["source_files"].items():
        if sha256_file(root/path)!=digest:raise ValueError(f"canonical source hash mismatch: {path}")
    baseline=run_deterministic_baseline(cases);legacy={r["case_id"] for r in baseline["results"] if r["actual_finding_count"]}
    replayed={}; raw_hashes=[]
    capture_ids=[x["case_id"] for x in p["raw_provider_response"]]
    if len(capture_ids)!=p["capture"].get("expected_capture_count"):raise ValueError("missing capture IDs under fixture count policy")
    if len(capture_ids)!=len(set(capture_ids)):raise ValueError("duplicate capture case_id")
    if set(capture_ids)-set(by_id):raise ValueError("unknown capture case_id")
    metadata=p["capture"]
    for capture in p["raw_provider_response"]:
        result,assertions=_replay(by_id[capture["case_id"]],capture,root,metadata);raw_hashes.append(result.raw_output_hash)
        if assertions:replayed[capture["case_id"]]=(result,assertions)
    lx=set(replayed); providers=[]
    def metrics(ids:set,kind:str):
        tp=len(ids&positives);fp=len(ids&negatives)
        if kind=="langextract":
            occ=sum(len(r.occurrences) for r,a in replayed.values());ass=sum(len(a) for r,a in replayed.values())
            exact_occ=sum(o.alignment.status.value=="exact" for r,a in replayed.values() for o in r.occurrences)
            grounded=sum(any(a.normalized_assertion.candidate_family_id==by_id[i]["family_id"] for a in aa) for i,(r,aa) in replayed.items())
            cond_gold=p["gold"]["condition_expected"]
            cond=sum(i in replayed and any(a.normalized_assertion.condition==expected for a in replayed[i][1]) for i,expected in cond_gold.items())
            duplicate_assertions=sum(max(0,len(a)-1) for r,a in replayed.values())
            alignment=_ratio(exact_occ,occ);grounding=_ratio(grounded,len(ids));condition=_ratio(cond,len(cond_gold));duplicate=_ratio(duplicate_assertions,ass)
        else: alignment=grounding=condition=_ratio(0,0);duplicate=_ratio(0,len(ids))
        null=_ratio(0,0)
        return {"candidate_recall":_ratio(tp,len(positives)),"candidate_precision":_ratio(tp,len(ids)),"source_alignment_accuracy":alignment,"duplicate_assertion_rate":duplicate,"condition_extraction_accuracy":condition,"exception_extraction_accuracy":null,"hard_negative_proposal_rate":_ratio(fp,len(negatives)),"family_concept_grounding_quality":grounding,"human_accept_rate":null,"human_edit_rate":null,"human_reject_rate":null,"reviewer_minutes_per_accepted_assertion":null,"extraction_runtime_ms":{"value":None,"numerator":None,"denominator":None},"extraction_cost_usd":{"value":None,"numerator":None,"denominator":None},"replay_provenance_completeness":_ratio(10,10) if kind=="langextract" else _ratio(5,7)}
    provenance={k:p["capture"][k] for k in ("command","config","provider_version","model_ref","profile_ref","prompt_ref","recorded_at")};provenance["raw_output_hashes"]=raw_hashes;provenance["provider_options"]=p["capture"]["config"];provenance["capture_count"]=len(p["raw_provider_response"])
    complete=sum(v is not None and v!="" for v in provenance.values());assert complete==len(provenance)==10
    providers=[{"provider":"legacy","output_kind":"executed_deterministic_control","provenance":{"provider":"legacy","provider_version":"legacy.v1","model_ref":"deterministic-term-detector.v1","run_hash":baseline["result_hash"],"case_manifest_hash":baseline["case_manifest_hash"],"prompt_ref":None,"cost_config":None},"metrics":metrics(legacy,"legacy")},{"provider":"langextract","output_kind":"recorded_provider_output_not_live","metrics":metrics(lx,"langextract"),"provenance":provenance}]
    matrix=[]
    for f in range(1,16):
        fam=[c for c in cases if c["family_id"]==f];row={"family_id":f,"family":fam[0]["family"],"providers":{}}
        for name,ids in (("legacy",legacy),("langextract",lx)):
            pos={c["case_id"] for c in fam if c["expected_finding_count"]};neg={c["case_id"] for c in fam if not c["expected_finding_count"]};tp=len(ids&pos);fp=len(ids&neg)
            proposed=ids&{c["case_id"] for c in fam}; fam_occ=[o for i in proposed if i in replayed for o in replayed[i][0].occurrences];aligned=sum(o.alignment.status.value=="exact" for o in fam_occ);grounded=sum(i in replayed and any(a.normalized_assertion.candidate_family_id==f for a in replayed[i][1]) for i in proposed);context_gold={i:e for i,e in p["gold"]["condition_expected"].items() if i in {c["case_id"] for c in fam}};context_den=len(context_gold);context_correct=sum(i in replayed and any(a.normalized_assertion.condition==e for a in replayed[i][1]) for i,e in context_gold.items())
            row["providers"][name]={"tp":tp,"fp":fp,"fn":len(pos-ids),"recall":_ratio(tp,len(pos)),"precision":_ratio(tp,tp+fp),"alignment":_ratio(aligned,len(fam_occ)) if name=="langextract" else _ratio(0,0),"grounding":_ratio(grounded,len(proposed)) if name=="langextract" else _ratio(0,0),"condition_context":{"correct":context_correct if name=="langextract" else 0,"denominator":context_den if name=="langextract" else 0,"value":context_correct/context_den if name=="langextract" and context_den else None}}
        matrix.append(row)
    promotion={"mandatory_replacement":False,"eligibility":{"two_consecutive_wins":False,"smoke_and_deterministic_gates_green":True,"production_baseline_status":False,"human_effort_win_measured":False,"runtime_cost_win_measured":False,"quality_win":False},"reasons":["no two consecutive distinct benchmark wins","status is not READY_FOR_PRODUCTION_BASELINE","human reviewer effort is unmeasured","runtime and cost wins are unmeasured","recorded output is not a live quality win"],"recommendation":"retain_legacy_and_keep_langextract_optional"}
    excerpt_map=_verify_sources(root,cases,p["corpus"]["source_files"]);verified=sum(x["verification_state"]=="verified" for x in excerpt_map);promotion["eligibility"]["source_correspondence_complete"]=verified==len(excerpt_map)
    if verified<len(excerpt_map):promotion["reasons"].append(f"source correspondence verified for only {verified}/{len(excerpt_map)} excerpts")
    fixture_ref=fixture.relative_to(root).as_posix() if fixture.is_relative_to(root) else f"external_fixture:{fixture.name}"
    report={"schema_version":"phase1_extraction_bakeoff_report.v2","authority":"proposal_only_no_graph_mutation","execution_mode":"recorded_provider_output_not_live","fixture_ref":fixture_ref,"fixture_hash":sha256_file(fixture),"corpus":{"case_count":len(cases),"source_root":p["corpus"]["source_root"],"source_file_hashes":p["corpus"]["source_files"],"fixture_hashes":{x:sha256_file(root/x) for x in p["corpus"]["fixture_paths"]},"normalized_excerpt_map":excerpt_map,"source_correspondence":_ratio(verified,len(excerpt_map))},"providers":providers,"family_coverage_matrix":matrix,"promotion":promotion}
    report["report_hash"]=content_hash(report);return report

def render_phase1_extraction_bakeoff_markdown(report:dict)->str:
    p={x["provider"]:x for x in report["providers"]};lines=["# Phase 1 Extraction Bake-off Report","",f"Report hash: `{report['report_hash']}`","",f"Mode: `{report['execution_mode']}`. Authority: `{report['authority']}`.","",f"Corpus: {report['corpus']['case_count']} locked smoke cases from `{report['corpus']['source_root']}`.","","| Metric | legacy/current | recorded LangExtract |","|---|---:|---:|"]
    for n in METRIC_NAMES:
        def fmt(m):return "not measured" if m["value"] is None else f"{m['numerator']}/{m['denominator']} ({m['value']:.4f})" if m["denominator"] is not None else str(m["value"])
        lines.append(f"| {n} | {fmt(p['legacy']['metrics'][n])} | {fmt(p['langextract']['metrics'][n])} |")
    lines += ["","## Promotion","",f"Recommendation: `{report['promotion']['recommendation']}`. Mandatory replacement: `false`.",""]+[f"- {r}" for r in report["promotion"]["reasons"]]
    return "\n".join(lines)+"\n"
