"""Lossless semantic inputs and label-independent, globally reserved composition keys.

This experimental format does not replace TypedDecisionFrontierV1. Annotation
validation establishes mechanics, not semantic truth or independent authorship.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re
from typing import Any


SEMANTIC_FIELDS = (
    "semantic_specification", "conditions", "exceptions", "exclusions", "evidence_requirements",
)
SPLITS = ("TRAIN", "DEV", "H1", "H2", "H3")
ORIGINS = {"SOURCE_BACKED_PARENT", "CONTROLLED_DESCENDANT", "SYNTHETIC_STRUCTURAL_COVERAGE"}
POPULATIONS = ("SOURCE_BACKED_PARENT", "SOURCE_BACKED_DESCENDANT", "SYNTHETIC_STRUCTURAL_COVERAGE")
PREDICATE_ROLES = {
    "semantic_match", "practical_restriction", "operative_meaning", "scope_match",
    "evidence_sufficiency", "functional_equivalence", "qualification_applies",
    "insufficient_evidence",
}
EVIDENCE_ROLES = {"clause", "qualification", "supporting_record", "scope"}
SHAPES = {
    "condition_pattern": {"interpreted", "semantic_and", "semantic_or", "semantic_not"},
    "exception_topology": {"unless", "except_when", "override"},
    "exclusion_topology": {"out_of_scope", "not_evidence_for", "incompatible_meaning"},
    "evidence_roles": EVIDENCE_ROLES,
}
RELATIONS = {"exception_to", "excludes", "requires_evidence_from", "qualifies", "alternative_to"}
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts/research/tender_structural_dlm/score.txt"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty string")


def _string_list(value: Any, label: str, allowed: set[str] | None = None) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected list")
    for item in value:
        _text(item, label)
        if allowed is not None and item not in allowed:
            raise ValueError(f"{label}: unknown structural role {item!r}")


def candidate_semantics(candidate: dict) -> dict:
    return {key: candidate[key] for key in SEMANTIC_FIELDS}


def _hash(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("provenance requires a SHA-256 hash")


def _artifact_pin(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"artifact_ref", "artifact_hash"}:
        raise ValueError(f"{label} requires an artifact reference and hash")
    _text(value["artifact_ref"], label)
    _hash(value["artifact_hash"])


def _validate_origin(episode: dict) -> None:
    """Validate declared lineage shape; pinned artifacts still require qualification."""
    origin = episode["origin"]
    if origin not in ORIGINS:
        raise ValueError("unknown episode origin")
    original = episode["transform_type"] == "original"
    if original == (origin == "CONTROLLED_DESCENDANT"):
        raise ValueError("origin must distinguish original parents from controlled descendants")
    if original:
        if episode["transformation_contract"] is not None:
            raise ValueError("original parents do not have a descendant transformation contract")
    else:
        _artifact_pin(episode["transformation_contract"], "transformation contract")
    binding = episode["source_binding"]
    if binding is not None:
        sources = [binding]
        if isinstance(binding, dict) and set(binding) == {"sources"}:
            sources = binding["sources"]
            if not isinstance(sources, list) or len(sources) < 2:
                raise ValueError("multiple source binding requires at least two sources")
        fields = {"source_package_ref", "source_package_hash", "evidence_refs",
                  "governance_receipt_ref", "governance_receipt_hash"}
        packages = set()
        evidence_refs = {}
        for source in sources:
            if not isinstance(source, dict) or set(source) != fields:
                raise ValueError("source binding must pin source, evidence and governance receipt")
            for field in ("source_package_ref", "governance_receipt_ref"):
                _text(source[field], field)
            for field in ("source_package_hash", "governance_receipt_hash"):
                _hash(source[field])
            if source["source_package_ref"] in packages:
                raise ValueError("duplicate source package binding")
            packages.add(source["source_package_ref"])
            if not isinstance(source["evidence_refs"], list) or not source["evidence_refs"]:
                raise ValueError("source binding requires governed evidence references")
            for ref in source["evidence_refs"]:
                _artifact_pin(ref, "governed evidence")
                owner = (source["source_package_ref"], ref["artifact_hash"])
                if ref["artifact_ref"] in evidence_refs and evidence_refs[ref["artifact_ref"]] != owner:
                    raise ValueError("conflicting governed evidence binding")
                evidence_refs[ref["artifact_ref"]] = owner
        if episode["synthetic_coverage_reason"] is not None:
            raise ValueError("source-backed lineage cannot be relabeled synthetic coverage")
    else:
        _text(episode["synthetic_coverage_reason"], "synthetic coverage reason")
    if origin == "SOURCE_BACKED_PARENT" and binding is None:
        raise ValueError("source-backed parent requires source binding")
    if origin == "SYNTHETIC_STRUCTURAL_COVERAGE" and binding is not None:
        raise ValueError("synthetic parent cannot claim a source binding")


def population_group(episode: dict) -> str:
    """Source ancestry, not an authored child's tag alone, owns its evidence group."""
    if episode["source_binding"] is None:
        return "SYNTHETIC_STRUCTURAL_COVERAGE"
    return "SOURCE_BACKED_PARENT" if episode["origin"] == "SOURCE_BACKED_PARENT" else "SOURCE_BACKED_DESCENDANT"


def validate_episode(episode: dict) -> None:
    """Reject malformed inputs, ambiguous support and incomplete structural annotations."""
    required = {
        "case_id", "parent_case_id", "split", "transform_type", "ancestry", "question",
        "evidence", "candidates", "composition_edges", "gold_candidate_id", "support",
        "origin", "source_binding", "transformation_contract", "synthetic_coverage_reason",
    }
    if not isinstance(episode, dict) or set(episode) != required:
        raise ValueError("episode fields differ from the research contract")
    for key in ("case_id", "parent_case_id", "transform_type", "question", "gold_candidate_id"):
        _text(episode[key], key)
    if episode["split"] not in SPLITS:
        raise ValueError("unknown split")
    _validate_origin(episode)
    _string_list(episode["ancestry"], "ancestry")
    if not episode["ancestry"] or len(set(episode["ancestry"])) != len(episode["ancestry"]):
        raise ValueError("ancestry must be nonempty and unique")
    if not isinstance(episode["evidence"], list) or not episode["evidence"]:
        raise ValueError("evidence must describe the available or explicitly missing material")
    for evidence in episode["evidence"]:
        if not isinstance(evidence, dict) or set(evidence) != {"role", "text"}:
            raise ValueError("evidence fields differ from the research contract")
        if evidence["role"] not in EVIDENCE_ROLES:
            raise ValueError("unknown evidence role")
        _text(evidence["text"], "evidence text")
    candidates = episode["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 4:
        raise ValueError("frontier size must be between two and four")
    ids, meanings, unknown_ids = set(), set(), set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {*SEMANTIC_FIELDS, "candidate_id", "structure"}:
            raise ValueError("candidate fields differ from the research contract")
        cid = candidate["candidate_id"]
        _text(cid, "candidate_id")
        if cid in ids:
            raise ValueError("duplicate candidate ID")
        ids.add(cid)
        _text(candidate["semantic_specification"], "semantic specification")
        for key in SEMANTIC_FIELDS[1:]:
            _string_list(candidate[key], key)
        meaning = canonical_json(candidate_semantics(candidate))
        if meaning in meanings:
            raise ValueError("duplicate semantic candidate")
        meanings.add(meaning)
        shape = candidate["structure"]
        if not isinstance(shape, dict) or set(shape) != {"role", "predicate_roles", *SHAPES}:
            raise ValueError("candidate structural fields differ from the contract")
        if shape["role"] not in {"substantive", "unknown"}:
            raise ValueError("invalid candidate role")
        _string_list(shape["predicate_roles"], "predicate_roles", PREDICATE_ROLES)
        if not shape["predicate_roles"]:
            raise ValueError("predicate roles cannot be empty")
        if shape["role"] == "unknown":
            unknown_ids.add(cid)
            if shape["predicate_roles"] != ["insufficient_evidence"]:
                raise ValueError("UNKNOWN requires insufficient-evidence structural applicability")
        elif "insufficient_evidence" in shape["predicate_roles"]:
            raise ValueError("insufficient-evidence role belongs only to UNKNOWN")
        for semantic_key, shape_key in zip(SEMANTIC_FIELDS[1:], SHAPES, strict=True):
            _string_list(shape[shape_key], shape_key, SHAPES[shape_key])
            if len(candidate[semantic_key]) != len(shape[shape_key]):
                raise ValueError(f"missing structural annotation for {semantic_key}")
    if len(unknown_ids) != 1:
        raise ValueError("exactly one supplied UNKNOWN is required")
    if not isinstance(episode["composition_edges"], list):
        raise ValueError("composition_edges must be a list")
    seen_edges = set()
    for edge in episode["composition_edges"]:
        if not isinstance(edge, dict) or set(edge) != {"source", "relation", "target"}:
            raise ValueError("invalid composition edge")
        if edge["source"] not in ids or edge["target"] not in ids or edge["relation"] not in RELATIONS:
            raise ValueError("dangling or invalid composition edge")
        key = (edge["source"], edge["relation"], edge["target"])
        if key in seen_edges:
            raise ValueError("duplicate composition edge")
        seen_edges.add(key)
    support = episode["support"]
    if not isinstance(support, dict) or set(support) != ids:
        raise ValueError("support annotations must cover exactly the supplied frontier")
    supported = set()
    for cid, annotation in support.items():
        if not isinstance(annotation, dict) or set(annotation) != {"supported", "rationale", "evidence_indices"}:
            raise ValueError("invalid support annotation")
        if type(annotation["supported"]) is not bool:
            raise ValueError("supported must be a Boolean annotation")
        _text(annotation["rationale"], "support rationale")
        indices = annotation["evidence_indices"]
        if not isinstance(indices, list) or not indices:
            raise ValueError("support rationale must reference evidence or an explicit missing-record entry")
        if any(type(i) is not int or not 0 <= i < len(episode["evidence"]) for i in indices):
            raise ValueError("invalid support evidence index")
        if annotation["supported"]:
            supported.add(cid)
    if supported != {episode["gold_candidate_id"]}:
        raise ValueError("exactly one supported candidate must equal gold")


def model_episode(episode: dict) -> str:
    validate_episode(episode)
    return canonical_json({
        "question": episode["question"],
        "evidence": episode["evidence"],
        "candidates": sorted((candidate_semantics(c) for c in episode["candidates"]), key=canonical_json),
    })


def render_scoring_inputs(episode: dict) -> dict[str, str]:
    """Canonical candidate iteration makes handle/order changes byte-invariant."""
    body = model_episode(episode)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return {
        c["candidate_id"]: template.format(episode=body, target=canonical_json(candidate_semantics(c)))
        for c in sorted(episode["candidates"], key=lambda c: canonical_json(candidate_semantics(c)))
    }


def composition_key(episode: dict) -> str:
    """Canonical labeled candidate graph, independent of wording and observed outcome.

    At most 24 candidate permutations; no graph-library dependency or heuristic hash.
    Each candidate's clause-shape arrays retain their semantic binding/order.
    """
    validate_episode(episode)
    representations = []
    for ordering in permutations(episode["candidates"]):
        numbering = {c["candidate_id"]: i for i, c in enumerate(ordering)}
        representations.append(canonical_json({
            "candidate_count": len(ordering),
            "candidates": [
                {**c["structure"], "predicate_roles": sorted(c["structure"]["predicate_roles"])}
                for c in ordering
            ],
            # Observed missingness is an outcome, not a different permitted evidence role.
            # Missing entries retain the intended role in the authored contract.
            "unknown_applicable": True,
            "evidence_role_pattern": sorted(e["role"] for e in episode["evidence"]),
            "edges": sorted(
                [numbering[e["source"]], e["relation"], numbering[e["target"]]]
                for e in episode["composition_edges"]
            ),
        }))
    return "sha256:" + hashlib.sha256(min(representations).encode("utf-8")).hexdigest()


def outcome_stratum(episode: dict) -> dict:
    validate_episode(episode)
    ordered = sorted(episode["candidates"], key=lambda c: canonical_json(candidate_semantics(c)))
    gold = next(c for c in ordered if c["candidate_id"] == episode["gold_candidate_id"])
    return {
        "outcome": gold["structure"]["role"],
        "gold_candidate_role": gold["structure"]["predicate_roles"],
        "gold_semantic_hash": digest(candidate_semantics(gold)),
        "support_pattern": [episode["support"][c["candidate_id"]]["supported"] for c in ordered],
    }


def audit_population(episodes: list[dict], reserved_h3_keys: set[str]) -> dict:
    """Reject leakage before reporting counts; descendants never count as parents."""
    parents, ancestry_owners, seen_cases = {}, {}, set()
    rows_by_split = defaultdict(list)
    keys_by_split = defaultdict(set)
    primary_parents_by_key = defaultdict(set)
    semantic_splits, semantic_parents = {}, {}
    originals = Counter()
    original_rows = {}
    for episode in episodes:
        validate_episode(episode)
        cid, parent, split = episode["case_id"], episode["parent_case_id"], episode["split"]
        if cid in seen_cases:
            raise ValueError("duplicate case ID")
        seen_cases.add(cid)
        if parent in parents and parents[parent] != split:
            raise ValueError("parent and descendant cross split boundary")
        parents[parent] = split
        if episode["transform_type"] == "original":
            originals[parent] += 1
            original_rows[parent] = episode
        for ancestor in episode["ancestry"]:
            if ancestor in ancestry_owners and ancestry_owners[ancestor] != parent:
                raise ValueError("shared ancestry must use the same independent parent")
            ancestry_owners[ancestor] = parent
        key = composition_key(episode)
        if (key in reserved_h3_keys) != (split == "H3"):
            raise ValueError("reserved H3 composition leaked or H3 key was not reserved")
        semantic_hash = digest(json.loads(model_episode(episode)))
        if semantic_hash in semantic_splits and semantic_splits[semantic_hash] != split:
            raise ValueError("identical semantic input crosses split boundary")
        if semantic_hash in semantic_parents and semantic_parents[semantic_hash] != parent:
            raise ValueError("identical semantic input assigned to different independent parents")
        semantic_splits[semantic_hash] = split
        semantic_parents[semantic_hash] = parent
        rows_by_split[split].append(episode)
        keys_by_split[split].add(key)
        if episode["source_binding"] is not None:
            primary_parents_by_key[key].add(parent)
    if any(originals[parent] != 1 for parent in parents):
        raise ValueError("every parent must have exactly one original")
    for episode in episodes:
        parent = original_rows[episode["parent_case_id"]]
        if episode["source_binding"] != parent["source_binding"]:
            raise ValueError("descendant source binding must equal its actual parent's binding")
        if episode["synthetic_coverage_reason"] != parent["synthetic_coverage_reason"]:
            raise ValueError("descendant synthetic coverage origin must remain bound to its parent")
        if set(episode["ancestry"]) != set(parent["ancestry"]):
            raise ValueError("descendant ancestry must preserve the complete parent ancestry")

    def stats(rows):
        return {
            "observations": len(rows),
            "parents": len({r["parent_case_id"] for r in rows}),
            "composition_keys": len({composition_key(r) for r in rows}),
            "transforms": dict(Counter(r["transform_type"] for r in rows)),
        }

    split_stats = {
        split: stats(rows_by_split[split]) for split in SPLITS
    }
    primary_rows = {
        split: [r for r in rows_by_split[split] if r["source_binding"] is not None] for split in SPLITS
    }
    primary_stats = {split: stats(primary_rows[split]) for split in SPLITS}
    population_stats = {
        group: {**stats([r for r in episodes if population_group(r) == group]),
                "splits": {split: stats([r for r in rows_by_split[split] if population_group(r) == group])
                           for split in SPLITS}}
        for group in POPULATIONS
    }
    deficits = []
    for split in SPLITS:
        minimum = 128 if split == "TRAIN" else 32
        if primary_stats[split]["parents"] < minimum:
            deficits.append(f"{split} needs {minimum} independent source-backed parents")
    h3_keys = keys_by_split["H3"]
    primary_h3_keys = {composition_key(r) for r in primary_rows["H3"]}
    if len(primary_h3_keys) < 4:
        deficits.append("H3 needs four observed reserved composition keys with source-backed parents")
    if any(len(primary_parents_by_key[key]) < 8 for key in primary_h3_keys):
        deficits.append("each primary H3 composition needs eight source-backed parents")
    for split in ("DEV", "H1", "H2", "H3"):
        for transform in ("id_remap", "permutation", "paraphrase", "contrast", "unknown"):
            count = len({r["parent_case_id"] for r in primary_rows[split] if r["transform_type"] == transform})
            if count < 32:
                deficits.append(f"{split}/{transform} needs 32 eligible source-backed parents")
    return {
        "gate": "INCONCLUSIVE_DATA_POWER" if deficits else "STRUCTURAL_POPULATION_MECHANICS_PASS",
        "splits": split_stats, "deficits": deficits,
        "primary_splits": primary_stats, "populations": population_stats,
        "reserved_h3_keys": sorted(reserved_h3_keys),
        "observed_h3_keys": sorted(h3_keys),
        "claim_limit": "Declared-origin mechanics only; pinned governed source packages and transformation contracts still require artifact verification. Does not prove semantic annotation truth, source governance or independent authorship.",
    }
