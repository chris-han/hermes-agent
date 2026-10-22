"""Import only the pinned, label-free eight-case Tender preparation.

Called by a trusted setup operation with a governed workspace root. No runtime
route accepts a host path; the input owner/allowlist is fixed in this repository.
This importer does not assign a person or grant reviewer/owner capability.
"""

import hashlib
import json
from pathlib import Path
import os
import tempfile
from services.tender_gold_service import TenderGoldService

HANDOFF = "ad6d51789e471499f43c6c5a239b537440834d877c4867e730ec07226230d893"
PENDING_ROOT = "studies/tender-gold-bootstrap-v1"
REVIEWER = "studies/twil-predicate-scope-owner-clarification-v1/adoption/c0016-source-witness-v1/reviewer"


def import_bootstrap(service: TenderGoldService, *, engineering_fixture=False):
    repo = Path(__file__).resolve().parents[2]
    research = repo / "dlm-research"
    source = research / REVIEWER
    hpath = source / "handoff-manifest.v1.json"
    sha = lambda value: hashlib.sha256(value).hexdigest()
    if sha(hpath.read_bytes()) != HANDOFF:
        raise ValueError("HANDOFF_DRIFT")
    handoff = json.loads(hpath.read_text())
    members = {}
    for name, expected in handoff["sha256"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("UNSAFE_ALLOWLIST_MEMBER")
        value = (source / path).read_bytes()
        if sha(value) != expected:
            raise ValueError("ALLOWLIST_HASH_MISMATCH")
        members[name] = value
    packet = json.loads(members["packet.v1.json"])
    pending = json.loads(
        (research / PENDING_ROOT / "pending-manifest.v1.json").read_text()
    )
    if (
        packet["adopted_contract_sha256"] != pending["contract_sha256"]
        or sha(members["packet.v1.json"]) != pending["packet_sha256"]
    ):
        raise ValueError("CONTRACT_PACKET_DRIFT")
    cases = []
    uploads = service.root / "wiki/uploads/tender-gold-bootstrap-v1"
    if not uploads.resolve().is_relative_to(service.root) or any(
        p.is_symlink()
        for p in [service.root / "wiki", service.root / "wiki/uploads", uploads]
    ):
        raise PermissionError("GOVERNED_UPLOAD_ROOT_REQUIRED")
    uploads.mkdir(parents=True, exist_ok=True)
    for c, binding, s in zip(
        packet["cases"], pending["cases"], packet["sources"], strict=True
    ):
        if (
            c["review_id"],
            c["source_id"],
            c["payload_sha256"],
            c["target_text_sha256"],
        ) != (
            binding["review_id"],
            s["source_id"],
            binding["payload_sha256"],
            binding["target_text_sha256"],
        ):
            raise ValueError("CASE_IDENTITY_DRIFT")
        raw = members[s["raw_file"]]
        if sha(raw) != binding["raw_sha256"]:
            raise ValueError("RAW_SOURCE_DRIFT")
        target = uploads / Path(s["raw_file"]).name
        if target.exists():
            if target.is_symlink() or sha(target.read_bytes()) != sha(raw):
                raise ValueError("SOURCE_IMMUTABLE")
        else:
            descriptor, tmp = tempfile.mkstemp(dir=uploads, prefix=".source-")
            try:
                with os.fdopen(descriptor, "wb") as f:
                    f.write(raw)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp, 0o444)
                os.link(tmp, target)
            finally:
                os.unlink(tmp)
        full = ""
        anchors = []
        for a in s["anchors"]:
            if sha(a["text"].encode()) != a["text_sha256"]:
                raise ValueError("SOURCE_ANCHOR_DRIFT")
            start = len(full)
            full += a["text"]
            anchors.append({**a, "start": start, "end": len(full)})
            full += "\n\n"
        cases.append(
            {
                **binding,
                "target_text": c["target"]["text"],
                "source_text": full,
                "source_text_sha256": sha(full.encode()),
                "anchors": anchors,
                "allowed_locators": [a["locator"] for a in anchors],
                "source_path": str(target.relative_to(service.root)),
            }
        )
    package = dict(
        package_id="tender-gold-bootstrap-v1",
        predicate_id=pending["predicate_id"],
        rule_id=pending["rule_id"],
        contract_sha256=pending["contract_sha256"],
        packet_sha256=pending["packet_sha256"],
        reference_owner=pending["reference_owner"],
        handoff_manifest_sha256=HANDOFF,
        scope=pending["scope"],
        engineering_fixture=engineering_fixture,
        cases=cases,
        semantic_contract=packet["semantic_contract"],
        instructions=members["REVIEW.md"].decode(),
        supplement=members["SUPPLEMENT.md"].decode(),
        human_reviews=0,
        qualified_references=0,
        fresh_holdout_eligible=False,
    )
    service.install_package(package)
    return package
