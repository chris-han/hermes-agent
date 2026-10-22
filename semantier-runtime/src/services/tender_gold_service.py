"""Native Tender annotation workflow and deterministic, actor-filtered exports.

Workflow rows share the governed workspace SQLite owner. Blinded records never
enter generic files: immutable receipts live under its protected .semantier root.
Engineering provisioning is explicit and cannot install reference-owner authority.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import io
import zipfile
from contracts.tender_gold import Annotation, DraftAnnotation
from storage.atomic_json_artifact import publish_immutable_json
from storage.ontology_build_store import ontology_build_store_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class TenderGoldService:
    def __init__(self, workspace_root, *, organization_id, workspace_id):
        self.root = Path(workspace_root).resolve()
        self.org = organization_id
        self.workspace = workspace_id
        path = ontology_build_store_path(self.root)
        if not path.resolve().is_relative_to(self.root):
            raise PermissionError("GOVERNED_STORE_ROOT_REQUIRED")
        self.receipts = self.root / ".semantier/tender-gold-receipts"
        if not self.receipts.resolve().is_relative_to(self.root):
            raise PermissionError("GOVERNED_RECEIPT_ROOT_REQUIRED")
        if enabled():
            self.db=connect_authority('standalone',path,autocommit=True)
            require_tables(self.db,'tender_gold_packages','tender_gold_assignments','tender_gold_records','tender_gold_drafts')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=30)
            self.db.row_factory = sqlite3.Row
            self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tender_gold_packages (
          id TEXT PRIMARY KEY, org TEXT NOT NULL, workspace TEXT NOT NULL, hash TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tender_gold_assignments (
          package TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY(package,actor,role));
        CREATE TABLE IF NOT EXISTS tender_gold_records (
          hash TEXT PRIMARY KEY, package TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL,
          kind TEXT NOT NULL, case_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
          UNIQUE(package,actor,role,kind,case_id,revision));
        CREATE TABLE IF NOT EXISTS tender_gold_drafts (
          package TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL, case_id TEXT NOT NULL,
          revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(package,actor,role,case_id));
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        begin_write(self.db)
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def install_package(self, package):
        """Trusted source importer only; there is no user-controlled install route."""
        if [c["review_id"] for c in package["cases"]] != [
            f"C{i:04d}" for i in range(9, 17)
        ]:
            raise ValueError("FIXED_EIGHT_CASES_REQUIRED")
        if package["scope"] != "EXPOSED_DEVELOPMENT_ONLY":
            raise ValueError("DEVELOPMENT_SCOPE_REQUIRED")
        h = digest(package)
        with self.transaction():
            old = self.db.execute(
                sql_query("SELECT hash,org,workspace FROM tender_gold_packages WHERE id=?",'SELECT hash,org,workspace FROM tender_gold_packages WHERE id=%s'),
                (package["package_id"],),
            ).fetchone()
            if old:
                if tuple(old) != (h, self.org, self.workspace):
                    raise ValueError("PACKAGE_IMMUTABLE")
                return
            self.db.execute(
                sql_query("INSERT INTO tender_gold_packages VALUES (?,?,?,?,?)",'INSERT INTO tender_gold_packages VALUES (%s,%s,%s,%s,%s)'),
                (package["package_id"], self.org, self.workspace, h, encoded(package)),
            )

    def package(self, package_id):
        row = self.db.execute(
            sql_query("SELECT * FROM tender_gold_packages WHERE id=?",'SELECT * FROM tender_gold_packages WHERE id=%s'), (package_id,)
        ).fetchone()
        if not row:
            raise KeyError("PACKAGE_NOT_FOUND")
        if (row["org"], row["workspace"]) != (self.org, self.workspace):
            raise PermissionError("FORBIDDEN_SCOPE")
        package = json.loads(row["payload"])
        if digest(package) != row["hash"]:
            raise ValueError("PACKAGE_INTEGRITY_FAILURE")
        return package

    def provision_engineering_assignment(self, package_id, *, actor_id, role):
        p = self.package(package_id)
        if not p["engineering_fixture"]:
            raise PermissionError("FIXTURE_PROVISIONING_ONLY")
        if role not in {"R1", "R2", "R3"} or not actor_id:
            raise ValueError("INVALID_ASSIGNMENT")
        receipt = dict(
            actor_id=actor_id,
            role=role,
            package_hash=digest(p),
            engineering_fixture=True,
            eligibility_receipt=None,
            owner_binding=None,
        )
        with self.transaction():
            self.db.execute(
                sql_query("INSERT INTO tender_gold_assignments VALUES (?,?,?,?)",'INSERT INTO tender_gold_assignments VALUES (%s,%s,%s,%s)'),
                (package_id, actor_id, role, encoded(receipt)),
            )

    def provision_engineering_owner(self, package_id, *, actor_id):
        """Disposable fixture capability. Never grants genuine owner authority."""
        p = self.package(package_id)
        if not p["engineering_fixture"]:
            raise PermissionError("FIXTURE_PROVISIONING_ONLY")
        receipt = dict(
            actor_id=actor_id,
            role="OWNER",
            package_hash=digest(p),
            engineering_fixture=True,
            eligibility_receipt=None,
            owner_binding="ENGINEERING_FIXTURE_NOT_AUTHENTIC_OWNER",
        )
        with self.transaction():
            self.db.execute(
                sql_query("INSERT INTO tender_gold_assignments VALUES (?,?,?,?)",'INSERT INTO tender_gold_assignments VALUES (%s,%s,%s,%s)'),
                (package_id, actor_id, "OWNER", encoded(receipt)),
            )

    def _owner(self, package_id, actor_id):
        return self.assignment(package_id, actor_id, "OWNER")

    def assignment(self, package_id, actor_id, role):
        p = self.package(package_id)
        row = self.db.execute(
            sql_query("SELECT payload FROM tender_gold_assignments WHERE package=? AND actor=? AND role=?",'SELECT payload FROM tender_gold_assignments WHERE package=%s AND actor=%s AND role=%s'),
            (package_id, actor_id, role),
        ).fetchone()
        if not row:
            raise PermissionError("APPROVED_ASSIGNMENT_REQUIRED")
        receipt = json.loads(row["payload"])
        if receipt["package_hash"] != digest(p):
            raise PermissionError("ASSIGNMENT_PACKAGE_DRIFT")
        if (receipt["actor_id"], receipt["role"]) != (actor_id, role) or receipt[
            "engineering_fixture"
        ] != p["engineering_fixture"]:
            raise PermissionError("ASSIGNMENT_IDENTITY_OR_FIXTURE_SCOPE_MISMATCH")
        # Actual owner-approved assignments are installed through governed transport,
        # never through role selectors, self-claims or administrator privileges.
        if not receipt["engineering_fixture"]:
            raise PermissionError("AUTHENTIC_OWNER_TRANSPORT_UNAVAILABLE")
        return receipt

    def _case(self, p, case_id):
        for c in p["cases"]:
            if c["review_id"] == case_id:
                return c
        raise KeyError("CASE_NOT_FOUND")

    def _validate(self, p, case_id, annotation, *, complete=True):
        a = (Annotation if complete else DraftAnnotation).model_validate(
            annotation.model_dump(mode="json")
        )
        self._case(p, case_id)
        for cite in a.cited_spans:
            source = next(
                (c for c in p["cases"] if c["source_id"] == cite.source_id), None
            )
            if not source or source["raw_sha256"] != cite.raw_sha256:
                raise ValueError("CITATION_SOURCE_MISMATCH")
            if complete:
                original = self.root / source["source_path"]
                if (
                    not original.resolve().is_relative_to(self.root)
                    or original.is_symlink()
                    or not original.is_file()
                    or hashlib.sha256(original.read_bytes()).hexdigest()
                    != source["raw_sha256"]
                ):
                    raise ValueError("ORIGINAL_SOURCE_MISSING_OR_DRIFTED")
            text = source["source_text"]
            if source.get("anchors"):
                anchor = next(
                    (a for a in source["anchors"] if a["locator"] == cite.locator), None
                )
                if not anchor or not (
                    anchor["start"] <= cite.start < cite.end <= anchor["end"]
                ):
                    raise ValueError("CITATION_ANCHOR_MISMATCH")
            if cite.end <= cite.start or text[cite.start : cite.end] != cite.quote:
                raise ValueError("CITATION_SPAN_MISMATCH")
            if cite.locator not in source.get(
                "allowed_locators", [source["target_locator"]]
            ):
                raise ValueError("CITATION_LOCATOR_MISMATCH")
        return a

    def _records(
        self, package_id, *, actor_id=None, role=None, kind=None, case_id=None
    ):
        query = sql_query("SELECT * FROM tender_gold_records WHERE package=?",'SELECT * FROM tender_gold_records WHERE package=%s')
        args = [package_id]
        for column, value in [
            ("actor", actor_id),
            ("role", role),
            ("kind", kind),
            ("case_id", case_id),
        ]:
            if value is not None:
                query += sql_query(f" AND {column}=?",f" AND {column}=%s")
                args.append(value)
        query += " ORDER BY revision,hash"
        records = []
        for row in self.db.execute(query, args):
            record = json.loads(row["payload"])
            if (
                (
                    record["hash"],
                    record["package_id"],
                    record["actor_id"],
                    record["role"],
                    record["kind"],
                    record["case_id"],
                    record["revision"],
                )
                != (
                    row["hash"],
                    row["package"],
                    row["actor"],
                    row["role"],
                    row["kind"],
                    row["case_id"],
                    row["revision"],
                )
                or (record["organization_id"], record["workspace_id"])
                != (self.org, self.workspace)
                or digest({k: v for k, v in record.items() if k != "hash"})
                != record["hash"]
            ):
                raise ValueError("RECORD_INTEGRITY_FAILURE")
            records.append(record)
        return records

    def _record(self, package_id, actor_id, role, kind, case_id, revision, body):
        record = dict(
            package_id=package_id,
            organization_id=self.org,
            workspace_id=self.workspace,
            actor_id=actor_id,
            role=role,
            kind=kind,
            case_id=case_id,
            revision=revision,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            **body,
        )
        record["hash"] = digest(record)
        published = publish_immutable_json(
            self.receipts / (record["hash"] + ".json"),
            record,
            conflict_error="IMMUTABLE_RECEIPT_CONFLICT",
        )
        self.db.execute(
            sql_query("INSERT INTO tender_gold_records VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO tender_gold_records VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
            (
                record["hash"],
                package_id,
                actor_id,
                role,
                kind,
                case_id,
                revision,
                encoded(published),
            ),
        )
        return published

    def save_draft(
        self, package_id, *, actor_id, role, case_id, annotation, expected_revision
    ):
        receipt = self.assignment(package_id, actor_id, role)
        p = self.package(package_id)
        a = self._validate(p, case_id, annotation, complete=False)
        with self.transaction():
            row = self.db.execute(
                sql_query("SELECT revision FROM tender_gold_drafts WHERE package=? AND actor=? AND role=? AND case_id=?",'SELECT revision FROM tender_gold_drafts WHERE package=%s AND actor=%s AND role=%s AND case_id=%s'),
                (package_id, actor_id, role, case_id),
            ).fetchone()
            revision = row["revision"] if row else 0
            if revision != expected_revision:
                raise ValueError("STALE_DRAFT_REVISION")
            value = dict(
                revision=revision + 1,
                annotation=a.model_dump(mode="json"),
                assignment_hash=digest(receipt),
            )
            self.db.execute(
                sql_query("INSERT INTO tender_gold_drafts VALUES (?,?,?,?,?,?) ON CONFLICT(package,actor,role,case_id) DO UPDATE SET revision=excluded.revision,payload=excluded.payload",'INSERT INTO tender_gold_drafts VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(package,actor,role,case_id) DO UPDATE SET revision=excluded.revision,payload=excluded.payload'),
                (package_id, actor_id, role, case_id, revision + 1, encoded(value)),
            )
        return value

    def submit(
        self, package_id, *, actor_id, role, case_id, annotation, expected_revision
    ):
        receipt = self.assignment(package_id, actor_id, role)
        if role not in {"R1", "R2"}:
            raise PermissionError("PRIMARY_ROLE_REQUIRED")
        p = self.package(package_id)
        a = self._validate(p, case_id, annotation)
        with self.transaction():
            if self._records(package_id, actor_id=actor_id, role=role, kind="seal"):
                raise ValueError("PRIMARY_BATCH_SEALED")
            history = self._records(
                package_id,
                actor_id=actor_id,
                role=role,
                kind="annotation",
                case_id=case_id,
            )
            if len(history) != expected_revision:
                raise ValueError("STALE_SUBMISSION_REVISION")
            c = self._case(p, case_id)
            record = self._record(
                package_id,
                actor_id,
                role,
                "annotation",
                case_id,
                len(history) + 1,
                dict(
                    annotation=a.model_dump(mode="json"),
                    package_hash=digest(p),
                    case_version_hash=digest(c),
                    assignment_hash=digest(receipt),
                    assignment=receipt,
                    predicate_id=p["predicate_id"],
                    contract_sha256=p["contract_sha256"],
                    packet_sha256=p["packet_sha256"],
                    predecessor_hash=history[-1]["hash"] if history else None,
                    engineering_fixture=receipt["engineering_fixture"],
                ),
            )
        # Compiler is a derived projection. Submission success never depends on a
        # writable export cache and replay always reconstructs the exact projection.
        compiled = self.compile_dataset(
            package_id, actor_id=actor_id, role=role, kind="candidate"
        )
        return {
            **record,
            "candidate_version": compiled["version"],
            "cache_status": compiled["cache_status"],
        }

    def seal(self, package_id, *, actor_id, role, attestation):
        receipt = self.assignment(package_id, actor_id, role)
        if role not in {"R1", "R2"}:
            raise PermissionError("PRIMARY_ROLE_REQUIRED")
        if not attestation.strip():
            raise ValueError("ATTESTATION_REQUIRED")
        p = self.package(package_id)
        with self.transaction():
            seals = self._records(package_id, actor_id=actor_id, role=role, kind="seal")
            if seals:
                if seals[0]["attestation"] != attestation:
                    raise ValueError("SEAL_IMMUTABLE")
                return seals[0]
            ordered = []
            for c in p["cases"]:
                history = self._records(
                    package_id,
                    actor_id=actor_id,
                    role=role,
                    kind="annotation",
                    case_id=c["review_id"],
                )
                if not history:
                    raise ValueError("MISSING_CASES")
                ordered.append(
                    dict(
                        case_id=c["review_id"],
                        case_version_hash=digest(c),
                        submission_hash=history[-1]["hash"],
                    )
                )
            return self._record(
                package_id,
                actor_id,
                role,
                "seal",
                "",
                1,
                dict(
                    case_version_hashes=ordered,
                    attestation=attestation,
                    package_hash=digest(p),
                    assignment_hash=digest(receipt),
                    engineering_fixture=receipt["engineering_fixture"],
                ),
            )

    def comparison(self, package_id, *, actor_id):
        self.assignment(package_id, actor_id, "R3")
        return self._sealed_comparison(package_id, adjudicator=actor_id)

    def _sealed_comparison(self, package_id, *, adjudicator=None):
        p = self.package(package_id)
        seals = self._records(package_id, kind="seal")
        by_role = {
            role: [s for s in seals if s["role"] == role] for role in ["R1", "R2"]
        }
        if any(len(v) != 1 for v in by_role.values()):
            raise PermissionError("BOTH_PRIMARY_BATCHES_MUST_BE_SEALED")
        left, right = by_role["R1"][0], by_role["R2"][0]
        if left["actor_id"] == right["actor_id"] or (
            adjudicator is not None
            and adjudicator in {left["actor_id"], right["actor_id"]}
        ):
            raise PermissionError("INDEPENDENT_REVIEWERS_REQUIRED")
        disagreements = []
        agreements = []
        records = {r["hash"]: r for r in self._records(package_id, kind="annotation")}
        for c, l, r in zip(
            p["cases"],
            left["case_version_hashes"],
            right["case_version_hashes"],
            strict=True,
        ):
            if l["case_version_hash"] != digest(c) or r["case_version_hash"] != digest(
                c
            ):
                raise ValueError("SEALED_CASE_DRIFT")
            a, b = records[l["submission_hash"]], records[r["submission_hash"]]
            row = dict(case_id=c["review_id"], R1=a, R2=b)
            (
                agreements if a["annotation"] == b["annotation"] else disagreements
            ).append(row)
        return dict(
            agreements=agreements,
            disagreements=disagreements,
            primary_seals=[left["hash"], right["hash"]],
        )

    def view(self, package_id, *, actor_id, role):
        assignment = self.assignment(package_id, actor_id, role)
        p = self.package(package_id)
        history = {
            c["review_id"]: self._records(
                package_id,
                actor_id=actor_id,
                role=role,
                kind="adjudication" if role == "R3" else "annotation",
                case_id=c["review_id"],
            )
            for c in p["cases"]
        }
        drafts = {
            r["case_id"]: json.loads(r["payload"])
            for r in self.db.execute(
                sql_query("SELECT case_id,payload FROM tender_gold_drafts WHERE package=? AND actor=? AND role=?",'SELECT case_id,payload FROM tender_gold_drafts WHERE package=%s AND actor=%s AND role=%s'),
                (package_id, actor_id, role),
            )
        }
        return dict(
            package=p,
            actor_id=actor_id,
            role=role,
            assignment=assignment,
            drafts=drafts,
            history=history,
            seals=self._records(package_id, actor_id=actor_id, role=role, kind="seal"),
            qualified_available=False,
            qualified_unavailable_reason="AUTHENTIC_REFERENCE_OWNER_BINDING_REQUIRED",
            engineering_owner_available=self.db.execute(
                sql_query("SELECT 1 FROM tender_gold_assignments WHERE package=? AND actor=? AND role='OWNER'","SELECT 1 FROM tender_gold_assignments WHERE package=%s AND actor=%s AND role='OWNER'"),
                (package_id, actor_id),
            ).fetchone()
            is not None,
        )

    def adjudicate(
        self,
        package_id,
        *,
        actor_id,
        case_id,
        annotation,
        primary_hashes,
        expected_revision,
        accept=False,
    ):
        comparison = self.comparison(package_id, actor_id=actor_id)
        pair = next(
            (r for r in comparison["disagreements"] if r["case_id"] == case_id), None
        )
        if not pair:
            raise ValueError("MATERIAL_DISAGREEMENT_REQUIRED")
        expected = [pair["R1"]["hash"], pair["R2"]["hash"]]
        if primary_hashes != expected:
            raise ValueError("PRIMARY_SUBMISSION_DRIFT")
        p = self.package(package_id)
        a = self._validate(p, case_id, annotation)
        # Combined action validates both permissions before any write. The two
        # distinct durable receipts are committed together, preserving originals.
        if accept:
            self._owner(package_id, actor_id)
        with self.transaction():
            previous = self._records(package_id, kind="adjudication", case_id=case_id)
            if len(previous) != expected_revision:
                raise ValueError("STALE_ADJUDICATION_REVISION")
            record = self._record(
                package_id,
                actor_id,
                "R3",
                "adjudication",
                case_id,
                len(previous) + 1,
                dict(
                    annotation=a.model_dump(mode="json"),
                    primary_hashes=expected,
                    primary_seals=comparison["primary_seals"],
                    package_hash=digest(p),
                    predecessor_hash=previous[-1]["hash"] if previous else None,
                    engineering_fixture=p["engineering_fixture"],
                ),
            )
            acceptance = (
                self._accept(package_id, actor_id, case_id, record, comparison)
                if accept
                else None
            )
        release = (
            self.compile_dataset(
                package_id, actor_id=actor_id, role="R3", kind="qualified"
            )
            if accept
            else None
        )
        return dict(
            adjudication=record,
            acceptance=acceptance,
            release_version=release["version"] if release else None,
            cache_status=release["cache_status"] if release else None,
        )

    def _accept(
        self, package_id, actor_id, case_id, reference, comparison, disputed=False
    ):
        owner = self._owner(package_id, actor_id)
        p = self.package(package_id)
        prior = self._records(package_id, kind="acceptance", case_id=case_id)
        pair = next(
            r
            for r in comparison["agreements"] + comparison["disagreements"]
            if r["case_id"] == case_id
        )
        return self._record(
            package_id,
            actor_id,
            "OWNER",
            "acceptance",
            case_id,
            len(prior) + 1,
            dict(
                package_hash=digest(p),
                reference_hash=reference["hash"],
                reference=reference,
                primary_hashes=[pair["R1"]["hash"], pair["R2"]["hash"]],
                primary_seals=comparison["primary_seals"],
                owner_binding=owner,
                reference_status="DISPUTED" if disputed else "ACCEPTED",
                predecessor_hash=prior[-1]["hash"] if prior else None,
                engineering_fixture=p["engineering_fixture"],
            ),
        )

    def accept(self, package_id, *, actor_id, case_id, reference_hash, disputed=False):
        self._owner(package_id, actor_id)
        # Owner comparison obeys the same complete-batch blind boundary, with no
        # requirement that owner also be the designated adjudicator.
        comparison = self._comparison_for_owner(package_id, actor_id)
        pair = next(
            (
                r
                for r in comparison["agreements"] + comparison["disagreements"]
                if r["case_id"] == case_id
            ),
            None,
        )
        if not pair:
            raise KeyError("CASE_NOT_FOUND")
        disagreements = {r["case_id"] for r in comparison["disagreements"]}
        with self.transaction():
            if case_id in disagreements:
                adjudications = self._records(
                    package_id, kind="adjudication", case_id=case_id
                )
                if not adjudications:
                    raise ValueError("INDEPENDENT_ADJUDICATION_REQUIRED")
                reference = adjudications[-1]
            else:
                reference = pair["R1"]
            if reference["hash"] != reference_hash:
                raise ValueError("REFERENCE_DRIFT")
            receipt = self._accept(
                package_id, actor_id, case_id, reference, comparison, disputed=disputed
            )
        release = self.compile_dataset(
            package_id, actor_id=actor_id, role="OWNER", kind="qualified"
        )
        return {
            **receipt,
            "release_version": release["version"],
            "cache_status": release["cache_status"],
        }

    def _comparison_for_owner(self, package_id, actor_id):
        self._owner(package_id, actor_id)
        return self._sealed_comparison(package_id)

    def qualified_dataset(self, package_id, *, actor_id):
        self._owner(package_id, actor_id)
        comparison = self._comparison_for_owner(package_id, actor_id)
        p = self.package(package_id)
        ledger = []
        semantic = []
        routing = []
        all_records = self._records(package_id)
        for c in p["cases"]:
            accepted = self._records(
                package_id, kind="acceptance", case_id=c["review_id"]
            )
            receipt = accepted[-1] if accepted else None
            status = receipt["reference_status"] if receipt else "MISSING"
            row = dict(
                case_id=c["review_id"],
                source_group_id=c["source_id"],
                reference_status=status,
                case_version_hash=digest(c),
                development_exposure="KNOWN",
                external_exposure="UNKNOWN",
                source_template_independence="UNQUALIFIED",
                acceptance=receipt,
                original_submissions=[
                    r for r in all_records if r["case_id"] == c["review_id"]
                ],
            )
            ledger.append(row)
            if status == "ACCEPTED":
                reference = receipt["reference"]
                record = dict(
                    case=c,
                    reference=reference["annotation"],
                    receipt=receipt,
                    predicate_id=p["predicate_id"],
                    contract_sha256=p["contract_sha256"],
                    packet_sha256=p["packet_sha256"],
                    source_sha256=c["raw_sha256"],
                    scope=p["scope"],
                    engineering_fixture=p["engineering_fixture"],
                )
                routing.append(record)
                if record["reference"]["result"] in {"TRUE", "FALSE"}:
                    semantic.append(record)
        # Fixture releases exercise the compiler but are explicitly excluded from
        # real gold. No scientific ledger or original pending manifest is changed.
        body = dict(
            schema="tender.gold.release.v1",
            compiler_version="1",
            package_hash=digest(p),
            scope=p["scope"],
            kind="qualified",
            engineering_fixture=p["engineering_fixture"],
            qualified_references=0,
            qualification_status="ENGINEERING_ONLY_NOT_REFERENCE",
            fresh_holdout_eligible=False,
            engineering_accepted_references=sum(
                r["reference_status"] == "ACCEPTED" for r in ledger
            ),
            counts={
                state: sum(r["reference_status"] == state for r in ledger)
                for state in ["ACCEPTED", "DISPUTED", "MISSING"]
            },
            semantic=semantic,
            routing=routing,
            case_ledger=ledger,
            primary_seals=comparison["primary_seals"],
        )
        return {**body, "version": digest(body)}

    def dataset(self, package_id, *, actor_id, role, kind, version=None):
        self.assignment(package_id, actor_id, role)
        p = self.package(package_id)
        if version is not None:
            import re

            if not re.fullmatch(r"[0-9a-f]{64}", version):
                raise ValueError("INVALID_DATASET_VERSION")
            if kind == "qualified":
                self._owner(package_id, actor_id)
            elif role == "R3":
                self.comparison(package_id, actor_id=actor_id)
            target = self._dataset_path(package_id, actor_id, role, kind, version)
            if not target.is_file():
                current = self.dataset(
                    package_id, actor_id=actor_id, role=role, kind=kind
                )
                if current["version"] == version:
                    return current
                raise KeyError("PINNED_DATASET_VERSION_UNAVAILABLE")
            data = json.loads(target.read_text())
            if (
                data["version"] != version
                or data["package_hash"] != digest(p)
                or digest({k: v for k, v in data.items() if k != "version"}) != version
            ):
                raise ValueError("PINNED_DATASET_INTEGRITY_FAILURE")
            return data
        if kind == "qualified":
            return self.qualified_dataset(package_id, actor_id=actor_id)
        if kind != "candidate":
            raise ValueError("INVALID_DATASET_KIND")
        if role == "R3":
            comparison = self.comparison(package_id, actor_id=actor_id)
            rows = [
                r
                for pair in comparison["agreements"] + comparison["disagreements"]
                for r in [pair["R1"], pair["R2"]]
            ]
        else:
            rows = self._records(
                package_id, actor_id=actor_id, role=role, kind="annotation"
            )
        ledger = [
            dict(
                case_id=c["review_id"],
                source_id=c["source_id"],
                source_group_id=c["source_id"],
                case_version_hash=digest(c),
                reference_status="MISSING",
                development_exposure="KNOWN",
                external_exposure="UNKNOWN",
                source_template_independence="UNQUALIFIED",
            )
            for c in p["cases"]
        ]
        body = dict(
            schema="tender.gold.candidate.v1",
            compiler_version="1",
            kind=kind,
            scope=p["scope"],
            engineering_fixture=p["engineering_fixture"],
            qualified_references=0,
            qualification_status="ENGINEERING_ONLY_NOT_REFERENCE",
            fresh_holdout_eligible=False,
            package_hash=digest(p),
            predicate_id=p["predicate_id"],
            contract_sha256=p["contract_sha256"],
            packet_sha256=p["packet_sha256"],
            case_ledger=ledger,
            records=rows,
        )
        return {**body, "version": digest(body)}

    def export(self, package_id, *, actor_id, role, kind, version=None):
        return (
            encoded(
                self.dataset(
                    package_id, actor_id=actor_id, role=role, kind=kind, version=version
                )
            )
            + "\n"
        ).encode()

    def bundle(self, package_id, *, actor_id, role, kind, version=None):
        data = self.dataset(
            package_id, actor_id=actor_id, role=role, kind=kind, version=version
        )
        lines = lambda rows: "".join(encoded(row) + "\n" for row in rows).encode()
        files = {
            "case-ledger.jsonl": lines(data["case_ledger"]),
            "semantic.jsonl": lines(data.get("semantic", [])),
            "routing.jsonl": lines(data.get("routing", data.get("records", []))),
        }
        manifest = {
            k: v
            for k, v in data.items()
            if k not in {"case_ledger", "semantic", "routing", "records"}
        }
        manifest["files"] = {
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        }
        files["manifest.json"] = (encoded(manifest) + "\n").encode()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, value in sorted(files.items()):
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o444 << 16
                archive.writestr(entry, value)
        return stream.getvalue()

    def compile_dataset(self, package_id, *, actor_id, role, kind):
        data = self.dataset(package_id, actor_id=actor_id, role=role, kind=kind)
        target = self._dataset_path(package_id, actor_id, role, kind, data["version"])
        try:
            publish_immutable_json(
                target, data, conflict_error="DATASET_VERSION_CONFLICT"
            )
        except OSError:
            return {**data, "cache_status": "CACHE_UNAVAILABLE_EXPORT_REPLAY_READY"}
        return {**data, "cache_status": "PUBLISHED"}

    def _dataset_path(self, package_id, actor_id, role, kind, version):
        key = {"package": package_id, "kind": kind}
        if kind == "candidate":
            key.update(actor=actor_id, role=role)
        return self.receipts / "datasets" / digest(key) / (version + ".json")
