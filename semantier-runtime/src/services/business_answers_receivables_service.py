"""Native candidate ontology and deterministic source-bounded receivables answers."""

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import io
import json
import re
from hashlib import sha256
from pathlib import Path
from agents.auth_session import RequestContext
from agents.company_dataset_imports import (
    read_authorized_import_file,
    CompanyDatasetImportError,
    CompanyDatasetAuthorizationError,
    MAX_FILE_BYTES,
)
from contracts.business_answers_receivables import (
    ReceivablesRunRequest,
    ReceivablesSourceV1,
    ReceivablesReportV1,
    ReceivablesRunV1,
    ReceivablesAnswerV1,
    ReceivablesCheck,
    ReceivablesInvoice,
    ReceivablesCustomer,
    ReceivablesCurrencyTotal,
    make_artifact,
)
from contracts.evidence_location import SourceIdentity
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import (
    ArtifactPinV1,
    ExploratoryGraphNodeV1,
    ExploratoryGraphEdgeV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
)
from contracts.ontology_artifact_integrity import create_exploratory_snapshot_integrity
from eos.artifact_hashing import content_hash
from services.receivables_tabular_adapter import (
    parse_tabular,
    tabular_canonical_ir,
    tabular_evidence_refs,
)
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    persist_canonical_source_ir,
)
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_pin,
)
from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore

FIELDS = (
    "customer_id",
    "invoice_id",
    "outstanding_balance",
    "currency",
    "due_date",
    "customer_name",
    "snapshot_date",
)
# Localized source headers are recipe metadata, never machine identifiers.
RECIPE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "catalogs"
        / "business_answers_receivables_recipe.v1.json"
    ).read_text(encoding="utf-8")
)
RECIPE_HASH = "sha256:" + content_hash(RECIPE)
ALIASES = RECIPE["field_aliases"]


def _pin(kind, artifact):
    return ArtifactPinV1(
        artifact_kind=kind,
        artifact_ref=artifact.artifact_ref,
        artifact_hash=artifact.artifact_hash,
    )


def _money(value):
    return format(value, ".2f")


def _mapping(headers, supplied):
    if supplied and (
        set(supplied) - set(FIELDS)
        or any(value and value not in headers for value in supplied.values())
    ):
        raise ValueError("INVALID_FIELD_MAPPING")
    mapping = dict(supplied or {})
    for field, aliases in ALIASES.items():
        if field in (supplied or {}):
            continue
        matches = [h for h in headers if h.strip().casefold() in aliases]
        if len(matches) == 1:
            mapping[field] = matches[0]
    mapping = {field: column for field, column in mapping.items() if column}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("FIELD_MAPPING_MUST_BE_DISTINCT")
    return mapping


class BusinessAnswersReceivablesService:
    def __init__(self, *, build_store, knowledge_store, authority_resolver):
        self._store, self._knowledge, self._authority = (
            build_store,
            knowledge_store,
            authority_resolver,
        )

    def _access(self, ctx):
        if (
            not isinstance(ctx, RequestContext)
            or not ctx.authenticated
            or not ctx.organization_id
            or not ctx.workspace_id
        ):
            raise PermissionError("AUTHORITY_CONTEXT_REQUIRED")
        self._authority.require_current_actor(
            actor_context=RuntimeContextDTO(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                actor_id=ctx.user_id,
            )
        )
        if self._knowledge.workspace_root != ctx.workspace_root.resolve():
            raise PermissionError("WORKSPACE_OWNER_MISMATCH")

    def _get(self, ctx, ref, model):
        artifact = self._store.get(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            artifact_ref=ref,
            contract_type=model,
        )
        if artifact is None:
            raise ValueError("RECEIVABLES_ARTIFACT_NOT_FOUND")
        return artifact

    def _source(self, ctx, record, file, data):
        source_hash = "sha256:" + sha256(data).hexdigest()
        source_key = content_hash(
            {
                "org": ctx.organization_id,
                "workspace": ctx.workspace_id,
                "import_id": record["import_id"],
                "file_id": file["file_id"],
                "source_hash": source_hash,
                "parser_profile_hash": record["parser_profile_hash"],
            }
        )
        source_id = "kbs_receivables_" + source_key
        identity_ref = "governed-source:" + source_id
        parsed = parse_tabular(
            data,
            original_name=file["original_filename"],
            parser_profile=record["parser_profile"],
        )
        media = (
            "text/csv"
            if file["original_filename"].lower().endswith(".csv")
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        blob = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=MAX_FILE_BYTES)
        stage = blob.stage(
            byte_stream=io.BytesIO(data),
            expected_hash=source_hash,
            expected_size=len(data),
            original_name=file["original_filename"],
            declared_media_type=media,
        )
        byte_ref = blob.publish(stage=stage, artifact_ref=identity_ref)
        source = self._knowledge.create_source(
            source_id=source_id,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            source_kind="file",
            source_ref=record["import_id"] + ":" + file["file_id"],
            source_text=byte_ref,
            source_metadata={
                "original_path": byte_ref,
                "original_content_hash": source_hash,
                "original_name": file["original_filename"],
            },
        )
        now = datetime.fromisoformat(source["created_at"].replace("Z", "+00:00"))
        identity = SourceIdentity(
            schemaVersion="semantier.source_identity.v1",
            sourceIdentityRef=identity_ref,
            organizationId=ctx.organization_id,
            workspaceId=ctx.workspace_id,
            sourceHash=source_hash,
            mediaType=media,
            originalName=file["original_filename"],
            artifactRef=byte_ref,
            sourceVersion=None,
            supersedesSourceIdentityRef=None,
            createdAt=now,
        )
        document = tabular_canonical_ir(
            parsed,
            source_identity_ref=identity_ref,
            source_hash=source_hash,
            media_type=media,
        )
        ir_ref = persist_canonical_source_ir(
            document,
            ctx.workspace_root,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
        )
        self._knowledge.bind_canonical_ir_ref(
            source_id=source_id,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            canonical_ir_ref=ir_ref,
            expected_content_hash=source["content_hash"],
        )
        self._knowledge.register_source_identity(
            source_identity=identity, lineage_id=source_id
        )
        identity = self._knowledge.get_source_identity(
            source_identity_ref=identity_ref,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            lineage_id=source_id,
        )
        evidence = tabular_evidence_refs(
            document, source_identity=identity, parsed=parsed, created_at=now
        )
        artifact = make_artifact(
            ReceivablesSourceV1,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            created_at=now,
            import_id=record["import_id"],
            file_id=file["file_id"],
            source_identity_ref=identity_ref,
            source_hash=source_hash,
            byte_owner_ref=byte_ref,
            byte_size=len(data),
            original_name=file["original_filename"],
            parser_profile=record["parser_profile"],
            parser_profile_hash=record["parser_profile_hash"],
            canonical_source_ir=document,
            canonical_source_ir_pin=ArtifactPinV1(
                artifact_kind="canonical_source_ir",
                artifact_ref=ir_ref,
                artifact_hash="sha256:"
                + content_hash(
                    document.model_dump(mode="json", by_alias=True, exclude_none=False)
                ),
            ),
            evidence=evidence,
            sheet_name=parsed.sheet_name,
            header_row=parsed.header_row,
            headers=parsed.headers,
            rows=parsed.rows,
            row_numbers=parsed.row_numbers,
        )
        self._store.put(artifact)
        return artifact

    def _read_import(self, **kwargs):
        try:
            return read_authorized_import_file(**kwargs)
        except CompanyDatasetAuthorizationError as exc:
            raise PermissionError("RECEIVABLES_SOURCE_ACCESS_DENIED") from exc
        except CompanyDatasetImportError as exc:
            raise ValueError(str(exc)) from exc
        except OSError as exc:
            raise ValueError("RECEIVABLES_SOURCE_UNAVAILABLE") from exc

    def _verify_source(self, ctx, source):
        try:
            return self._verify_source_pins(ctx, source)
        except PermissionError:
            raise
        except OSError as exc:
            raise ValueError("RECEIVABLES_SOURCE_UNAVAILABLE") from exc

    def _verify_source_pins(self, ctx, source):
        # Current membership was checked before resolving any source contents.
        record, file, data = self._read_import(
            ctx=ctx, import_id=source.import_id, file_id=source.file_id
        )
        if (
            "sha256:" + sha256(data).hexdigest() != source.source_hash
            or record["parser_profile_hash"] != source.parser_profile_hash
        ):
            raise ValueError("RECEIVABLES_SOURCE_CHANGED")
        with WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=source.byte_size).exact_get(
            byte_owner_ref=source.byte_owner_ref,
            expected_hash=source.source_hash,
            expected_size=source.byte_size,
        ):
            pass
        ir_path = ctx.workspace_root / source.canonical_source_ir_pin.artifact_ref
        if ir_path.is_symlink() or not ir_path.resolve().is_relative_to(
            ctx.workspace_root.resolve()
        ):
            raise ValueError("RECEIVABLES_SOURCE_PIN_MISMATCH")
        if (
            "sha256:" + content_hash(json.loads(ir_path.read_text()))
            != source.canonical_source_ir_pin.artifact_hash
        ):
            raise ValueError("RECEIVABLES_SOURCE_PIN_MISMATCH")
        retained_ir = json.loads(ir_path.read_text())
        if retained_ir != source.canonical_source_ir.model_dump(
            mode="json", by_alias=True, exclude_none=False
        ):
            raise ValueError("RECEIVABLES_SOURCE_PIN_MISMATCH")
        identity = self._knowledge.get_source_identity(
            source_identity_ref=source.source_identity_ref,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            lineage_id=source.source_identity_ref.removeprefix("governed-source:"),
        )
        if (
            identity is None
            or identity.source_hash != source.source_hash
            or identity.artifact_ref != source.byte_owner_ref
        ):
            raise ValueError("RECEIVABLES_SOURCE_IDENTITY_MISMATCH")
        if len(source.rows) != len(source.row_numbers) or len(source.evidence) != len(
            source.canonical_source_ir.blocks
        ):
            raise ValueError("RECEIVABLES_SOURCE_ROW_CLOSURE_MISMATCH")
        cells = {
            (
                mapping.source_locators[0].native_metadata["row_number"],
                mapping.source_locators[0].native_metadata["column_number"],
            ): mapping.block_ids[0]
            for mapping in source.canonical_source_ir.source_map.mappings
        }
        content = {
            block.block_id: block.content for block in source.canonical_source_ir.blocks
        }
        for row_number, row in (
            (source.header_row, source.headers),
            *zip(source.row_numbers, source.rows),
        ):
            for column, value in enumerate(row, 1):
                if content.get(cells.get((row_number, column))) != value:
                    raise ValueError("RECEIVABLES_SOURCE_ROW_CLOSURE_MISMATCH")
        for evidence in source.evidence:
            evidence_ref_pin(evidence)
            if (
                evidence.source_identity_ref != source.source_identity_ref
                or evidence.provenance["source_hash"] != source.source_hash
            ):
                raise ValueError("RECEIVABLES_SOURCE_EVIDENCE_CLOSURE_MISMATCH")

    def create_run(self, *, ctx, request):
        self._access(ctx)
        request = ReceivablesRunRequest.model_validate(request)
        record, file, data = self._read_import(
            ctx=ctx, import_id=request.import_id, file_id=request.file_id
        )
        request_hash = "sha256:" + content_hash(
            {
                "organization_id": ctx.organization_id,
                "workspace_id": ctx.workspace_id,
                "actor_id": ctx.user_id,
                "request": request.model_dump(mode="json"),
                "file_id": file["file_id"],
                "source_hash": file["source_file_hash"],
                "parser_profile_hash": record["parser_profile_hash"],
                "recipe_hash": RECIPE_HASH,
            }
        )
        run_ref = "receivables_run:" + request_hash[7:]
        saved = self._store.get(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            artifact_ref=run_ref,
            contract_type=ReceivablesRunV1,
        )
        if saved is not None:
            self._verify_source(
                ctx, self._get(ctx, saved.source_pin.artifact_ref, ReceivablesSourceV1)
            )
            return saved
        source = self._source(ctx, record, file, data)
        self._verify_source(ctx, source)
        now = datetime.now(timezone.utc)
        mapping = _mapping(source.headers, request.mapping)
        build_ref = "ontology-build:receivables:" + request_hash[7:]
        predecessor = None
        if request.predecessor_build_ref:
            predecessor = self._get(
                ctx, request.predecessor_build_ref, OntologyBuildManifestV1
            )
            # A native OW2 edit may own a different producer prefix. Source pins
            # and exact tenant lineage are validated before using its snapshot.
        checks, invoices, definition_status, snapshot_date = self._check(
            source, mapping, request, ctx
        )
        source_pin = _pin("receivables_source", source)
        snapshot = self._graph(
            ctx, source, invoices, build_ref, predecessor, now, checks
        )
        checks = checks + (self._structure_check(source, snapshot, invoices),)
        report = make_artifact(
            ReceivablesReportV1,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            created_at=now,
            build_ref=build_ref,
            source_pin=source_pin,
            mapping=mapping,
            mapping_hash="sha256:" + content_hash(mapping),
            schema_hash="sha256:"
            + content_hash(
                {
                    "types": ["Customer", "Invoice", "SourceSnapshot"],
                    "fields": mapping,
                    "relations": ["billed_to", "observed_in"],
                    "recipe": "receivables.v1",
                    "recipe_hash": RECIPE_HASH,
                }
            ),
            recipe_hash=RECIPE_HASH,
            snapshot_pin=ArtifactPinV1(
                artifact_kind="exploratory_graph_snapshot",
                artifact_ref=snapshot.snapshot_ref,
                artifact_hash=snapshot.snapshot_hash,
            ),
            canonical_source_ir_pin=source.canonical_source_ir_pin,
            parser_profile_hash=source.parser_profile_hash,
            snapshot_date=snapshot_date,
            checks=checks,
            business_definition_status=definition_status,
            confirmation_actor_id=ctx.user_id
            if definition_status == "confirmed"
            else None,
            confirmation=(
                {"accepted": True}
                if request.confirmation is True
                else request.confirmation
            )
            if definition_status == "confirmed"
            else None,
            governance_state="candidate",
            row_count=len(source.rows),
        )
        integrity = create_exploratory_snapshot_integrity(
            snapshot=snapshot,
            producer_profile="receivables_descriptive_build.v1",
            producer_version="receivables_descriptive_v1",
            recorded_at=now,
        )
        manifest_basis = dict(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            assurance_profile="A0_EXPLORATORY",
            child_artifacts=(
                source_pin,
                _pin("receivables_report", report),
                ArtifactPinV1(
                    artifact_kind="exploratory_graph_snapshot",
                    artifact_ref=snapshot.snapshot_ref,
                    artifact_hash=snapshot.snapshot_hash,
                ),
                ArtifactPinV1(
                    artifact_kind="exploratory_graph_snapshot_integrity",
                    artifact_ref=integrity.integrity_ref,
                    artifact_hash=integrity.integrity_hash,
                ),
            ),
            gates=(),
            predecessor_build_ref=predecessor.build_ref if predecessor else None,
            predecessor_build_hash=predecessor.build_hash if predecessor else None,
            created_at=now,
        )
        from pydantic_core import to_jsonable_python

        manifest = OntologyBuildManifestV1(
            build_ref=build_ref,
            build_hash="sha256:" + content_hash(to_jsonable_python(manifest_basis)),
            **manifest_basis,
        )
        hard_failure = any(c.status == "failed" and c.blocks == "all" for c in checks)
        overdue_available = not any(
            c.status != "passed" and c.blocks == "overdue" for c in checks
        )
        needs_confirmation = (
            definition_status == "needs_confirmation"
            or not all(
                field in mapping
                for field in (
                    "customer_id",
                    "invoice_id",
                    "outstanding_balance",
                    "currency",
                )
            )
            or snapshot_date is None
        )
        state = (
            "needs_confirmation"
            if needs_confirmation
            else (
                "needs_correction"
                if hard_failure
                else ("answered" if overdue_available else "partially_answered")
            )
        )
        answer = None
        if state in ("answered", "partially_answered"):
            answer = self._answer(
                ctx,
                source,
                report,
                invoices,
                snapshot,
                run_ref,
                snapshot_date,
                overdue_available,
                now,
                manifest.build_hash,
            )
        run = make_artifact(
            ReceivablesRunV1,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            created_at=now,
            request=request,
            request_hash=request_hash,
            state=state,
            build_ref=build_ref,
            snapshot_ref=snapshot.snapshot_ref,
            build_hash=manifest.build_hash,
            source_pin=source_pin,
            report_pin=_pin("receivables_report", report),
            answer_ref=answer.artifact_ref if answer else None,
            mapping=mapping,
            available_columns=source.headers,
            checks=checks,
            stages=(
                "source_read",
                "fields_identified",
                "ontology_built",
                "checks_executed",
            )
            + (("answer_computed",) if answer else ()),
            governance_state="candidate",
            business_definition_status=definition_status,
        )
        try:
            self._store.put_many_with_integrity(
                artifacts=(source, snapshot, report, manifest, run)
                + ((answer,) if answer else ()),
                snapshot_integrities=(integrity,),
            )
        except ValueError as exc:
            if str(exc) != "ontology artifacts are append-only; create a successor ref":
                raise
            # Another request may have atomically published the exact input while
            # this request computed it. Return that retained result after checks.
            winner = self._store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=run_ref,
                contract_type=ReceivablesRunV1,
            )
            if winner is None or winner.request_hash != request_hash:
                raise
            return self.get_run(ctx=ctx, run_ref=run_ref)

        return run

    def _check(self, source, mapping, request, ctx, *, definition_override=None):
        checks = [
            ReceivablesCheck(
                check_id="source_integrity",
                status="passed",
                message="Full original bytes, parser profile, canonical IR and row count pinned.",
            )
        ]

        def check(key, status, message, rows=(), blocks="all"):
            affected_rows = set(rows)
            refs = tuple(
                e.evidence_ref
                for e in source.evidence
                if e.provenance["source_locator"]
                .get("nativeMetadata", {})
                .get("row_number")
                in affected_rows
            )
            checks.append(
                ReceivablesCheck(
                    check_id=key,
                    status=status,
                    message=message,
                    row_numbers=tuple(rows),
                    evidence_refs=refs,
                    blocks=blocks,
                )
            )

        required = ("customer_id", "invoice_id", "outstanding_balance", "currency")
        missing = [field for field in required if field not in mapping]
        check(
            "field_semantics",
            "failed" if missing else "passed",
            "Missing explicit mapping: " + ", ".join(missing)
            if missing
            else "Explicit named balance, identity and currency fields.",
            blocks="all",
        )
        confirmation = (
            request.confirmation is True
            or isinstance(request.confirmation, dict)
            and request.confirmation.get("accepted") is True
        )
        explicit = definition_override is None and all(
            field in mapping and mapping[field].casefold() in ALIASES[field]
            for field in required
        )
        definition = definition_override or (
            "confirmed"
            if confirmation
            else ("source_explicit" if explicit else "needs_confirmation")
        )
        check(
            "business_definition",
            "passed" if definition != "needs_confirmation" else "unavailable",
            "Positive balance means outstanding; due date before snapshot means overdue. Definition basis: "
            + definition,
            blocks="all",
        )
        index = {key: source.headers.index(value) for key, value in mapping.items()}
        snapshot_date = request.snapshot_date
        if snapshot_date is None and "snapshot_date" in index:
            raw_dates = {row[index["snapshot_date"]] for row in source.rows}
            if len(raw_dates) == 1:
                try:
                    snapshot_date = date.fromisoformat(next(iter(raw_dates)))
                except ValueError:
                    pass
        check(
            "snapshot_date",
            "passed" if snapshot_date else "unavailable",
            "Explicit source snapshot date required."
            if not snapshot_date
            else "Answer is bounded to " + snapshot_date.isoformat(),
            blocks="all",
        )
        if missing:
            return tuple(checks), (), definition, snapshot_date
        (
            invalid_balance,
            invalid_currency,
            invalid_keys,
            duplicates,
            negative,
            invalid_due,
            name_conflicts,
            invalid_snapshot,
        ) = [], [], [], [], [], [], [], []
        seen, names, invoices = set(), {}, []
        evidence_by_row = {}
        for evidence in source.evidence:
            row = evidence.provenance["source_locator"]["nativeMetadata"]["row_number"]
            evidence_by_row.setdefault(row, []).append(evidence.evidence_ref)
        for number, row in zip(source.row_numbers, source.rows):
            values = {key: row[col] for key, col in index.items()}
            invoice_id, customer_id = values["invoice_id"], values["customer_id"]
            if not invoice_id or not customer_id:
                invalid_keys.append(number)
            if invoice_id in seen:
                duplicates.append(number)
            seen.add(invoice_id)
            name = values.get("customer_name") or None
            if customer_id in names and names[customer_id] != name:
                name_conflicts.append(number)
            names[customer_id] = name
            balance = None
            raw = values["outstanding_balance"]
            if (
                re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]{1,2})?|\.[0-9]{1,2})", raw)
                and len(raw) <= 40
            ):
                try:
                    balance = Decimal(raw)
                    if not balance.is_finite():
                        balance = None
                except InvalidOperation:
                    pass
            if balance is None:
                invalid_balance.append(number)
            elif balance < 0:
                negative.append(number)
            currency = values["currency"]
            if not re.fullmatch("[A-Z]{3}", currency):
                invalid_currency.append(number)
            due = None
            if values.get("due_date"):
                try:
                    due = date.fromisoformat(values["due_date"])
                except ValueError:
                    invalid_due.append(number)
            else:
                invalid_due.append(number)
            if "snapshot_date" in values and snapshot_date:
                if values["snapshot_date"] != snapshot_date.isoformat():
                    invalid_snapshot.append(number)
            if balance is not None:
                positive = max(balance, Decimal(0))
                overdue = (
                    positive
                    if due and snapshot_date and due < snapshot_date
                    else Decimal(0)
                )
                invoices.append(
                    ReceivablesInvoice(
                        invoice_id=invoice_id,
                        customer_id=customer_id,
                        customer_name=name,
                        currency=currency,
                        outstanding_balance=_money(positive),
                        overdue_balance=_money(overdue)
                        if due and snapshot_date
                        else None,
                        due_date=due,
                        row_number=number,
                        sheet_name=source.sheet_name,
                        node_ref="Invoice:"
                        + content_hash(
                            {
                                "build_source": source.artifact_ref,
                                "invoice_id": invoice_id,
                                "row": number,
                            }
                        ),
                        evidence_refs=tuple(evidence_by_row[number]),
                    )
                )
        for key, rows, message in (
            (
                "identity_keys",
                invalid_keys,
                "Customer and invoice keys must be nonempty.",
            ),
            (
                "invoice_uniqueness",
                duplicates,
                "Invoice keys must be unique within this source.",
            ),
            (
                "customer_consistency",
                name_conflicts,
                "One customer key must identify one customer name.",
            ),
            (
                "balances",
                invalid_balance,
                "Every balance must be a finite Decimal with at most two fractional digits.",
            ),
            (
                "currency",
                invalid_currency,
                "Every currency must be an explicit three-letter uppercase currency code.",
            ),
            (
                "credits",
                negative,
                "Negative balances/credits require a separate explicit credit policy; no complete aggregate is produced.",
            ),
            (
                "snapshot_consistency",
                invalid_snapshot,
                "Source snapshot dates must equal the requested snapshot date.",
            ),
        ):
            check(key, "failed" if rows else "passed", message, rows)
        check(
            "question_representability",
            "passed",
            "Customer/currency grouping, Decimal balance and snapshot predicate are represented by the explicit schema.",
            blocks="all",
        )
        check(
            "due_dates",
            "unavailable" if invalid_due or "due_date" not in mapping else "passed",
            "All due dates are needed for complete overdue totals; outstanding remains available.",
            invalid_due,
            blocks="overdue",
        )
        return tuple(checks), tuple(invoices), definition, snapshot_date

    def _graph(self, ctx, source, invoices, build_ref, predecessor, now, checks):
        source_evidence = tuple(e.evidence_ref for e in source.evidence)
        if not source_evidence:
            raise ValueError("SOURCE_EVIDENCE_REQUIRED")
        nodes = [
            ExploratoryGraphNodeV1(
                node_id="SourceSnapshot:" + source.artifact_hash[7:],
                node_kind="SourceSnapshot",
                label=source.original_name,
                evidence_refs=source_evidence,
            )
        ]
        for kind in ("Customer", "Invoice", "SourceSnapshot"):
            nodes.append(
                ExploratoryGraphNodeV1(
                    node_id="type:" + kind,
                    node_kind="descriptive_type",
                    label=kind,
                    evidence_refs=source_evidence[:1],
                )
            )
        customer_refs = {}
        edges = []
        for invoice in invoices:
            customer_ref = "Customer:" + content_hash(
                {"source": source.artifact_ref, "customer_id": invoice.customer_id}
            )
            if customer_ref not in customer_refs:
                customer_refs[customer_ref] = invoice.evidence_refs
                nodes.append(
                    ExploratoryGraphNodeV1(
                        node_id=customer_ref,
                        node_kind="Customer",
                        label=invoice.customer_name
                        or invoice.customer_id
                        or "Missing customer key",
                        evidence_refs=invoice.evidence_refs,
                    )
                )
            nodes.append(
                ExploratoryGraphNodeV1(
                    node_id=invoice.node_ref,
                    node_kind="Invoice",
                    label=invoice.invoice_id or "Missing invoice key",
                    evidence_refs=invoice.evidence_refs,
                )
            )
            for relation, target in (
                ("billed_to", customer_ref),
                ("observed_in", nodes[0].node_id),
            ):
                edges.append(
                    ExploratoryGraphEdgeV1(
                        edge_id=relation + ":" + invoice.node_ref,
                        source_node_id=invoice.node_ref,
                        target_node_id=target,
                        relation_type=relation,
                        evidence_refs=invoice.evidence_refs,
                    )
                )
        previous = None
        if predecessor:
            pin = next(
                p
                for p in predecessor.child_artifacts
                if p.artifact_kind == "exploratory_graph_snapshot"
            )
            previous, _ = self._store.get_verified_exploratory_snapshot(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                pin=pin,
            )
            if (
                previous.source_identity_ref != source.source_identity_ref
                and not predecessor.build_ref.startswith("ontology-build:receivables:")
            ):
                raise ValueError("RECEIVABLES_PREDECESSOR_SOURCE_MISMATCH")
        basis = dict(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            build_ref=build_ref,
            source_identity_ref=source.source_identity_ref,
            source_hash=source.source_hash,
            canonical_source_ir_ref=source.canonical_source_ir_pin.artifact_ref,
            canonical_source_ir_hash=source.canonical_source_ir_pin.artifact_hash,
            authority_state="provisional",
            completeness_state="bounded_complete"
            if all(c.status == "passed" for c in checks)
            else "incomplete",
            nodes=nodes,
            edges=edges,
            unresolved_gap_ids=tuple(
                "gap:" + c.check_id for c in checks if c.status != "passed"
            ),
            predecessor_snapshot_ref=previous.snapshot_ref if previous else None,
            predecessor_snapshot_hash=previous.snapshot_hash if previous else None,
            change_cause="schema_rebuild" if previous else "initial_exploration",
            triggering_ref=source.artifact_ref,
            created_at=now,
        )
        from pydantic_core import to_jsonable_python

        digest = content_hash(
            {
                "schema_version": "exploratory_graph_snapshot.v1",
                **to_jsonable_python(basis),
            }
        )
        return ExploratoryGraphSnapshotV1(
            snapshot_ref="exploratory-snapshot:" + digest,
            snapshot_hash="sha256:" + digest,
            **basis,
        )

    def _structure_check(self, source, snapshot, invoices):
        evidence = {e.evidence_ref for e in source.evidence}
        node_ids = {n.node_id for n in snapshot.nodes}
        kinds = {n.node_id: n.node_kind for n in snapshot.nodes}
        valid = len(node_ids) == len(snapshot.nodes) and len(
            {e.edge_id for e in snapshot.edges}
        ) == len(snapshot.edges)
        valid = valid and len(invoices) == len(source.rows)
        for node in snapshot.nodes:
            valid = (
                valid
                and bool(node.evidence_refs)
                and set(node.evidence_refs).issubset(evidence)
            )
        edges_by_source_relation = {}
        for edge in snapshot.edges:
            edges_by_source_relation.setdefault(
                (edge.source_node_id, edge.relation_type), []
            ).append(edge)
            valid = (
                valid
                and edge.source_node_id in node_ids
                and edge.target_node_id in node_ids
                and bool(edge.evidence_refs)
                and set(edge.evidence_refs).issubset(evidence)
            )
        for invoice in invoices:
            billed = edges_by_source_relation.get((invoice.node_ref, "billed_to"), [])
            observed = edges_by_source_relation.get(
                (invoice.node_ref, "observed_in"), []
            )
            customer_ref = "Customer:" + content_hash(
                {"source": source.artifact_ref, "customer_id": invoice.customer_id}
            )
            valid = (
                valid
                and kinds.get(invoice.node_ref) == "Invoice"
                and len(billed) == 1
                and len(observed) == 1
            )
            valid = (
                valid
                and bool(billed)
                and billed[0].target_node_id == customer_ref
                and kinds.get(customer_ref) == "Customer"
            )
            valid = (
                valid
                and bool(observed)
                and kinds.get(observed[0].target_node_id) == "SourceSnapshot"
            )
        retained_rows = {
            e.provenance["source_locator"]["nativeMetadata"]["row_number"]
            for e in source.evidence
        }
        valid = valid and set(source.row_numbers).issubset(retained_rows)
        return ReceivablesCheck(
            check_id="structure",
            status="passed" if valid else "failed",
            message="Typed endpoints, one billed_to/observed_in per invoice, row/evidence closure, and source scope checked.",
            blocks="all",
        )

    def _answer(
        self,
        ctx,
        source,
        report,
        invoices,
        snapshot,
        run_ref,
        snapshot_date,
        overdue_available,
        now,
        build_hash,
    ):
        groups, currencies = {}, {}
        with localcontext() as precision:
            precision.prec = 60
            for invoice in invoices:
                key = invoice.customer_id, invoice.currency
                group = groups.setdefault(
                    key,
                    {
                        "balance": Decimal(0),
                        "overdue": Decimal(0),
                        "count": 0,
                        "positive": 0,
                        "refs": [],
                        "overdue_refs": [],
                        "name": invoice.customer_name,
                    },
                )
                total = currencies.setdefault(
                    invoice.currency,
                    {
                        "balance": Decimal(0),
                        "overdue": Decimal(0),
                        "positive": 0,
                        "refs": [],
                        "overdue_refs": [],
                    },
                )
                balance, overdue = (
                    Decimal(invoice.outstanding_balance),
                    Decimal(invoice.overdue_balance or "0"),
                )
                group["count"] += 1
                for item in (group, total):
                    item["balance"] += balance
                    item["overdue"] += overdue
                    item["positive"] += int(balance > 0)
                    if balance > 0:
                        item["refs"].extend(invoice.evidence_refs)
                    if overdue > 0 and overdue_available:
                        item["overdue_refs"].extend(invoice.evidence_refs)
            customers = tuple(
                ReceivablesCustomer(
                    customer_id=customer_id,
                    customer_name=g["name"],
                    currency=currency,
                    outstanding_balance=_money(g["balance"]),
                    overdue_balance=_money(g["overdue"]) if overdue_available else None,
                    invoice_count=g["count"],
                    outstanding_invoice_count=g["positive"],
                    contributing_row_refs=tuple(g["refs"]),
                    overdue_contributing_row_refs=tuple(g["overdue_refs"]),
                    node_ref="Customer:"
                    + content_hash(
                        {"source": source.artifact_ref, "customer_id": customer_id}
                    ),
                )
                for (customer_id, currency), g in sorted(groups.items())
            )
            totals = tuple(
                ReceivablesCurrencyTotal(
                    currency=currency,
                    outstanding_balance=_money(g["balance"]),
                    overdue_balance=_money(g["overdue"]) if overdue_available else None,
                    outstanding_invoice_count=g["positive"],
                    contributing_row_refs=tuple(g["refs"]),
                    overdue_contributing_row_refs=tuple(g["overdue_refs"]),
                )
                for currency, g in sorted(currencies.items())
            )
        return make_artifact(
            ReceivablesAnswerV1,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            created_at=now,
            run_ref=run_ref,
            build_ref=snapshot.build_ref,
            snapshot_ref=snapshot.snapshot_ref,
            build_hash=build_hash,
            source_pin=_pin("receivables_source", source),
            report_pin=_pin("receivables_report", report),
            snapshot_date=snapshot_date,
            source_name=source.original_name,
            governance_state="candidate",
            business_definition_status=report.business_definition_status,
            row_count=len(source.rows),
            outstanding_invoice_count=sum(t.outstanding_invoice_count for t in totals),
            overdue_available=overdue_available,
            customers=customers,
            invoices=invoices,
            totals=totals,
        )

    def get_run(self, *, ctx, run_ref):
        self._access(ctx)
        run = self._get(ctx, run_ref, ReceivablesRunV1)
        source = self._get(ctx, run.source_pin.artifact_ref, ReceivablesSourceV1)
        if source.artifact_hash != run.source_pin.artifact_hash:
            raise ValueError("RECEIVABLES_SOURCE_PIN_MISMATCH")
        self._verify_source(ctx, source)
        manifest, _, _ = self._verify_build(
            ctx, run.build_ref, run.report_pin, run.snapshot_ref
        )
        if manifest.build_hash != run.build_hash:
            raise ValueError("RECEIVABLES_BUILD_PIN_MISMATCH")
        return run

    def _verify_build(self, ctx, build_ref, report_pin, snapshot_ref):
        manifest = self._get(ctx, build_ref, OntologyBuildManifestV1)
        if manifest.build_hash != "sha256:" + content_hash(
            manifest.model_dump(
                mode="json", exclude={"build_ref", "build_hash", "schema_version"}
            )
        ):
            raise ValueError("RECEIVABLES_MANIFEST_HASH_MISMATCH")
        report = self._get(ctx, report_pin.artifact_ref, ReceivablesReportV1)
        if (
            report.artifact_hash != report_pin.artifact_hash
            or report.build_ref != build_ref
            or report_pin not in manifest.child_artifacts
        ):
            raise ValueError("RECEIVABLES_REPORT_PIN_MISMATCH")
        pin = next(
            p
            for p in manifest.child_artifacts
            if p.artifact_kind == "exploratory_graph_snapshot"
        )
        snapshot, _ = self._store.get_verified_exploratory_snapshot(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, pin=pin
        )
        if snapshot.snapshot_ref != snapshot_ref or snapshot.build_ref != build_ref:
            raise ValueError("RECEIVABLES_BUILD_PIN_MISMATCH")
        source = self._get(ctx, report.source_pin.artifact_ref, ReceivablesSourceV1)
        if (
            source.artifact_hash != report.source_pin.artifact_hash
            or report.source_pin not in manifest.child_artifacts
            or report.snapshot_pin != pin
            or report.canonical_source_ir_pin != source.canonical_source_ir_pin
            or report.parser_profile_hash != source.parser_profile_hash
            or snapshot.source_hash != source.source_hash
            or snapshot.source_identity_ref != source.source_identity_ref
            or snapshot.canonical_source_ir_hash
            != source.canonical_source_ir_pin.artifact_hash
            or report.mapping_hash != "sha256:" + content_hash(report.mapping)
            or report.schema_hash
            != "sha256:"
            + content_hash(
                {
                    "types": ["Customer", "Invoice", "SourceSnapshot"],
                    "fields": report.mapping,
                    "relations": ["billed_to", "observed_in"],
                    "recipe": "receivables.v1",
                    **(
                        {"recipe_hash": report.recipe_hash}
                        if report.recipe_hash
                        else {}
                    ),
                }
            )
        ):
            raise ValueError("RECEIVABLES_VERIFICATION_BASIS_MISMATCH")
        return manifest, report, snapshot

    def get_answer(self, *, ctx, answer_ref):
        self._access(ctx)
        answer = self._get(ctx, answer_ref, ReceivablesAnswerV1)
        run = self.get_run(ctx=ctx, run_ref=answer.run_ref)
        if (
            run.answer_ref != answer.artifact_ref
            or run.source_pin != answer.source_pin
            or run.report_pin != answer.report_pin
            or run.snapshot_ref != answer.snapshot_ref
        ):
            raise ValueError("RECEIVABLES_ANSWER_BASIS_MISMATCH")
        source = self._get(ctx, answer.source_pin.artifact_ref, ReceivablesSourceV1)
        manifest, report, snapshot = self._verify_build(
            ctx, answer.build_ref, answer.report_pin, answer.snapshot_ref
        )
        checks, invoices, definition, snapshot_date = self._check(
            source,
            report.mapping,
            run.request,
            ctx,
            definition_override=report.business_definition_status,
        )
        checks = checks + (self._structure_check(source, snapshot, invoices),)
        replay = self._answer(
            ctx,
            source,
            report,
            invoices,
            snapshot,
            answer.run_ref,
            snapshot_date,
            answer.overdue_available,
            answer.created_at,
            manifest.build_hash,
        )
        if (
            replay != answer
            or checks != report.checks
            or definition != report.business_definition_status
        ):
            raise ValueError("RECEIVABLES_REPLAY_MISMATCH")
        return answer

    def get_source_by_identity(self, *, ctx, source_identity_ref):
        self._access(ctx)
        matches = []
        for pin in self._store.list_artifact_pins(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            schema_version="receivables_source.v1",
        ):
            source = self._get(ctx, pin.artifact_ref, ReceivablesSourceV1)
            if source.source_identity_ref == source_identity_ref:
                if source.artifact_hash != pin.artifact_hash:
                    raise ValueError("RECEIVABLES_SOURCE_PIN_MISMATCH")
                matches.append(source)
        if len(matches) != 1:
            raise ValueError("RECEIVABLES_SOURCE_NOT_FOUND")
        self._verify_source(ctx, matches[0])
        return matches[0]

    def get_run_basis(self, *, ctx, run):
        current = self.get_run(ctx=ctx, run_ref=run.artifact_ref)
        if current != run:
            raise ValueError("RECEIVABLES_RUN_PIN_MISMATCH")
        source = self._get(ctx, current.source_pin.artifact_ref, ReceivablesSourceV1)
        manifest, report, snapshot = self._verify_build(
            ctx, current.build_ref, current.report_pin, current.snapshot_ref
        )
        return {
            "source": source,
            "report": report,
            "manifest": manifest,
            "snapshot": snapshot,
        }

    def answer_state(self, *, ctx, answer):
        self._access(ctx)
        # Successor discovery is owner-controlled, bounded to the current scope.
        successors = []
        for pin in self._store.list_artifact_pins(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            schema_version="ontology_build_manifest.v1",
        ):
            manifest = self._get(ctx, pin.artifact_ref, OntologyBuildManifestV1)
            if manifest.predecessor_build_ref == answer.build_ref:
                successors.append(manifest.build_ref)
        state = {
            "stale": bool(successors),
            "model_changed": bool(successors),
            "successor_build_refs": sorted(successors),
        }
        if successors:
            runs = []
            for pin in self._store.list_artifact_pins(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                schema_version="receivables_run.v1",
            ):
                run = self._get(ctx, pin.artifact_ref, ReceivablesRunV1)
                if run.build_ref in successors:
                    runs.append(run)
            if runs:
                latest = max(runs, key=lambda run: (run.created_at, run.artifact_ref))
                self.get_run(ctx=ctx, run_ref=latest.artifact_ref)
                state["refresh_request"] = latest.request.model_copy(
                    update={"predecessor_build_ref": latest.build_ref}
                ).model_dump(mode="json")
        return state
