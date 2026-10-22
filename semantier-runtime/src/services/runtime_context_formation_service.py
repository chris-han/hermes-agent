"""Advisory formation and bounded, independently trusted historical verification."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import ValidationError

from contracts import runtime_context_reconstruction as rr
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import canonical_json
from services._runtime_context_advisory.kernel import (
    verify_advisory_capture_v1,
    _authenticate_advisory_package,
)
from services._runtime_context_advisory.operator import (
    build_advisory_episode_representation,
)
from services import runtime_context_reconstruction_owners as owners
from storage.runtime_working_context_store import (
    RuntimeContextError,
    RuntimeTaskContextStore,
    _with_context_error_origin,
)

_MAX_SERIALIZED_BYTES = 48 * 1024 * 1024
_MAX_DECODED_BYTES = 32 * 1024 * 1024


def _translate(exc: ValueError) -> RuntimeContextError:
    if isinstance(exc, RuntimeContextError):
        if exc._origin is not None:
            return exc
        if exc.code in {"FORBIDDEN_ACTOR", "FORBIDDEN_SESSION", "MISSING_WORKSPACE"}:
            return RuntimeContextError(
                "FORBIDDEN_SCOPE", _origin="AUTHORITY_SESSION"
            )
        if exc.code == "SESSION_NOT_FOUND":
            return RuntimeContextError("NOT_FOUND", _origin="AUTHORITY_SESSION")
        return exc
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if len(errors) == 1 and errors[0]["loc"] == ("verifier_package",):
            cause = errors[0].get("ctx", {}).get("error")
            if type(cause) is rr._RetainedResourceMissingError:
                return RuntimeContextError("INCOMPLETE_CLOSURE")
            if type(cause) is rr._RetainedResourceLengthError:
                return RuntimeContextError("HASH_MISMATCH")
        if any("scope mismatch" in item["msg"] for item in exc.errors()):
            return RuntimeContextError("FORBIDDEN_SCOPE")
        identity = any(
            "identity" in item["msg"]
            or "hash mismatch" in item["msg"]
            or "bytes_hash" in item["msg"]
            for item in exc.errors()
        )
        return RuntimeContextError("HASH_MISMATCH" if identity else "INVALID_STATE")
    code = str(exc)
    kernel_codes = {
        "full canonical JSON identity does not bind payload bytes": "HASH_MISMATCH",
        "self-bound envelope identity does not bind exact content": "HASH_MISMATCH",
        "capture selector does not name its exact registry entry": "UNSUPPORTED_SOURCE",
        "capture schema does not match the selected registry entry": "UNSUPPORTED_SOURCE",
        "profile capture differs from the authenticated profile": "HASH_MISMATCH",
        "operator descriptor differs from authenticated product values": "HASH_MISMATCH",
        "sources exceed the 2048 capture-node bound": "RESOURCE_LIMIT_EXCEEDED",
        "capture dependency depth exceeds 32": "RESOURCE_LIMIT_EXCEEDED",
        "capture closure is missing an exact dependency": "INCOMPLETE_CLOSURE",
    }
    if code in kernel_codes:
        return RuntimeContextError(kernel_codes[code])
    if code in {
        "INVALID_STATE",
        "INCOMPLETE_CLOSURE",
        "UNSUPPORTED_SOURCE",
        "UNSUPPORTED_VERIFIER_REGISTRY",
        "HASH_MISMATCH",
        "REQUEST_CONFLICT",
        "RESOURCE_LIMIT_EXCEEDED",
        "NOT_FOUND",
        "FORBIDDEN_SCOPE",
    }:
        return RuntimeContextError(code)
    if "SCOPE" in code or "ACTOR" in code:
        return RuntimeContextError("FORBIDDEN_SCOPE")
    if "UNAVAILABLE" in code or "MISSING" in code:
        return RuntimeContextError("INCOMPLETE_CLOSURE")
    if "MISMATCH" in code or "CHANGED" in code:
        return RuntimeContextError("HASH_MISMATCH")
    return RuntimeContextError("INVALID_STATE")


def _revalidate(model, value):
    if type(value) is not model:
        raise RuntimeContextError("INVALID_STATE")
    try:
        return model.model_validate(value.model_dump(mode="json"))
    except ValueError as exc:
        raise _translate(exc) from exc


def _bind(model, **payload):
    return model.model_validate(
        rr.bind_envelope_identity(
            payload,
            prefix=model.identity_prefix,
            ref_field=model.identity_ref_field,
            hash_field=model.identity_hash_field,
        )
    )


def verify_advisory_formation_v1(*, raw: bytes) -> rr.AdvisoryContextFormationV1:
    """Verify saved bytes without source IO, reconstruction or release selection."""
    if type(raw) is not bytes:
        raise RuntimeContextError("INVALID_STATE")
    if len(raw) > _MAX_SERIALIZED_BYTES:
        raise RuntimeContextError("RESOURCE_LIMIT_EXCEEDED")
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise RuntimeContextError("INVALID_STATE") from exc
    try:
        if not isinstance(payload, dict):
            raise RuntimeContextError("INVALID_STATE")
        # Account before model validators decode base64. Missing fields take the
        # normal strict DTO path rather than dereferencing constructed packages.
        package = payload.get("verifier_package")
        sources = payload.get("sources")
        representation = payload.get("representation")
        if (
            not isinstance(package, dict)
            or not isinstance(sources, list)
            or not isinstance(representation, dict)
        ):
            raise RuntimeContextError("INVALID_STATE")
        resources = package.get("resources")
        registry = package.get("registry")
        if not isinstance(resources, list) or not isinstance(registry, dict):
            raise RuntimeContextError("INVALID_STATE")
        if len(sources) > 2048:
            raise RuntimeContextError("RESOURCE_LIMIT_EXCEEDED")
        total = 0
        for item in [
            registry,
            representation,
            *(source.get("payload") for source in sources if isinstance(source, dict)),
        ]:
            if (
                not isinstance(item, dict)
                or type(item.get("canonical_json")) is not str
            ):
                raise RuntimeContextError("INVALID_STATE")
            total += len(item["canonical_json"].encode("utf-8"))
        for resource in resources:
            if (
                not isinstance(resource, dict)
                or type(resource.get("bytes_base64")) is not str
            ):
                raise RuntimeContextError("INVALID_STATE")
            encoded = resource["bytes_base64"]
            total += len(encoded) // 4 * 3 - (
                2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
            )
            if total > _MAX_DECODED_BYTES:
                raise RuntimeContextError("RESOURCE_LIMIT_EXCEEDED")
        if total > _MAX_DECODED_BYTES:
            raise RuntimeContextError("RESOURCE_LIMIT_EXCEEDED")
        formation = rr.AdvisoryContextFormationV1.model_validate(payload)
        _, trusted = owners._require_advisory_registration_for_pin(
            pin=formation.verifier_registry
        )
        _authenticate_advisory_package(
            package=formation.verifier_package,
            registry_pin=formation.verifier_registry,
            trusted=trusted,
        )
        keys = {rr._pin_key(source.source) for source in formation.sources}
        required = [
            formation.profile,
            formation.operator,
            *(
                pin
                for pair in formation.request.outcomes
                for pin in (pair.episode, pair.assessment)
            ),
            *(pin for source in formation.sources for pin in source.dependencies),
        ]
        if any(
            pin.artifact_kind
            in {"frozen_context_formation", "runtime_context_formation_command"}
            for pin in required
        ):
            raise RuntimeContextError("INVALID_STATE")
        if any(rr._pin_key(pin) not in keys for pin in required):
            raise RuntimeContextError("INCOMPLETE_CLOSURE")
        verify_advisory_capture_v1(
            outcomes=tuple(
                (pair.episode, pair.assessment) for pair in formation.request.outcomes
            ),
            profile=formation.profile,
            operator=formation.operator,
            verifier_registry=formation.verifier_registry,
            sources=formation.sources,
            representation=formation.representation,
            verifier_package=formation.verifier_package,
            trusted=trusted,
        )
        return formation
    except ValueError as exc:
        raise _translate(exc) from exc


def _verify_advisory_triple(*, actor_context, command, formation, receipt):
    command = _revalidate(rr.AdvisoryContextCommandV1, command)
    receipt = _revalidate(rr.AdvisoryContextReceiptV1, receipt)
    if type(formation) is not rr.AdvisoryContextFormationV1:
        raise RuntimeContextError("INVALID_STATE")
    formation = verify_advisory_formation_v1(
        raw=canonical_json(formation.model_dump(mode="json")).encode("utf-8")
    )
    identity = ("organization_id", "workspace_id", "actor_id")
    if any(
        getattr(value, field) != getattr(actor_context, field)
        for value in (command, formation, receipt)
        for field in identity
    ):
        raise RuntimeContextError("FORBIDDEN_SCOPE")
    if any(value.session_ref != command.session_ref for value in (formation, receipt)):
        raise RuntimeContextError("INVALID_STATE")
    if (
        formation.request != command.request
        or receipt.command != rr.artifact_pin_for(command)
        or receipt.formation != rr.artifact_pin_for(formation)
        or receipt.request_id != command.request.request_id
        or receipt.request_hash != command.command_hash
        or receipt.recorded_at != formation.recorded_at
    ):
        raise RuntimeContextError("INVALID_STATE")
    return command, formation, receipt


def _current_advisory_scope(*, context, actor_context, session_ref):
    try:
        context.authority_resolver.require_current_actor(actor_context=actor_context)
        canonical = context.session_resolver.resolve(
            actor_context=actor_context,
            session_ref=session_ref,
        )
    except RuntimeContextError as exc:
        raise _with_context_error_origin(
            _translate(exc), "AUTHORITY_SESSION"
        ) from exc
    if canonical != context.session_ref or actor_context != context.actor_context:
        raise RuntimeContextError(
            "FORBIDDEN_SCOPE", _origin="AUTHORITY_SESSION"
        )
    return canonical


def _source_access_error(*, context, actor_context, session_ref, error):
    error = _translate(error)
    if error._origin is not None:
        return error
    if error.code == "FORBIDDEN_SCOPE":
        try:
            _current_advisory_scope(
                context=context,
                actor_context=actor_context,
                session_ref=session_ref,
            )
        except RuntimeContextError as authority_error:
            return authority_error
        return _with_context_error_origin(error, "ARTIFACT_LOOKUP")
    if error.code == "NOT_FOUND":
        return _with_context_error_origin(error, "ARTIFACT_LOOKUP")
    return error


def _require_bound_advisory_access(*, build_store, actor_context, session_ref, pins):
    context = getattr(build_store, "_advisory_owner_context", None)
    if context is None or context.build_store is not build_store:
        raise RuntimeContextError("INVALID_STATE")
    try:
        registry = owners.AdvisoryNativeSourceRegistryV1(context=context)
        registry._require_current_sources(
            actor_context=actor_context, session_ref=session_ref, pins=pins
        )
    except ValueError as exc:
        raise _source_access_error(
            context=context,
            actor_context=actor_context,
            session_ref=session_ref,
            error=exc,
        ) from exc
    except PermissionError as exc:
        error = RuntimeContextError("FORBIDDEN_SCOPE")
        raise _source_access_error(
            context=context,
            actor_context=actor_context,
            session_ref=session_ref,
            error=error,
        ) from exc


class RuntimeContextFormationService:
    def __init__(self, *, store: RuntimeTaskContextStore, context):
        from services.runtime_learning_trial_service import AdvisoryTrialOwnerContext

        if (
            type(store) is not RuntimeTaskContextStore
            or type(context) is not AdvisoryTrialOwnerContext
        ):
            raise RuntimeContextError("INVALID_STATE")
        if store._build_store is not context.build_store:
            raise RuntimeContextError("INVALID_STATE")
        self._store = store
        self._context = context
        self._sources = owners.AdvisoryNativeSourceRegistryV1(context=context)
        previous = getattr(context.build_store, "_advisory_owner_context", None)
        if previous is not None and previous is not context:
            raise RuntimeContextError("INVALID_STATE")
        context.build_store._advisory_owner_context = context

    def _scope(self, actor_context, session_ref):
        return _current_advisory_scope(
            context=self._context,
            actor_context=actor_context,
            session_ref=session_ref,
        )

    def _access(self, actor_context, session_ref, pins):
        try:
            self._sources._require_current_sources(
                actor_context=actor_context, session_ref=session_ref, pins=pins
            )
        except ValueError as exc:
            raise _source_access_error(
                context=self._context,
                actor_context=actor_context,
                session_ref=session_ref,
                error=exc,
            ) from exc
        except PermissionError as exc:
            error = RuntimeContextError("FORBIDDEN_SCOPE")
            raise _source_access_error(
                context=self._context,
                actor_context=actor_context,
                session_ref=session_ref,
                error=error,
            ) from exc

    def get_advisory(
        self,
        *,
        actor_context: RuntimeContextDTO,
        pin: ArtifactPinV1,
    ) -> rr.AdvisoryContextFormationV1:
        canonical = self._scope(actor_context, self._context.session_ref)
        formation = self._store.get_advisory_formation(
            actor_context=actor_context, pin=pin
        )
        self._access(
            actor_context,
            canonical,
            tuple(source.source for source in formation.sources),
        )
        self._scope(actor_context, canonical)
        return formation

    def form_advisory(
        self,
        *,
        actor_context: RuntimeContextDTO,
        session_ref: str,
        request: rr.AdvisoryContextRequestV1,
    ) -> rr.AdvisoryContextReceiptV1:
        request = _revalidate(rr.AdvisoryContextRequestV1, request)
        canonical = self._scope(actor_context, session_ref)
        roots = tuple(
            pin for pair in request.outcomes for pin in (pair.episode, pair.assessment)
        )
        self._access(actor_context, canonical, roots)
        identity = dict(
            organization_id=actor_context.organization_id,
            workspace_id=actor_context.workspace_id,
            actor_id=actor_context.actor_id,
            session_ref=canonical,
        )
        command = _bind(
            rr.AdvisoryContextCommandV1,
            schema_version="advisory_context_command.v1",
            **identity,
            operation="FORM_ADVISORY_CONTEXT",
            request=request,
        )
        saved = self._store.get_advisory_receipt_for_request(
            actor_context=actor_context, command=command
        )
        if saved is not None:
            self.get_advisory(actor_context=actor_context, pin=saved.formation)
            return saved
        try:
            package, trusted = owners.require_advisory_registration()
            registry = package.parsed_registry()
            profile_raw = next(
                item.decoded_bytes()
                for item in package.resources
                if item.resource_id == "advisory_profile_v1"
            )
            profile_payload = owners._captured_json(profile_raw)
            profile = ArtifactPinV1(
                artifact_kind="advisory_context_profile",
                artifact_ref="advisory-context-profile:"
                + profile_payload.bytes_hash.removeprefix("sha256:"),
                artifact_hash=profile_payload.bytes_hash,
            )
            operator_payload = owners._advisory_operator_descriptor(
                profile=profile,
                operator_resource_hash=next(
                    item.bytes_hash
                    for item in registry.resources
                    if item.resource_id == "advisory_operator_v1"
                ),
            )
            operator = ArtifactPinV1(
                artifact_kind="advisory_context_operator",
                artifact_ref="advisory-context-operator:"
                + operator_payload.bytes_hash.removeprefix("sha256:"),
                artifact_hash=operator_payload.bytes_hash,
            )
            closure = {}
            visiting = set()
            refs = {}
            total = sum(item.byte_length for item in registry.resources) + len(
                package.registry.canonical_json.encode("utf-8")
            )

            def capture(pin, depth):
                nonlocal total
                if depth > 32:
                    raise RuntimeContextError("RESOURCE_LIMIT_EXCEEDED")
                key = rr._pin_key(pin)
                if key in visiting:
                    raise RuntimeContextError("INVALID_STATE")
                if key in closure:
                    return
                ref = key[:2]
                if ref in refs and refs[ref] != key[2]:
                    raise RuntimeContextError("HASH_MISMATCH")
                refs[ref] = key[2]
                if len(refs) > 2048:
                    raise RuntimeContextError("RESOURCE_LIMIT_EXCEEDED")
                visiting.add(key)
                source = self._sources.capture(
                    actor_context=actor_context, session_ref=canonical, pin=pin
                )
                total += len(source.payload.canonical_json.encode("utf-8"))
                if total > _MAX_DECODED_BYTES:
                    raise RuntimeContextError("RESOURCE_LIMIT_EXCEEDED")
                for dependency in source.dependencies:
                    capture(dependency, depth + 1)
                visiting.remove(key)
                closure[key] = source

            for root in (*roots, profile, operator):
                capture(root, 1)
            representation = build_advisory_episode_representation(
                outcomes=tuple(
                    (pair.episode, pair.assessment) for pair in request.outcomes
                ),
                closure=closure,
            )
            formation = _bind(
                rr.AdvisoryContextFormationV1,
                schema_version="advisory_context_formation.v1",
                **identity,
                request=request,
                profile=profile,
                operator=operator,
                verifier_registry=trusted.registry_pin,
                verifier_package=package,
                sources=tuple(closure[key] for key in sorted(closure)),
                representation=representation,
                usage="CONTEXT_ONLY",
                recorded_at=datetime.now(timezone.utc),
            )
            receipt = _bind(
                rr.AdvisoryContextReceiptV1,
                schema_version="advisory_context_receipt.v1",
                **identity,
                operation="FORM_ADVISORY_CONTEXT",
                request_id=request.request_id,
                request_hash=command.command_hash,
                command=rr.artifact_pin_for(command),
                formation=rr.artifact_pin_for(formation),
                usage="CONTEXT_ONLY",
                recorded_at=formation.recorded_at,
            )

            def final_access():
                self._scope(actor_context, canonical)
                self._access(
                    actor_context,
                    canonical,
                    tuple(source.source for source in formation.sources),
                )

            return self._store.publish_advisory(
                actor_context=actor_context,
                command=command,
                formation=formation,
                receipt=receipt,
                verify_before_commit=final_access,
            )
        except ValueError as exc:
            raise _translate(exc) from exc
