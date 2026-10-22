"""Immutable workspace store for profile relation-type proposals."""

from __future__ import annotations

import json
from pathlib import Path
import re

from contracts.profile_relation_type_proposal import ProfileRelationTypeProposalV1
from storage.atomic_json_artifact import publish_immutable_json


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROPOSAL_REF = re.compile(r"^profile-relation-type-proposal:[0-9a-f]{24}$")


class ProfileRelationTypeProposalStore:
    __slots__ = (
        "_sealed",
        "workspace_id",
        "organization_id",
        "root",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("profile relation proposal store ownership is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        workspace_id: str,
        organization_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        for field, value in (
            ("workspace_id", workspace_id),
            ("organization_id", organization_id),
        ):
            if value is None and field == "organization_id":
                continue
            if (
                not isinstance(value, str)
                or not value.isascii()
                or _SAFE_SEGMENT.fullmatch(value) is None
                or value in {".", ".."}
            ):
                raise ValueError(
                    f"{field} must be a safe single-segment ASCII identifier"
                )
        root = Path(workspace_root).resolve()
        if root.name == workspace_id and root.parent.name == "workspaces":
            workspaces_root = root.parent.resolve()
            workspace_path = root
        else:
            workspaces_root = (root / "workspaces").resolve()
            workspace_path = (workspaces_root / workspace_id).resolve()
        if (
            workspace_path.parent != workspaces_root
            or not workspace_path.is_relative_to(workspaces_root)
        ):
            raise ValueError("workspace path escapes governed workspace root")
        artifact_root = (workspace_path / "artifacts").resolve()
        proposal_root = (
            artifact_root / "profile-relation-type-proposals"
        ).resolve()
        if not proposal_root.is_relative_to(artifact_root):
            raise ValueError("proposal root escapes governed workspace artifacts")
        self.workspace_id = workspace_id
        self.organization_id = organization_id
        self.root = proposal_root
        object.__setattr__(self, "_sealed", True)

    def put(
        self,
        proposal: ProfileRelationTypeProposalV1,
        *,
        organization_id: str | None = None,
    ) -> dict:
        expected_organization = self.organization_id or organization_id
        if expected_organization is None:
            raise ValueError("proposal store organization ownership is required")
        if proposal.workspace_id != self.workspace_id:
            raise ValueError("proposal store workspace ownership mismatch")
        if proposal.organization_id != expected_organization:
            raise ValueError("proposal store organization ownership mismatch")
        digest = proposal.proposal_hash.removeprefix("sha256:")
        target = (
            self.root / digest / "profile_relation_type_proposal.v1.json"
        ).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("proposal artifact path escapes governed workspace")
        persisted = publish_immutable_json(
            target,
            proposal.as_dict(),
            conflict_error="PROFILE_RELATION_TYPE_PROPOSAL_ARTIFACT_CONFLICT",
        )
        return ProfileRelationTypeProposalV1.model_validate(persisted).as_dict()

    def get(
        self,
        *,
        proposal_ref: str,
        proposal_hash: str,
        organization_id: str | None = None,
    ) -> dict:
        if (
            not isinstance(proposal_hash, str)
            or _SHA256_REF.fullmatch(proposal_hash) is None
        ):
            raise ValueError("proposal_hash must be an exact lowercase sha256 ref")
        if (
            not isinstance(proposal_ref, str)
            or _PROPOSAL_REF.fullmatch(proposal_ref) is None
        ):
            raise ValueError("proposal_ref is invalid")
        expected_organization = self.organization_id or organization_id
        if expected_organization is None:
            raise ValueError("proposal store organization ownership is required")
        target = (
            self.root
            / proposal_hash.removeprefix("sha256:")
            / "profile_relation_type_proposal.v1.json"
        ).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("proposal artifact path escapes governed workspace")
        if not target.is_file():
            raise ValueError(f"profile relation type proposal not found: {proposal_hash}")
        proposal = ProfileRelationTypeProposalV1.model_validate_json(
            target.read_text(encoding="utf-8")
        )
        if proposal.proposal_ref != proposal_ref or proposal.proposal_hash != proposal_hash:
            raise ValueError("profile relation type proposal identity mismatch")
        if proposal.workspace_id != self.workspace_id:
            raise ValueError("proposal store workspace ownership mismatch")
        if proposal.organization_id != expected_organization:
            raise ValueError("proposal store organization ownership mismatch")
        return proposal.as_dict()

    def find_for_graph_intent(
        self,
        *,
        graph_intent_ref: str,
        graph_intent_hash: str,
        organization_id: str | None = None,
    ) -> dict | None:
        """Resolve one exact persisted proposal for a governed GraphIntent."""

        expected_organization = self.organization_id or organization_id
        if expected_organization is None:
            raise ValueError("proposal store organization ownership is required")
        matches: list[dict] = []
        if self.root.is_dir():
            for target in sorted(
                self.root.glob("*/profile_relation_type_proposal.v1.json")
            ):
                proposal = ProfileRelationTypeProposalV1.model_validate_json(
                    target.read_text(encoding="utf-8")
                )
                binding = proposal.graph_intent_binding
                if (
                    binding["graph_intent_ref"] == graph_intent_ref
                    and binding["graph_intent_hash"] == graph_intent_hash
                ):
                    matches.append(proposal.as_dict())
        if len(matches) > 1:
            raise ValueError("multiple profile relation proposals bind GraphIntent")
        if not matches:
            return None
        proposal = matches[0]
        if proposal["workspace_id"] != self.workspace_id:
            raise ValueError("proposal store workspace ownership mismatch")
        if proposal["organization_id"] != expected_organization:
            raise ValueError("proposal store organization ownership mismatch")
        return proposal


__all__ = ["ProfileRelationTypeProposalStore"]
