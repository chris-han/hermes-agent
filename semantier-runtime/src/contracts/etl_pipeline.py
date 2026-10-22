"""ETL receipts are scoped workflow observations, never definition authority."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from eos.artifact_hashing import content_hash


class EtlCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    workflow_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    stage_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    mode: Literal["preview", "check", "run"] = "preview"
    request_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    draft_definition: dict[str, Any] | None = None
    draft_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def draft_preview_only(self):
        if (self.draft_definition is None) != (self.draft_digest is None):
            raise ValueError("ETL_DRAFT_BASIS_REQUIRED")
        if self.draft_definition is not None and self.mode != "preview":
            raise ValueError("ETL_DRAFT_PREVIEW_ONLY")
        return self


class EtlReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["etl_run_receipt.v1"] = "etl_run_receipt.v1"
    organization_id: str
    workspace_id: str
    actor_id: str
    request_id: str
    workflow_ref: str
    workflow_version: str
    workflow_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    snapshot_bytes_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    executor_version: Literal["native-etl.v1"] = "native-etl.v1"
    stage_id: str
    source_basis: list[dict[str, Any]]
    result: dict[str, Any]
    exports: dict[str, str]
    produced_at: str
    receipt_ref: str
    receipt_hash: str

    @model_validator(mode="after")
    def integrity(self):
        timestamp = datetime.fromisoformat(self.produced_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(
            timestamp
        ):
            raise ValueError("ETL_TIMESTAMP_MUST_BE_UTC")
        basis = self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})
        digest = content_hash(basis)
        if (
            self.receipt_hash != "sha256:" + digest
            or self.receipt_ref != "etl-run:" + digest
        ):
            raise ValueError("ETL_RECEIPT_HASH_MISMATCH")
        if (
            self.result.get("mode") != "run"
            or self.result.get("stage_id") != self.stage_id
            or self.result.get("executor_version") != self.executor_version
        ):
            raise ValueError("ETL_RECEIPT_BASIS_MISMATCH")
        if (
            self.result.get("status") != "completed"
            or self.result.get("full_check_status") != "passed"
        ) and self.exports:
            raise ValueError("ETL_SUCCESS_REQUIRED_FOR_EXPORT")
        return self


class EtlSourceRowCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    workflow_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    row: int = Field(ge=1, le=100100)
    sheet: str = Field(min_length=1, max_length=100)


class EtlSourceProfileCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_pin: dict[str, str]
    sheet: str | None = None
    header_row: int = Field(default=1, ge=1, le=100)
    encoding: Literal["utf-8", "utf-8-sig", "gb18030"] = "utf-8-sig"


# The versioned editor contract is closed independently of WorkflowDefinitionIR.
# Legacy configs remain untouched; only an explicit migration adopts these models.

ColumnName = Annotated[str, Field(min_length=1, pattern=r"^(?!__).+")]
ColumnList = Annotated[
    list[ColumnName], Field(min_length=1, json_schema_extra={"uniqueItems": True})
]
ColumnMapping = Annotated[dict[ColumnName, ColumnName], Field(min_length=1)]


class EtlOperationBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, regex_engine="python-re")
    operationSchemaVersion: Literal["etl_operation.v2"]

    @field_validator("*", mode="after", check_fields=False)
    @classmethod
    def unique_columns(cls, value, info):
        if info.field_name in {"columns", "keys"} and isinstance(value, list):
            if len(set(value)) != len(value):
                raise ValueError("INVALID_ETL_COLUMNS")
        return value


class EtlSourceOperation(EtlOperationBase):
    operation: Literal["source"]
    source: ColumnName
    artifact_pin: dict[str, str] | None = None
    sheet: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    header_row: Annotated[int, Field(ge=1, le=100)] = 1
    encoding: Literal["utf-8", "utf-8-sig", "gb18030"] = "utf-8-sig"


class EtlSelectOperation(EtlOperationBase):
    operation: Literal["select"]
    columns: ColumnList


class EtlRenameOperation(EtlOperationBase):
    operation: Literal["rename"]
    columns: ColumnMapping


class EtlNormalizeOperation(EtlOperationBase):
    operation: Literal["normalize"]
    columns: ColumnList


class EtlDecimalOperation(EtlOperationBase):
    operation: Literal["decimal"]
    column: ColumnName
    target: ColumnName


class EtlFilterOperation(EtlOperationBase):
    operation: Literal["filter"]
    column: ColumnName
    comparison: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    value: str
    type: Literal["string", "decimal"]


class EtlJoinOperation(EtlOperationBase):
    operation: Literal["join"]
    left_key: ColumnName
    right_key: ColumnName
    columns: ColumnMapping
    cardinality: Literal["many_to_one", "one_to_one"]


class EtlAssertOperation(EtlOperationBase):
    operation: Literal["assert"]
    column: ColumnName
    rule: Literal["required", "unique", "allowed", "decimal", "join_match"]
    values: (
        Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)] | None
    ) = None

    @model_validator(mode="after")
    def allowed_values(self):
        if self.rule == "allowed" and not self.values:
            raise ValueError("ETL_CONFIGURATION_REQUIRED")
        if self.rule != "allowed" and self.values is not None:
            raise ValueError("UNSUPPORTED_ETL_CONFIGURATION")
        return self


class EtlQuarantineOperation(EtlOperationBase):
    operation: Literal["quarantine"]
    reasons: Annotated[
        list[
            Literal[
                "invalid_decimal",
                "join_unmatched",
                "required",
                "unique",
                "allowed",
                "decimal",
                "join_match",
            ]
        ],
        Field(min_length=1),
    ]


class EtlGroupOperation(EtlOperationBase):
    operation: Literal["group"]
    keys: ColumnList
    sum: ColumnName
    count: ColumnName

    @model_validator(mode="after")
    def output_names(self):
        if self.sum in self.keys or self.count in self.keys or self.sum == self.count:
            raise ValueError("ETL_COLUMN_COLLISION")
        return self


class EtlExportOperation(EtlOperationBase):
    operation: Literal["export"]


EtlOperation = Annotated[
    Union[
        EtlSourceOperation,
        EtlSelectOperation,
        EtlRenameOperation,
        EtlNormalizeOperation,
        EtlDecimalOperation,
        EtlFilterOperation,
        EtlJoinOperation,
        EtlAssertOperation,
        EtlQuarantineOperation,
        EtlGroupOperation,
        EtlExportOperation,
    ],
    Field(discriminator="operation"),
]
ETL_OPERATION_ADAPTER = TypeAdapter(EtlOperation)


class EtlInputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schemaVersion: Literal["etl_binding.v2"]
    sourcePort: Literal["rows"]
    targetPort: Literal["input", "left", "right"]


class EtlRelationColumn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, regex_engine="python-re")
    name: ColumnName
    type: Literal["unknown", "string", "decimal", "integer"] = "unknown"


class EtlRelationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schemaVersion: Literal["etl_relation.v2"] = "etl_relation.v2"
    columns: list[EtlRelationColumn]
    known: bool = True

    @model_validator(mode="after")
    def unique_names(self):
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("ETL_COLUMN_COLLISION")
        return self
