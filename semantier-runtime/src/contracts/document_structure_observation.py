"""Research-only normalized Type-2 / Type-3 document observation contract."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ObservationClass = Literal["NATIVE_SOURCE","DETERMINISTIC_RECONSTRUCTION","COMPUTED_LAYOUT","VISUAL_INFERENCE"]
NodeType = Literal["document","section","heading","paragraph","list","list_item","table","row","cell","footnote","endnote","figure","caption","page_region"]
BBoxSpace = Literal["authored_pdf","computed_typeset","rendered_image","none"]
RelationType = Literal["CONTAINS","PRECEDES","HEADING_OF","LIST_PARENT","SAME_ROW","SAME_COLUMN","CELL_OF","SPANS_COLUMNS","MERGED_WITH","FOOTNOTE_OF","CAPTION_OF","CONTINUATION_OF","VISUALLY_GROUPED_WITH","SAME_REGION"]
LinkageType = Literal["EXACT","ONE_TO_MANY","MANY_TO_ONE","APPROXIMATE"]

class _C(BaseModel):
    model_config=ConfigDict(extra="forbid")

class ObservationNode(_C):
    node_id:str=Field(min_length=1)
    node_type:NodeType
    text:str|None=None
    heading_level:int|None=Field(default=None,ge=1)
    list_level:int|None=Field(default=None,ge=0)
    page:int|None=Field(default=None,ge=1)
    bbox:tuple[float,float,float,float]|None=None
    bbox_space:BBoxSpace="none"
    confidence:float|None=Field(default=None,ge=0,le=1)
    observation_class:ObservationClass

    @model_validator(mode="after")
    def geometry_consistent(self):
        if self.bbox is None and self.bbox_space!="none":
            raise ValueError("bbox_space requires bbox")
        if self.bbox is not None and self.bbox_space=="none":
            raise ValueError("bbox requires non-none bbox_space")
        return self

class ObservationRelation(_C):
    relation_id:str=Field(min_length=1)
    relation_type:RelationType
    source_node_id:str=Field(min_length=1)
    target_node_id:str=Field(min_length=1)
    confidence:float|None=Field(default=None,ge=0,le=1)
    observation_class:ObservationClass

class ObservationLink(_C):
    semantic_node_id:str=Field(min_length=1)
    layout_node_id:str=Field(min_length=1)
    linkage_type:LinkageType
    confidence:float|None=Field(default=None,ge=0,le=1)

class DocumentStructureObservation(_C):
    schema_version:Literal["document_structure_observation.v1"]="document_structure_observation.v1"
    source_identity_ref:str=Field(min_length=1)
    source_hash:str=Field(pattern=r"^sha256:[0-9a-f]{64}$")
    engine_id:str=Field(min_length=1)
    engine_version_or_commit:str=Field(min_length=1)
    nodes:list[ObservationNode]
    relations:list[ObservationRelation]=Field(default_factory=list)
    linkage:list[ObservationLink]=Field(default_factory=list)

    @model_validator(mode="after")
    def ids_resolve(self):
        ids={n.node_id for n in self.nodes}
        if len(ids)!=len(self.nodes): raise ValueError("duplicate node_id")
        for r in self.relations:
            if r.source_node_id not in ids or r.target_node_id not in ids:
                raise ValueError("relation endpoint missing")
        for l in self.linkage:
            if l.semantic_node_id not in ids or l.layout_node_id not in ids:
                raise ValueError("link endpoint missing")
        return self
