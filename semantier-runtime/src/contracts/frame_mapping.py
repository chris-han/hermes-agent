"""Explicit DEVELOPMENT frame mapping, pinned to proposal and complete source spans."""
from __future__ import annotations
from dataclasses import dataclass
import json
import re
from typing import Any


def _object(value: Any, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f'{name} requires exact typed fields')
    return value


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]*', value):
        raise ValueError(f'{name} must be an ASCII machine identifier')


def _span(value: Any) -> None:
    span = _object(value, {'evidence_ref','start','end','quote'}, 'frame mapping span')
    _identifier(span['evidence_ref'], 'span.evidence_ref')
    if (type(span['start']) is not int or type(span['end']) is not int
            or span['start'] < 0 or span['end'] <= span['start']
            or not isinstance(span['quote'], str) or len(span['quote']) != span['end']-span['start']):
        raise ValueError('frame mapping requires exact Unicode source spans')


def _source_conditions(blocks: dict, evidence_refs) -> tuple[dict, list[dict]]:
    """Resolve the unique complete list from pinned evidence, never a review span.

    The v1 operand supports line-start numbered clauses, separated by semicolons
    or sentence punctuation. Multiple lists or unterminated clauses are ambiguous
    and need a more precise producer binding, not a submitted scope override.
    """
    marker = re.compile(r'^[ \t]*([（(](?:[一二三四五六七八九十]+|[0-9]+)[）)])', re.MULTILINE)
    lists = []
    for ref in evidence_refs:
        block = blocks.get(ref)
        if block is None or not isinstance(block.content, str):
            raise ValueError('frame mapping missing or wrong-source span')
        content = block.content
        marks = list(marker.finditer(content))
        members = []
        for index, mark in enumerate(marks):
            start = mark.start(1)
            next_start = marks[index + 1].start(1) if index + 1 < len(marks) else len(content)
            segment = content[start:next_start]
            terminal = re.search(r'[。.]', segment)
            if terminal:
                end = start + terminal.end()
                continues = index + 1 < len(marks) and not content[end:next_start].strip()
                # Whitespace belongs to the preceding clause when another follows.
                if continues:
                    end = next_start
            else:
                if index + 1 == len(marks) or not segment.rstrip().endswith((';', '；')):
                    raise ValueError('frame mapping source condition is unterminated or ambiguous')
                end = next_start
                continues = True
            members.append(dict(evidence_ref=ref, start=start, end=end, quote=content[start:end]))
            if not continues:
                lists.append(members)
                members = []
        if members:
            raise ValueError('frame mapping source list is unterminated')
    if len(lists) != 1:
        raise ValueError('frame mapping requires one unambiguous complete source list')
    members = lists[0]
    first, last = members[0], members[-1]
    scope = dict(evidence_ref=first['evidence_ref'], start=first['start'], end=last['end'],
                 quote=blocks[first['evidence_ref']].content[first['start']:last['end']])
    return scope, members


@dataclass(frozen=True)
class FrameMapping:
    # Immutable normalized representation; callers cannot change hashed content.
    canonical_json: str

    @classmethod
    def from_dict(cls, value: Any) -> 'FrameMapping':
        data = _object(value, {'schema_version','proposal_ref','proposal_hash','source_binding',
            'question_id','authorship','decision_scope','representation_type_id','grouping',
            'combination','rationale','shared_context','member_scope','members','frames','cardinality'}, 'frameMapping')
        if (data['schema_version'] != 'frame_mapping.v1' or data['authorship'] != 'agent_proposal'
                or data['decision_scope'] != 'development_modeling' or data['combination'] != 'any_of'
                or data['grouping'] not in {'independent_frames','grouped_alternatives'}):
            raise ValueError('unsupported frame mapping semantics or authorship')
        for field in ('proposal_ref','question_id','representation_type_id'):
            _identifier(data[field], field)
        if not isinstance(data['proposal_hash'], str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',data['proposal_hash']):
            raise ValueError('frame mapping proposal hash required')
        if not isinstance(data['source_binding'],dict) or not isinstance(data['rationale'],str) or not data['rationale'].strip():
            raise ValueError('frame mapping source binding and rationale required')
        if not isinstance(data['shared_context'],list) or not data['shared_context']:
            raise ValueError('frame mapping shared source context required')
        for span in data['shared_context']:
            _span(span)
        _span(data['member_scope'])
        if not isinstance(data['members'],list) or not data['members'] or not isinstance(data['frames'],list) or not data['frames']:
            raise ValueError('frame mapping members and frames required')
        ids=[]
        scope=data['member_scope']; cursor=scope['start']
        for item in data['members']:
            _object(item,{'member_id','span'},'frame member'); _identifier(item['member_id'],'member_id'); _span(item['span'])
            span=item['span']; ids.append(item['member_id'])
            if span['evidence_ref'] != scope['evidence_ref'] or span['start'] < cursor or span['end'] > scope['end']:
                raise ValueError('contradictory or overlapping frame member spans')
            if scope['quote'][cursor-scope['start']:span['start']-scope['start']].strip():
                raise ValueError('missing frame member source text')
            if scope['quote'][span['start']-scope['start']:span['end']-scope['start']] != span['quote']:
                raise ValueError('frame member quote differs from full scope')
            cursor=span['end']
        if scope['quote'][cursor-scope['start']:].strip() or len(set(ids)) != len(ids):
            raise ValueError('missing or duplicate frame members')
        frame_ids=[]; assigned=[]
        for frame in data['frames']:
            _object(frame,{'frame_id','member_ids'},'frame'); _identifier(frame['frame_id'],'frame_id')
            if not isinstance(frame['member_ids'],list) or not frame['member_ids']:
                raise ValueError('frame member membership required')
            frame_ids.append(frame['frame_id']); assigned.extend(frame['member_ids'])
            if data['grouping']=='independent_frames' and len(frame['member_ids']) != 1:
                raise ValueError('contradictory independent grouping')
        if len(set(frame_ids)) != len(frame_ids) or sorted(assigned) != sorted(ids):
            raise ValueError('contradictory frame membership')
        if data['grouping']=='grouped_alternatives' and len(data['frames']) != 1:
            raise ValueError('contradictory grouped alternatives')
        card=_object(data['cardinality'],{'frame_count','member_count'},'cardinality')
        if (type(card['frame_count']) is not int or type(card['member_count']) is not int
                or card != {'frame_count':len(data['frames']),'member_count':len(ids)}):
            raise ValueError('frame mapping cardinality does not match membership')
        return cls(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':')))

    def as_dict(self) -> dict:
        return json.loads(self.canonical_json)

    def validate_source(self, binding, canonical, *, evidence_refs=None) -> None:
        data=self.as_dict()
        if data['source_binding'] != binding.as_dict():
            raise ValueError('frame mapping wrong source or version pins')
        blocks={f'evidence:canonical-block:{b.block_id.removeprefix("sha256:")}':b for b in canonical.blocks}
        if not any(re.search(r'之一|任一|any (?:one|of)', s['quote'], re.IGNORECASE) for s in data['shared_context']):
            raise ValueError('any_of grouping lacks source-supported alternative connector')
        for span in [*data['shared_context'],data['member_scope'],*[m['span'] for m in data['members']]]:
            block=blocks.get(span['evidence_ref'])
            if block is None or not isinstance(block.content,str) or block.content[span['start']:span['end']] != span['quote']:
                raise ValueError('frame mapping missing or wrong-source span')
        required_scope, required_members = _source_conditions(
            blocks, evidence_refs if evidence_refs is not None else (data['member_scope']['evidence_ref'],))
        if data['member_scope'] != required_scope:
            raise ValueError('frame mapping must cover the complete source list scope')
        if [member['span'] for member in data['members']] != required_members:
            raise ValueError('frame mapping must preserve exact complete source condition identities')

    def validate_review(self, predecessor, canonical) -> None:
        data=self.as_dict()
        if data['proposal_ref'] != predecessor.graph_intent_id or data['proposal_hash'] != predecessor.graph_intent_hash:
            raise ValueError('stale frame mapping proposal pins')
        question=next((q for q in predecessor.unresolved_questions if q.question_id==data['question_id']),None)
        if question is None or not question.material or not {'grouping','cardinality'}.issubset(question.affected_semantics):
            raise ValueError('frame mapping requires exact unresolved grouping/cardinality question')
        self.validate_source(predecessor.source_binding, canonical, evidence_refs=question.evidence_refs)
        used={s['evidence_ref'] for s in [*data['shared_context'],data['member_scope']]}
        if used != set(question.evidence_refs):
            raise ValueError('frame mapping must preserve all question evidence')
        if predecessor.preferred_representation.kind != 'frame' or data['representation_type_id'] != predecessor.preferred_representation.type_id:
            raise ValueError('frame mapping cannot change admitted representation')
