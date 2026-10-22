"""Explicit synthetic descendants; parent evidence is never rewritten in place."""
from copy import deepcopy
from src.research.tender_small_dlm.context_envelope import content_hash
from src.research.tender_small_dlm.programmability_minimum import _bind_research_identity

FAMILIES=('scoring_not_elimination','contextual_mention','related_attribute','mere_mention')


def synthetic_pairs(original_fixture,parent,assets):
    rows=[]
    for kind in ('counterfactual_true',)+FAMILIES:
        row=deepcopy(original_fixture)
        text=assets['replacement_prefix'].format(target_text=parent['target_text'])+assets[kind]
        payload=row['payload']
        dlp=payload['dlp_model_input']
        dlp['source_excerpts']=[{'ref':'synthetic:e0','role':'target_clause','text':text}]
        dlp['assertions']=[{'ref':'synthetic:a0','subject_ref':'synthetic:subject','subject_label':text,
            'predicate':'candidate_for','object_ref':dlp['semantic_predicates'][0]['predicate_id'],
            'object_label':dlp['semantic_predicates'][0]['semantic_type_ref'],'evidence_set':'synthetic:e0'}]
        dlp['decision_subject_ref']='synthetic:subject'
        dlp['material_dependency_resolutions']=[]
        payload['source_identity']={'synthetic':True,'source_group_id':parent['source_group_id'],
            'text_hash':content_hash(text),'parent_source_hash':parent['raw_sha256'],
            'parent_source_anchor_ref':parent['source_anchor_ref']}
        _bind_research_identity(payload)
        row.update(fixture_id=parent['candidate_id']+':'+kind,kind=kind,
            gold='TRUE' if kind=='counterfactual_true' else 'FALSE',synthetic=True,
            semantic_delta='complete operative replacement in a fictional descendant; '+kind,
            hard_negative_family=None if kind=='counterfactual_true' else kind,
            pair_id=parent['candidate_id']+':operative-vs-nonoperative',
            parent_source_hash=parent['raw_sha256'])
        rows.append(row)
    return rows
