"""Joint personal-package transitions and one atomic runtime generation."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import uuid
from contracts.change_candidate_v2 import ChangeCandidateV2, ChangeSetProfileV1, ChangeSetMemberV1
from contracts.learning_workbench import SkillArtifactVersionV1
from contracts.learned_skill_publication import LearnedSkillAssessmentV1, LearnedSkillSetAssessmentV1, LearnedSkillSetReceiptV1, LearnedSkillSetProposalV1
from eos.artifact_hashing import canonical_json, content_hash
from services.learned_skill_publication_service import _now, _file


class LearnedSkillPackageSet:
    def __init__(self, owner):
        self.owner=owner

    def objects(self, context, ref, hash=None):
        candidate=self.owner._load(context,ref,ChangeCandidateV2,hash)
        if not isinstance(candidate.profile,ChangeSetProfileV1) or candidate.profile.authority_domain_ref!='personal-procedure':
            raise ValueError('INVALID_PACKAGE_SET')
        members=[]
        for member in candidate.profile.members:
            version=self.owner._load(context,member.delta.successor_candidate.version_id,SkillArtifactVersionV1,member.delta.successor_candidate.content_hash)
            base=self.owner._load(context,member.delta.predecessor.version_id,SkillArtifactVersionV1,member.delta.predecessor.content_hash)
            if version.predecessor != base.pin() or version.skill_id!=base.skill_id or member.member_id!=version.skill_id or member.delta.successor_candidate!=self.owner._version_pin(version) or member.delta.predecessor!=self.owner._version_pin(base):
                raise ValueError('EXACT_PACKAGE_SET_MISMATCH')
            members.append((version,base))
        return candidate,members

    def propose(self, context, packages, origin, request_id):
        from services.governed_delta_service import GovernedDeltaService
        from contracts.governed_delta import DeltaTargetType,DeltaLens,DeltaLensResultV1,DeltaLensStatus
        from tools.skill_review_capture import read_package,name_check
        owner=self.owner;scope=owner._scope(context)
        request_hash=content_hash({'packages':packages,'origin':origin})
        winner=owner._store.get_operation_result(**scope,actor_ref=context.user_id,operation='propose_skill_set',request_id=request_id,request_hash=request_hash,artifact_kind='change_candidate')
        if winner:return self.project(context,winner.artifact_ref,winner.artifact_hash)
        artifacts=[];members=[];bases=[]
        known=owner._owned(context,SkillArtifactVersionV1,'skill_artifact_version.v1')
        for package in packages:
            name=name_check(package['skill_id'])
            files=tuple(_file(p,base64.b64decode(b,validate=True)) for p,b in sorted(package['base'].items()))
            pinned=package.get('base_version')
            base=next((v for v in known if v.skill_id==name and v.files==files and (not pinned or (v.artifact_ref==pinned['artifact_ref'] and v.artifact_hash==pinned['artifact_hash']))),None)
            if base is None:
                if any(v.skill_id==name for v in known) or read_package(context.hermes_home/'skills'/name)!=package['base']:
                    raise ValueError('STALE_PREDECESSOR')
                base=SkillArtifactVersionV1.create(**scope,owner_ref=context.user_id,skill_id=name,version=1,provenance='legacy',files=files,recorded_at=_now())
                artifacts.append(base)
            owner._assert_base(context,base)
            after=tuple(_file(p,base64.b64decode(b,validate=True)) for p,b in sorted(package['candidate'].items()))
            version=SkillArtifactVersionV1.create(**scope,owner_ref=context.user_id,skill_id=name,version=base.version+1,provenance='managed',predecessor=base.pin(),files=after,recorded_at=_now())
            if after:owner._check_content(context,name,owner._content(version))
            delta=GovernedDeltaService.build_envelope(delta_id='package:'+version.artifact_hash[7:],target_type=DeltaTargetType.WORKFLOW,predecessor=owner._version_pin(base),successor_candidate=owner._version_pin(version),lens_results={lens:DeltaLensResultV1(lens=lens,status=DeltaLensStatus.UNAVAILABLE,basis_refs=tuple(sorted((base.artifact_ref,version.artifact_ref)))) for lens in DeltaLens})
            members.append(ChangeSetMemberV1(member_id=name,delta=delta,rationale_refs=(base.pin(),version.pin())))
            bases.append(base.pin());artifacts.append(version)
        candidate=ChangeCandidateV2.create(**scope,revision=0,candidate_kind='CHANGE_SET',profile=ChangeSetProfileV1(owner_ref=context.user_id,authority_domain_ref='personal-procedure',rationale='Review the exact coordinated personal procedure package change.',basis_refs=tuple(bases),coherence_refs=tuple(bases),members=tuple(members)),recorded_at=_now())
        artifacts.append(candidate)
        binding=LearnedSkillSetProposalV1.create(**scope,owner_ref=context.user_id,candidate=candidate.pin(),request_id=request_id,request_hash=request_hash,origin=origin,recorded_at=_now())
        artifacts.append(binding)
        winner=owner._store.put_many_with_operation_result(artifacts=tuple(artifacts),**scope,actor_ref=context.user_id,operation='propose_skill_set',request_id=request_id,request_hash=request_hash,result_pin=candidate.pin())
        return self.project(context,winner.artifact_ref,winner.artifact_hash)

    def assess(self, context, ref, hash):
        candidate,members=self.objects(context,ref,hash);owner=self.owner
        assessments=[]
        for version,base in members:
            owner._assert_base(context,base)
            if version.files:owner._check_content(context,version.skill_id,owner._content(version))
            classification=owner._classify_package(candidate,version,base)
            assessments.append(LearnedSkillAssessmentV1.create(**owner._scope(context),owner_ref=context.user_id,candidate=candidate.pin(),version=version.pin(),predecessor=base.pin(),classification=classification,eligible=classification.disposition=='PROCEDURAL_SAFE_TO_PROJECT',counterevidence=('Package effects assessed; empirical generalization and observed use unknown.',),recorded_at=_now()))
        assessment=LearnedSkillSetAssessmentV1.create(**owner._scope(context),owner_ref=context.user_id,candidate=candidate.pin(),member_assessments=tuple(assessments),eligible=all(a.eligible for a in assessments),recorded_at=_now())
        owner._store.put(assessment)
        return assessment.model_dump(mode='json')

    def approve(self,context,ref,hash,assessment_ref,assessment_hash):
        owner=self.owner;candidate,members=self.objects(context,ref,hash)
        assessment=owner._load(context,assessment_ref,LearnedSkillSetAssessmentV1,assessment_hash)
        if assessment.candidate!=candidate.pin() or not assessment.eligible or [(a.version,a.predecessor) for a in assessment.member_assessments]!=[(v.pin(),b.pin()) for v,b in members]:
            raise ValueError('STALE_OR_INELIGIBLE_SET_ASSESSMENT')
        for (version,base),a in zip(members,assessment.member_assessments):
            owner._assert_base(context,base)
            if owner._classify_package(candidate,version,base)!=a.classification:
                raise ValueError('STALE_ASSESSMENT')
        row=self._row(context,candidate,members,assessment)
        if row['status']=='PENDING':
            owner._governance.record_replay_result(row['candidate_id'],regressions=[],tested_claims=len(members))
            owner._governance.record_approval(row['candidate_id'],context.user_id,actor_roles={'scoped_owner'})
        return self.project(context,ref,hash)

    def _name(self,context,candidate):
        return self.owner._name(context,'set-'+candidate.artifact_hash[7:])

    def _row(self,context,candidate,members,assessment=None):
        row=self.owner._row(context,candidate)
        if row:return row
        payload={'candidate':candidate.pin().model_dump(mode='json'),'version':candidate.pin().model_dump(mode='json'),'member_versions':{v.skill_id:v.pin().model_dump(mode='json') for v,b in members},'assessment':assessment.pin().model_dump(mode='json') if assessment else None}
        return self.owner._governance.propose(name=self._name(context,candidate),category='learned_procedure',content_yaml=canonical_json(payload),content_hash=content_hash(payload),proposed_by=context.user_id,scoped_owner=context.user_id,rollback_window_hours=24,rollback_strategy='Publish exact retained predecessors in a new governed package set')

    def activate(self,context,ref,hash):
        owner=self.owner;candidate,members=self.objects(context,ref,hash);row=owner._row(context,candidate)
        if not row or row['status'] not in ('APPROVED','ACTIVE'):raise ValueError('PROCEDURE_NOT_APPROVED')
        with owner._governance.learned_procedure_transition(name=self._name(context,candidate)):
            if row['status']=='ACTIVE' and any((owner._active(context,v.skill_id) or {}).get('candidate_id')!=row['candidate_id'] for v,b in members):
                raise ValueError('STALE_ACTIVATION')
            if row['status']!='ACTIVE':
                for version,base in members:owner._assert_base(context,base)
                owner._governance.activate(row['candidate_id'],actor=context.user_id,actor_roles={'scoped_owner'})
        return self.project(context,ref,hash)

    def publish(self,context,ref,hash):
        from tools.skill_manager_tool import _skill_mutation_locks
        from tools.skill_review_capture import generation,digest
        from tools.skill_usage import skill_file_lock
        from utils import atomic_json_write
        owner=self.owner;candidate,members=self.objects(context,ref,hash);row=owner._row(context,candidate)
        if not row or row['status']!='ACTIVE':raise ValueError('PROCEDURE_NOT_ACTIVE')
        with owner._home(context),_skill_mutation_locks([v.skill_id for v,b in members]),owner._governance.learned_procedure_transition(name=self._name(context,candidate)):
            if any(owner._active(context,v.skill_id)['candidate_id']!=row['candidate_id'] for v,b in members):raise ValueError('STALE_PUBLICATION')
            activation=next(e['candidate_event_id'] for e in reversed(owner._governance.list_candidate_events(row['candidate_id'])) if e['event_type']=='ACTIVATED')
            root=context.hermes_home/'.learned-procedures'
            if root.is_symlink():raise ValueError('FORBIDDEN_PUBLICATION_PATH')
            root.mkdir(parents=True,exist_ok=True)
            for version,base in members:
                parent=root/version.skill_id
                directory=parent/version.artifact_hash[7:]
                if parent.is_symlink() or directory.is_symlink() or root.resolve() not in directory.resolve().parents:
                    raise ValueError('FORBIDDEN_PUBLICATION_PATH')
            updates={}
            for version,base in members:
                directory=root/version.skill_id/version.artifact_hash[7:]
                if not directory.exists():
                    directory.parent.mkdir(parents=True,exist_ok=True)
                    stage=directory.parent/('.stage-'+uuid.uuid4().hex);stage.mkdir()
                    for file in version.files:
                        path=stage/file.relative_path;path.parent.mkdir(parents=True,exist_ok=True)
                        with path.open('wb') as f:f.write(file.content_bytes());f.flush();os.fsync(f.fileno())
                    os.replace(stage,directory)
                if directory.is_symlink() or owner._files(directory)!=version.files:raise ValueError('PUBLICATION_HASH_MISMATCH')
                updates[version.skill_id]={'version_ref':version.artifact_ref,'version_hash':version.artifact_hash,'state':version.package_state,'relative_path':directory.relative_to(root).as_posix(),'files':{f.relative_path:f.content_hash for f in version.files}}
            # This is the only visible runtime commit: every reader resolves all
            # member packages from one complete generation, before or after it.
            with skill_file_lock(root/'.generation.lock'):
                packages=generation(live=True);packages.update(updates)
                generation_hash=digest(packages)
                atomic_json_write(root/'active-generation.json',{'schema_version':'hermes.managed_generation.v1','generation_hash':generation_hash,'packages':packages})
            receipt=next((r for r in owner._owned(context,LearnedSkillSetReceiptV1,'learned_skill_set_receipt.v1') if r.candidate==candidate.pin() and r.activation_ref==activation),None)
            if receipt is None:
                receipt=LearnedSkillSetReceiptV1.create(**owner._scope(context),owner_ref=context.user_id,candidate=candidate.pin(),versions=tuple(v.pin() for v,b in members),activation_ref=activation,generation_hash=generation_hash,recorded_at=_now());owner._store.put(receipt)
            owner._governance.record_learned_procedure_publication(candidate_id=row['candidate_id'],actor=context.user_id,receipt=receipt.pin().model_dump(mode='json'),versions=[v.pin().model_dump(mode='json') for v,b in members])
        return self.project(context,ref,hash)

    def recover(self,context,ref,hash,request_id):
        from tools.skill_manager_tool import _skill_mutation_locks
        owner=self.owner;candidate,members=self.objects(context,ref,hash)
        packages=[{'skill_id':v.skill_id,'base_version':{'artifact_ref':v.artifact_ref,'artifact_hash':v.artifact_hash},'base':{f.relative_path:f.content_base64 for f in v.files},'candidate':{f.relative_path:f.content_base64 for f in b.files}} for v,b in members]
        with owner._home(context),_skill_mutation_locks([v.skill_id for v,b in members]):
            result=self.propose(context,packages,{'operation':'restore_package_set','recovery_target':candidate.pin().model_dump(mode='json')},'recover-set:'+request_id)
            restored=self.owner._load(context,result['ref'],ChangeCandidateV2,result['hash'])
            row=owner._row(context,restored)
            if row is None:
                assessment=self.assess(context,result['ref'],result['hash'])
                self.approve(context,result['ref'],result['hash'],assessment['artifact_ref'],assessment['artifact_hash'])
            self.activate(context,result['ref'],result['hash'])
            return self.publish(context,result['ref'],result['hash'])

    def reject(self,context,ref,hash):
        candidate,members=self.objects(context,ref,hash);row=self._row(context,candidate,members)
        if row['status']=='PENDING':self.owner._governance.record_rejection(row['candidate_id'],context.user_id,'Owner rejected exact package set')
        elif row['status']!='REJECTED':raise ValueError('CANNOT_REJECT_ADMITTED_PROCEDURE')
        return self.project(context,ref,hash)

    def project(self,context,ref,hash=None):
        from tools.skill_review_capture import generation,resolve_managed
        owner=self.owner;candidate,members=self.objects(context,ref,hash);row=owner._row(context,candidate)
        assessments=[a for a in owner._owned(context,LearnedSkillSetAssessmentV1,'learned_skill_set_assessment.v1') if a.candidate==candidate.pin()]
        assessment=max(assessments,key=lambda a:a.recorded_at) if assessments else None
        receipts=[r for r in owner._owned(context,LearnedSkillSetReceiptV1,'learned_skill_set_receipt.v1') if r.candidate==candidate.pin()]
        publication='unpublished';receipt=None
        if row and row['status']=='ACTIVE':
            activation=next(e['candidate_event_id'] for e in reversed(owner._governance.list_candidate_events(row['candidate_id'])) if e['event_type']=='ACTIVATED')
            events=owner._governance.list_candidate_events(row['candidate_id'])
            receipt=next((r for r in receipts if r.activation_ref==activation and any(e['event_type']=='PROCEDURE_PUBLISHED' and e['event']['payload'].get('receipt')==r.pin().model_dump(mode='json') for e in events)),None)
            try:
                with owner._home(context):
                    packages=generation()
                    exact=all(packages.get(v.skill_id,{}).get('version_ref')==v.artifact_ref for v,b in members)
                    for v,b in members:
                        if v.files:resolve_managed(v.skill_id)
                if exact:publication='published' if receipt else 'runtime_matches_retained_version'
            except (OSError,ValueError):pass
        current=not row or row['status']!='ACTIVE' or all((owner._active(context,v.skill_id) or {}).get('candidate_id')==row['candidate_id'] for v,b in members)
        stale=False
        for version,base in members:
            try:owner._assert_base(context,base)
            except ValueError:stale=True
        actions=[]
        if not row and not stale:
            actions=['assess_candidate','reject_candidate']+(['approve_candidate'] if assessment and assessment.eligible else [])
        elif row and row['status']=='APPROVED' and not stale:actions=['activate_candidate']
        elif row and row['status']=='ACTIVE' and current:actions=['publish_procedure','recover_procedure']
        binding=next((p for p in owner._owned(context,LearnedSkillSetProposalV1,'learned_skill_set_proposal.v1') if p.candidate==candidate.pin()),None)
        return {'origin':binding.origin if binding else None,'proposal_ref':binding.artifact_ref if binding else None,'proposal_hash':binding.artifact_hash if binding else None,'object_kind':'candidate','kind':'candidate','candidate_kind':'CHANGE_SET','ref':candidate.artifact_ref,'hash':candidate.artifact_hash,'candidate_ref':candidate.artifact_ref,'candidate_hash':candidate.artifact_hash,'version_ref':candidate.artifact_ref,'version_hash':candidate.artifact_hash,'revision':candidate.revision,'title':'Coordinated skill changes: '+', '.join(v.skill_id for v,b in members),'disposition':row['status'] if row else 'PROPOSED','proposal':'saved','admission':'approved' if row and row['status'] in ('APPROVED','ACTIVE') else 'proposed','activation':'active' if row and row['status']=='ACTIVE' and current else ('superseded' if not current else 'inactive'),'publication':publication,'runtime_availability':'available' if publication=='published' else 'unconfirmed','assessment_ref':assessment.artifact_ref if assessment else None,'assessment_hash':assessment.artifact_hash if assessment else None,'receipt_ref':receipt.artifact_ref if receipt else None,'receipt_hash':receipt.artifact_hash if receipt else None,'package_diff':[d for v,b in members for d in owner.package_diff(b,v)],'members':[{'skill_id':v.skill_id,'base_ref':b.artifact_ref,'base_hash':b.artifact_hash,'version_ref':v.artifact_ref,'version_hash':v.artifact_hash} for v,b in members],'available_actions':actions,'applicability':{'organization_id':context.organization_id,'workspace_id':context.workspace_id,'owner_ref':context.user_id},'leading_answer':'Available for future sessions; use unknown.' if publication=='published' else 'Exact coordinated package proposal retained; runtime publication '+publication+'.','counter_evidence':['Empirical generalization, observed use and causal attribution unknown.']+(['Predecessor changed; refresh comparison.'] if stale else []),'history':owner._governance.list_candidate_events(row['candidate_id']) if row else []}
