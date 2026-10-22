"""Bounded research trainer: source-group isolation and explicit call accounting."""
from datetime import datetime,timezone
import os
import time
from scripts.research.tender_small_dlm.prerequisites import write,digest
from src.research.tender_small_dlm.context_envelope import canonical
from src.research.tender_small_dlm.scoring import score,tokenize
from src.research.tender_small_dlm.candidate_scoring_loss import decision_loss
from src.research.tender_small_dlm.evaluation import evaluate


def now():return datetime.now(timezone.utc).isoformat()


def require_b2_sealed(root):
    for name in ('b2-reveal-go.v1.json','b2-execution-start.v1.json','b2-result.v1.json'):
        if (root/name).exists():raise ValueError('B2_SEAL_BROKEN')


def validate_schedule(rows,schedule,budget):
    by={r['fixture_id']:r for r in rows}
    if len(by)!=len(rows):raise ValueError('DUPLICATE_FIXTURE')
    if len(schedule)>budget['max_update_steps']*budget['gradient_accumulation']:
        raise ValueError('UPDATE_BUDGET_EXCEEDED')
    for item in schedule:
        if by[item['fixture_id']]['split']!='TRAIN':raise ValueError('DEV_IN_OPTIMIZER_SCHEDULE')
        if not 0<=item['epoch']<budget['epoch_cap']:raise ValueError('EPOCH_BUDGET_EXCEEDED')
    if not schedule or len(schedule)%budget['gradient_accumulation']:
        raise ValueError('COMPLETE_ACCUMULATION_GROUPS_REQUIRED')


def train(model,tokenizer,rows,prefixes,schedule,budget,thresholds,dest,physical):
    import torch
    validate_schedule(rows,schedule,budget)
    by={r['fixture_id']:r for r in rows};dev=[r for r in rows if r['split']=='DEV']
    parameters=[p for p in model.parameters() if p.requires_grad]
    if any(p.device.type!='cuda' for p in model.parameters()):raise ValueError('OFFLOAD_PROHIBITED')
    optimizer=torch.optim.AdamW(parameters,lr=budget['learning_rate'],weight_decay=budget['weight_decay'])
    counts={'training_forward':0,'backward':0,'evaluation_forward':0,'optimizer_updates':0}
    journal=(dest/'execution-ledger.v1.jsonl').open('x')
    evaluations=[];baseline=None;error=None;pending_losses=[];started=time.monotonic()
    tokenized={f['fixture_id']:tokenize(tokenizer,prefixes[f['fixture_id']],f['candidate_order']) for f in rows}
    for _,tails in tokenized.values():
        if any(len(v)!=1 for v in tails.values()):raise ValueError('ONE_TOKEN_TRAINING_REQUIRED')
    caps={'training_forward':budget['training_forward_max'],'backward':budget['backward_max'],
          'evaluation_forward':budget['evaluation_forward_max'],'optimizer_updates':budget['max_update_steps']}
    def reserve(kind,fixture_id=None):
        require_b2_sealed(dest.parents[1])
        if counts[kind]>=caps[kind]:raise ValueError('EXECUTION_BUDGET_EXCEEDED:'+kind)
        if kind!='optimizer_updates':counts[kind]+=1
        journal.write(canonical({'produced_at':now(),'event':kind+'_reserved','fixture_id':fixture_id,'counts':dict(counts)})+'\n')
        journal.flush();os.fsync(journal.fileno())
    def physical_check():
        torch.cuda.synchronize()
        if max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>=physical:
            raise ValueError('PHYSICAL_TRAINING_BOUND_EXCEEDED')
    def evaluation(step):
        nonlocal baseline
        model.eval();results=[]
        for f in dev:
            row={'fixture_id':f['fixture_id'],'status':'COMPLETE',**score(model,tokenizer,prefixes[f['fixture_id']],
                f['candidate_order'],lambda tokens:reserve('evaluation_forward',f['fixture_id']))}
            results.append(row);physical_check()
        if baseline is None:baseline=results
        summary=evaluate(dev,results,baseline,thresholds)
        evidence={'produced_at':now(),'optimizer_updates':step,'results':results,**summary}
        write(dest/f'evaluation-{step:03}.json',evidence)
        evaluations.append({'step':step,'disposition':summary['disposition'],'failed_gates':summary['failed_gates']})
        print(canonical({'event':'evaluation','step':step,**summary}),flush=True)
        return summary
    try:
        torch.cuda.reset_peak_memory_stats()
        evaluation(0)
        optimizer.zero_grad(set_to_none=True)
        for index,item in enumerate(schedule):
            model.train();f=by[item['fixture_id']];ids,tails=tokenized[f['fixture_id']]
            tensor=torch.tensor([ids],device=model.device)
            reserve('training_forward',f['fixture_id'])
            output=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1)
            gold_label=next(k for k,v in f['labels'].items() if v==f['gold'])
            loss=decision_loss(output.logits[0,-1],[tails[k][0] for k in f['candidate_order']],f['candidate_order'].index(gold_label))
            if not torch.isfinite(loss):raise ValueError('NONFINITE_TRAINING_LOSS')
            pending_losses.append(float(loss.detach().item()))
            reserve('backward',f['fixture_id'])
            (loss/budget['gradient_accumulation']).backward()
            del loss,output,tensor
            physical_check()
            if (index+1)%budget['gradient_accumulation']:continue
            norm=torch.nn.utils.clip_grad_norm_(parameters,budget['max_gradient_norm'],error_if_nonfinite=True)
            reserve('optimizer_updates')
            optimizer.step();counts['optimizer_updates']+=1
            physical_check();optimizer.zero_grad(set_to_none=True)
            step=counts['optimizer_updates']
            receipt={'produced_at':now(),'step':step,'microbatches_completed':index+1,'epoch':item['epoch'],
                'mean_loss':sum(pending_losses)/len(pending_losses),'gradient_norm':float(norm.item()),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            write(dest/f'update-{step:03}.json',receipt);pending_losses=[]
            print(canonical({'event':'optimizer_update',**receipt}),flush=True)
            if step in budget['checkpoint_steps']:
                checkpoint=dest/f'adapter-{step:03}'
                if checkpoint.exists():raise ValueError('CHECKPOINT_ALREADY_EXISTS')
                model.save_pretrained(checkpoint,safe_serialization=True)
                write(dest/f'adapter-{step:03}.freeze.json',{'produced_at':now(),'optimizer_updates':step,
                    'adapter_files_sha256':{p.name:digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}})
            if step in budget['evaluation_steps']:
                result=evaluation(step)
                if result['disposition']=='PASS':break
        terminal='TRAINING_COMPLETED'
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
        terminal='DATA_FIREWALL_VIOLATION' if 'B2_SEAL_BROKEN' in str(exc) else 'TRAINING_MECHANICS_FAILURE'
    finally:
        journal.close()
    receipt={'produced_at':now(),'status':terminal,'counts':counts,'evaluations':evaluations,
        'error':error,'elapsed_seconds':time.monotonic()-started,'B2_revealed':False,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'physical_headroom_bytes':physical-max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()),
        'evidence_sha256':{p.name:digest(p) for p in sorted(dest.glob('*.json'))},
        'ledger_sha256':digest(dest/'execution-ledger.v1.jsonl')}
    write(dest/'training-terminal.v1.json',receipt)
    print(canonical({'event':'training_terminal',**receipt}),flush=True)
    return receipt
