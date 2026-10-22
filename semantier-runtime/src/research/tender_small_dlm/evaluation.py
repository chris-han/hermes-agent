"""Strict post-training gate. Quality/calibration use original DEV parents only."""
import math
from src.research.tender_small_dlm.transforms import FAMILIES


def evaluate(fixtures,results,baseline,thresholds):
    by={r['fixture_id']:r for r in results}
    if len(by)!=len(results) or set(by)!={f['fixture_id'] for f in fixtures}:
        raise ValueError('EXACT_EVALUATION_COVERAGE_REQUIRED')
    groups={};semantic={};correct={}
    for f in fixtures:
        r=by[f['fixture_id']];p=r['probabilities']
        if set(p)!=set(f['labels']) or r['selected_label'] not in p or any(not math.isfinite(x) or x<0 or x>1 for x in p.values()) or abs(sum(p.values())-1)>1e-6:
            raise ValueError('INVALID_EVALUATION_DISTRIBUTION')
        semantic[f['fixture_id']]=f['labels'][r['selected_label']]
        correct[f['fixture_id']]=semantic[f['fixture_id']]==f['gold']
        groups.setdefault(f['case_id'],{})[f['kind']]=f
    required={'original','permutation','label_remap','paraphrase','frontier','unknown','counterfactual_true',*FAMILIES}
    if not groups or any(set(g)!=required for g in groups.values()):raise ValueError('EXACT_VARIANT_COVERAGE_REQUIRED')
    n=len(groups);originals=[g['original'] for g in groups.values()]
    originals_correct=sum(correct[f['fixture_id']] for f in originals)
    recalls={};f1=[]
    for label in ('TRUE','FALSE'):
        tp=sum(f['gold']==label and semantic[f['fixture_id']]==label for f in originals)
        fp=sum(f['gold']!=label and semantic[f['fixture_id']]==label for f in originals)
        fn=sum(f['gold']==label and semantic[f['fixture_id']]!=label for f in originals)
        recalls[label]=tp/(tp+fn) if tp+fn else None
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    metrics={'original_accuracy':originals_correct/n,'A_recall':recalls['TRUE'],'B_recall':recalls['FALSE'],'macro_F1':sum(f1)/2}
    for kind,name in [('permutation','candidate_order_consistency'),('label_remap','label_remap_consistency'),('paraphrase','paraphrase_consistency')]:
        metrics[name]=sum(semantic[g['original']['fixture_id']]==semantic[g[kind]['fixture_id']] for g in groups.values())/n
    metrics['fixed_label_collapse_rate']=sum(by[g['original']['fixture_id']]['selected_label']==by[g['label_remap']['fixture_id']]['selected_label'] for g in groups.values())/n
    metrics['frontier_correctness']=sum(correct[g['frontier']['fixture_id']] for g in groups.values())/n
    metrics['unknown_accuracy']=sum(correct[g['unknown']['fixture_id']] for g in groups.values())/n
    metrics['counterfactual_pair_accuracy']=sum(correct[g['counterfactual_true']['fixture_id']] and correct[g['scoring_not_elimination']['fixture_id']] for g in groups.values())/n
    hard_negatives=[f for f in fixtures if f['gold']=='FALSE' and (f['kind'] in FAMILIES or f['kind']=='original')]
    metrics['hard_negative_gold_B_predicted_A']=sum(semantic[f['fixture_id']]=='TRUE' for f in hard_negatives)
    metrics['hard_negative_accuracy']=sum(correct[f['fixture_id']] for f in hard_negatives)/len(hard_negatives) if hard_negatives else None
    metrics['gold_B_predicted_A']=sum(f['gold']=='FALSE' and semantic[f['fixture_id']]=='TRUE' for f in originals)
    baseline_by={r['fixture_id']:r for r in baseline} if baseline is not None else by
    metrics['positive_protection_new_errors']=sum(f['gold']=='TRUE'
        and f['labels'][baseline_by[f['fixture_id']]['selected_label']]=='TRUE'
        and semantic[f['fixture_id']]!='TRUE' for f in originals)
    nll=brier=0.;bins=[[] for _ in range(10)];selected=[]
    for f in originals:
        r=by[f['fixture_id']];p=r['probabilities'];gold=next(k for k,v in f['labels'].items() if v==f['gold'])
        nll-=math.log(max(p[gold],1e-300))
        brier+=sum((value-(label==gold))**2 for label,value in p.items())
        confidence=max(p.values());okay=correct[f['fixture_id']]
        bins[min(9,int(confidence*10))].append((confidence,okay))
        if confidence>=.8:selected.append(okay)
    metrics.update(NLL=nll/n,Brier=brier/n,ECE=sum(abs(sum(c for c,_ in b)/len(b)-sum(ok for _,ok in b)/len(b))*len(b)/n for b in bins if b),
        selective_risk=(1-sum(selected)/len(selected)) if selected else None,selective_coverage=len(selected)/n)
    failures=[]
    for threshold,value in thresholds.items():
        if threshold.endswith('_min') or threshold.endswith('_max'):
            name,mode=threshold.rsplit('_',1)
            if name not in metrics:raise ValueError('UNIMPLEMENTED_GATE:'+threshold)
            metric=metrics[name]
            if metric is None or (metric<value if mode=='min' else metric>value):failures.append(threshold)
    family_report={}
    for f in hard_negatives:
        family=f.get('hard_negative_family') or 'unspecified'
        row=family_report.setdefault(family,{'count':0,'correct':0,'false_positive_A':0})
        row['count']+=1;row['correct']+=correct[f['fixture_id']];row['false_positive_A']+=semantic[f['fixture_id']]=='TRUE'
    return {'metrics':metrics,'hard_negative_families':family_report,'failed_gates':failures,
            'disposition':'FAIL' if failures else 'PASS','quality_basis':'original DEV source groups; binary TRUE/FALSE macro-F1 with UNKNOWN counted as error',
            'calibration_basis':'original DEV source groups; identity temperature; ten equal-width ECE bins; unscaled multiclass Brier',
            'counterfactual_basis':'paired synthetic mandatory versus scoring-only replacements; evidence-removal UNKNOWN measured separately'}
