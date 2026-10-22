"""Exact closed-continuation scoring over a runtime-supplied label frontier."""
import math


def probabilities(scores):
    if not scores or any(not math.isfinite(v) or v > 0 for v in scores.values()):
        raise ValueError('FINITE_LOG_PROBABILITIES_REQUIRED')
    peak=max(scores.values());weights={k:math.exp(v-peak) for k,v in scores.items()}
    total=sum(weights.values())
    return {k:v/total for k,v in weights.items()}


def tokenize(tokenizer, prefix, labels):
    if not prefix.endswith('Decision: ') or not labels or len(set(labels)) != len(labels):
        raise ValueError('INVALID_SCORING_CONTRACT')
    ids=tokenizer.encode(prefix,add_special_tokens=False)
    decode=lambda tokens:tokenizer.decode(tokens,clean_up_tokenization_spaces=False)
    if not ids or decode(ids)!=prefix:
        raise ValueError('TOKEN_PREFIX_CHANGED')
    tails={label:tokenizer.encode(label,add_special_tokens=False) for label in labels}
    for label,tokens in tails.items():
        if not tokens or decode(ids+tokens)!=prefix+label:
            raise ValueError('TOKEN_CONTINUATION_CHANGED')
    return ids,tails


def score(model,tokenizer,prefix,labels,before_forward):
    import torch
    ids,tails=tokenize(tokenizer,prefix,labels)
    scores={}
    with torch.inference_mode():
        # Equal, one-token continuations share the exact same conditional prefix.
        if all(len(tokens)==1 for tokens in tails.values()):
            tensor=torch.tensor([ids],device=model.device)
            before_forward(len(ids))
            output=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1)
            logp=output.logits[0,-1].float().log_softmax(-1)
            scores={label:float(logp[tokens[0]].item()) for label,tokens in tails.items()}
        else:
            for label,tokens in tails.items():
                tensor=torch.tensor([ids+tokens[:-1]],device=model.device)
                before_forward(tensor.shape[1])
                positions=torch.arange(len(ids)-1,tensor.shape[1],device=model.device)
                output=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=positions)
                logp=output.logits[0].float().log_softmax(-1)
                scores[label]=float(logp.gather(-1,torch.tensor(tokens,device=model.device)[:,None]).sum().item())
    p=probabilities(scores)
    return {'log_probabilities':scores,'probabilities':p,'selected_label':max(labels,key=lambda x:scores[x]),
            'input_tokens':len(ids),'continuation_ids':tails,'generated_tokens':0}
