"""Candidate relevance scoring; no generation or historical Tender class head.

This implementation is preparatory. Numerical unit checks do not qualify a
source population, freeze the study, or authorize scientific optimization.
"""
from __future__ import annotations


def listwise_loss(scores, gold_index):
    import torch
    if scores.ndim != 1 or not 2 <= scores.numel() <= 4:
        raise ValueError('FRONTIER_SIZE: expected two to four candidate scores')
    if type(gold_index) is not int or not 0 <= gold_index < scores.numel():
        raise ValueError('GOLD_INDEX: gold must be a supplied frontier member')
    scores = scores.float()
    if not torch.isfinite(scores).all():
        raise ValueError('NONFINITE_CANDIDATE_SCORE')
    return torch.logsumexp(scores, dim=0) - scores[gold_index]


def score_frontier(model, tokenizer, prompts, *, yes_id, no_id, device):
    """One whole frontier; sequential forwards retain every candidate's graph.

    Caller supplies the canonical mapping from render_scoring_inputs and sets
    train/eval mode. Validate the entire frontier's context before any forward.
    """
    import torch
    if not 2 <= len(prompts) <= 4:
        raise ValueError('FRONTIER_SIZE: expected two to four candidate prompts')
    if type(yes_id) is not int or type(no_id) is not int or min(yes_id, no_id) < 0 or yes_id == no_id:
        raise ValueError('VERBALIZER_IDS: require distinct nonnegative token IDs')
    encoded = []
    for text in prompts.values():
        item = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors='pt')
        tokens, attention = item['input_ids'], item['attention_mask']
        if tokens.ndim != 2 or tokens.shape[0] != 1 or attention.shape != tokens.shape:
            raise ValueError('FRONTIER_MICROBATCH: exactly one unpadded input per candidate')
        if tokens.shape[1] > 512:
            raise ValueError('CONTEXT_OVERFLOW: complete scoring input exceeds 512 tokens')
        if tokens.shape[1] == 0 or not torch.all(attention == 1):
            raise ValueError('EMPTY_OR_PADDED_SCORING_INPUT')
        encoded.append({'input_ids': tokens.to(device), 'attention_mask': attention.to(device)})
    logits = []
    for item in encoded:
        output = model(**item, use_cache=False)
        pair = output.logits[0, -1, [yes_id, no_id]].float()
        if not torch.isfinite(pair).all():
            raise ValueError('NONFINITE_VERBALIZER_LOGIT')
        logits.append(pair)
    raw = torch.stack(logits)
    scores = raw[:, 0] - raw[:, 1]
    if not torch.isfinite(scores).all():
        raise ValueError('NONFINITE_CANDIDATE_SCORE')
    return {'candidate_ids': list(prompts), 'verbalizer_logits': raw, 'scores': scores,
            'input_lengths': [item['input_ids'].shape[1] for item in encoded]}
