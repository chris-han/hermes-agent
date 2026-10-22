"""Runtime-aligned loss for a verified one-token runtime frontier."""

def decision_loss(last_logits, candidate_token_ids, gold_index):
    import torch
    if not candidate_token_ids or len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError('DISTINCT_NONEMPTY_CANDIDATE_TOKENS_REQUIRED')
    if not 0 <= gold_index < len(candidate_token_ids):
        raise ValueError('GOLD_OUTSIDE_FRONTIER')
    # The full-vocabulary log-normalizer cancels exactly across single-token
    # candidate scores from one prefix; this is not a classification head.
    selected = last_logits.float()[candidate_token_ids]
    return -torch.log_softmax(selected, dim=-1)[gold_index]
