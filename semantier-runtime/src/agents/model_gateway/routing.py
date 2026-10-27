"""Private heuristic choose-only adapter. No customer strategy activation or classifier inference."""
from fastapi import HTTPException
from contracts.model_gateway import canonical_hash, validate_chat


def candidate_identity(deployments):
    return canonical_hash(sorted((d.deployment_id, d.revision, d.config_fingerprint,
        d.account_id, d.currency, d.funding, d.native_model_id, d.supplier_price_version,
        d.normalizer_version, d.inclusion_rules_version) for d in deployments))


async def choose_only(service, actor, body, *, account_id, currency, funding, policy_id, policy_revision,
                      tier_deployments, expected_candidates, native_source_revision):
    body = validate_chat(body)
    if not policy_id or not policy_revision or not native_source_revision:
        raise HTTPException(422, 'pinned private router policy required')
    if not tier_deployments or set(tier_deployments) - {'SIMPLE', 'MEDIUM', 'COMPLEX', 'REASONING'}:
        raise HTTPException(422, 'bounded heuristic tiers required')
    eligible = [d for d in await service.authorized_deployments(actor)
                if d.account_id == account_id and d.currency == currency and d.funding == funding]
    ids = set(tier_deployments.values())
    selected = [d for d in eligible if d.deployment_id in ids]
    if len(selected) != len(ids) or candidate_identity(selected) != expected_candidates:
        raise HTTPException(409, 'router candidates changed or unavailable')
    by_id = {d.deployment_id: d for d in selected}
    tiers = {tier: [by_id[identity].proxy_model] for tier, identity in tier_deployments.items()}
    if len({d.proxy_model for d in selected}) != len(selected):
        raise HTTPException(409, 'ambiguous physical candidate')
    # Fixed heuristic only: no LLM, embeddings, semantic keywords or caller-supplied plugins.
    config = {'classifier_type': 'heuristic', 'tiers': tiers}
    request = {'messages': body['messages'], 'complexity_router_config': config}
    if body.get('tools'):
        request['tools'] = body['tools']
    response = await service.proxy.operation('POST', '/auto_router/test_routing', request)
    matches = [d for d in selected if d.proxy_model == response.get('routed_model')]
    if len(matches) != 1 or response.get('routed_model_configured') is not True:
        raise HTTPException(409, 'native choice outside exact eligible set')
    chosen = matches[0]
    decision = {
        'schema_version': 'routing_decision.v1', 'policy_id': policy_id, 'policy_revision': policy_revision,
        'candidate_identity': expected_candidates, 'candidate_ids': sorted(ids), 'account_id': account_id,
        'organization_id': actor.organization_id, 'actor_user_id': actor.user_id,
        'workspace_id': actor.workspace_id, 'chosen_deployment_id': chosen.deployment_id,
        'chosen_revision': chosen.revision, 'chosen_config_fingerprint': chosen.config_fingerprint,
        'native_source_revision': native_source_revision, 'native_config_hash': canonical_hash(config),
        'native_config': config, 'tier_deployments': dict(tier_deployments),
        'native_decision': {key: response.get('routing_decision', {}).get(key) for key in
            ('cause', 'tier', 'score', 'tier_boundaries', 'reasoning_override_min_score', 'conversation_continuing')},
        'replay_scope': 'retained_native_decision_to_physical_target',
        'feature_version': 'request-shape.v1',
        'features': {'message_count': len(body['messages']), 'has_tools': bool(body.get('tools'))},
        'request_hash': canonical_hash(body),
        'execution_request_hash': canonical_hash({**body, 'model': chosen.alias}),
        'currency': currency, 'funding': funding, 'action_propensity': None, 'propensity_status': 'UNAVAILABLE',
        'review_inclusion_probability': None, 'quality': 'UNMEASURED', 'candidate_kind': 'native_heuristic',
        'tier': response.get('routing_decision', {}).get('tier'),
        'supplier_price_version': chosen.supplier_price_version,
    }
    decision['decision_id'] = canonical_hash(decision)
    return chosen, decision


async def validate_choice(service, actor, deployment, decision, body):
    payload = {key: value for key, value in decision.items() if key != 'decision_id'}
    if decision.get('decision_id') != canonical_hash(payload):
        raise HTTPException(409, 'routing decision integrity changed')
    if (decision.get('organization_id'), decision.get('actor_user_id'), decision.get('workspace_id'),
            decision.get('account_id'), decision.get('chosen_deployment_id'),
            decision.get('chosen_revision'), decision.get('chosen_config_fingerprint')) != (
            actor.organization_id, actor.user_id, actor.workspace_id, deployment.account_id,
            deployment.deployment_id, deployment.revision, deployment.config_fingerprint):
        raise HTTPException(409, 'routing decision binding changed')
    if (decision.get('execution_request_hash') != canonical_hash(body)
            or decision.get('currency') != deployment.currency
            or decision.get('funding') != deployment.funding):
        raise HTTPException(409, 'routing request or financial scope changed')
    eligible = [d for d in await service.authorized_deployments(actor)
                if d.account_id == deployment.account_id and d.deployment_id in decision['candidate_ids']]
    if candidate_identity(eligible) != decision['candidate_identity']:
        raise HTTPException(409, 'routing candidate revision changed')
