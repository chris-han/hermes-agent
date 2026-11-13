"""Cloud metadata stays with its authenticated discovery snapshot, never inferred from ids."""
from hermes_cli.desktop_model_discovery import read_provider_snapshot


def test_cloud_snapshot_retains_safe_metadata_across_cache_and_refresh_failure(tmp_path,monkeypatch):
    from agent import semantier_gateway
    payload={'object':'list','data':[{'id':'gpt-4o','label':'GPT-4o','provider_id':'AZURE_OPENAI','provider_display':'Azure OpenAI','provider_aliases':['azure','az','openai'],'source_model_id':'deployment-gpt4o','customer_price_version':'retail-2','offer_revision':4,'publication_revision':2,'account_id':'account-a','currency':'CNY','api_key':'must-not-leak'}]}
    monkeypatch.setattr(semantier_gateway,'cloud_catalog',lambda **kw:payload)
    args=dict(provider='semantier-cloud',base_url='https://cloud.example/v1',api_key='account-a-credential',source_scope='connection-a',adapter='cloud',cache_path=tmp_path/'models.json')
    fresh=read_provider_snapshot(**args,refresh=True)
    assert fresh.models==['gpt-4o']
    assert fresh.model_profiles['gpt-4o']['provider_display']=='Azure OpenAI'
    assert fresh.model_profiles['gpt-4o']['customer_price_version']=='retail-2'
    assert 'must-not-leak' not in (tmp_path/'models.json').read_text()
    cached=read_provider_snapshot(**args)
    assert cached.model_profiles==fresh.model_profiles and cached.source_id==fresh.source_id
    def failed(**kw):raise OSError('offline')
    monkeypatch.setattr(semantier_gateway,'cloud_catalog',failed)
    stale=read_provider_snapshot(**args,refresh=True)
    assert stale.state=='stale' and stale.model_profiles==fresh.model_profiles
    other=read_provider_snapshot(**{**args,'api_key':'account-b-credential'})
    assert other.models is None and other.model_profiles is None


def test_cloud_duplicate_ids_do_not_merge_distinct_account_metadata(tmp_path,monkeypatch):
    from agent import semantier_gateway
    monkeypatch.setattr(semantier_gateway,'cloud_catalog',lambda **kw:{'object':'list','data':[{'id':'same','account_id':'a'},{'id':'same','account_id':'b'}]})
    result=read_provider_snapshot(provider='semantier-cloud',base_url='https://cloud.example/v1',api_key='fixture-key',source_scope='connection-a',adapter='cloud',cache_path=tmp_path/'models.json',refresh=True)
    assert result.state=='error' and result.models is None
