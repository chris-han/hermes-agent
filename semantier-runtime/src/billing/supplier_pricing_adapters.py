"""Explicit supplier source registrations for pricing management."""
from billing.supplier_pricing_ingestion import MODELS, capture_sources
from billing.supplier_pricing_projection import project_native_map


class AlibabaModelStudioAdapter:
    provider_id = 'alibaba'
    source_ref = 'alibaba-model-studio-cn-beijing'

    def __init__(self, capture=capture_sources):
        self._capture = capture

    def describe(self):
        return {
            'provider_id': self.provider_id,
            'adapter_id': 'alibaba-model-studio-v1',
            'name': 'Alibaba Cloud Model Studio',
            'sources': [{
                'source_ref': self.source_ref,
                'region': 'cn-beijing',
                'access_method': 'model-studio',
                'partition': 'cn-beijing',
                'models': list(MODELS),
                'capabilities': ['collect', 'normalize', 'project'],
            }],
        }

    async def collect(self, base, models, *, client=None, api_key=None):
        return await self._capture(base, models, client=client, api_key=api_key)

    def excluded_native_keys(self, projection):
        return ['dashscope/'+name for name in projection['excluded']]

    def project(self, snapshot, base_map, fx):
        return project_native_map(snapshot, base_map, fx)


_ADAPTERS = (AlibabaModelStudioAdapter(),)


def registered_adapters():
    return _ADAPTERS


def adapter_for(provider_id, source_ref, *, adapters=None):
    for adapter in _ADAPTERS if adapters is None else adapters:
        if adapter.provider_id == provider_id and adapter.source_ref == source_ref:
            return adapter
    raise ValueError('UNSUPPORTED_SUPPLIER_SOURCE')
