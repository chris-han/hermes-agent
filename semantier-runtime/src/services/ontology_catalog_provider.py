from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
from urllib.parse import quote

import httpx

from contracts.ontology_catalog import (
    CatalogProviderError,
    OntologyCatalogCandidate,
    OntologyCatalogMetadata,
    OntologyCatalogPayload,
    parse_catalog_external_ref,
)
from contracts.ontology_seed_query import OntologySeedQuery


_ONTOLOGY_SUFFIXES = {".owl", ".rdf", ".ttl", ".nt", ".n3", ".trig", ".jsonld"}


@dataclass(frozen=True)
class HuggingFaceOntologyCatalogConfig:
    allowed_dataset_repo_ids: tuple[str, ...]
    max_results: int = 10
    max_files_per_repository: int = 100
    request_timeout_seconds: float = 10.0
    search_deadline_seconds: float = 30.0
    max_response_bytes: int = 5_000_000
    max_redirects: int = 5
    api_base_url: str = "https://huggingface.co"

    def __post_init__(self) -> None:
        if not self.allowed_dataset_repo_ids:
            raise ValueError("at least one allowed dataset repository is required")
        if min(
            self.max_results, self.max_files_per_repository,
            self.max_response_bytes, self.max_redirects,
        ) <= 0:
            raise ValueError("catalog bounds must be positive")
        if min(self.request_timeout_seconds, self.search_deadline_seconds) <= 0:
            raise ValueError("catalog timeouts must be positive")
        base = httpx.URL(self.api_base_url)
        if base.scheme != "https" or base.host != "huggingface.co":
            raise ValueError("api_base_url must be the official HTTPS Hugging Face host")


class HuggingFaceOntologyCatalogProvider:
    def __init__(
        self,
        config: HuggingFaceOntologyCatalogConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=config.request_timeout_seconds, follow_redirects=True
        )
        self._metadata_cache: dict[str, OntologyCatalogMetadata] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: OntologySeedQuery) -> tuple[OntologyCatalogCandidate, ...]:
        query = OntologySeedQuery.model_validate(query.model_dump(mode="json"))
        try:
            async with asyncio.timeout(self._config.search_deadline_seconds):
                return await self._search_repositories(query=query)
        except TimeoutError as exc:
            raise CatalogProviderError("ontology catalog search deadline exceeded") from exc

    async def _search_repositories(
        self, query: OntologySeedQuery | None
    ) -> tuple[OntologyCatalogCandidate, ...]:
        results: list[OntologyCatalogCandidate] = []
        terms: set[str] = set()
        if query is not None:
            hypothesis = query.competency_hypothesis
            terms = set(re.findall(r"[a-z0-9]+", " ".join((
                hypothesis.domain, *hypothesis.intended_use,
                *hypothesis.required_competencies,
                *hypothesis.required_definition_families,
                *hypothesis.required_normative_dimensions,
                *hypothesis.required_relation_families,
            )).lower()))
        for repo_id in self._config.allowed_dataset_repo_ids:
            raw = await self._get_bytes(f"{self._config.api_base_url}/api/datasets/{repo_id}")
            document, commit, siblings, description, license_name = self._decode_metadata(
                raw, expected_repo=repo_id
            )
            for path in siblings:
                if not self._valid_path(path):
                    continue
                if PurePosixPath(path).suffix.lower() not in _ONTOLOGY_SUFFIXES:
                    continue
                lexical_material = f"{path} {description or ''}".lower()
                score = sum(1 for term in terms if term in lexical_material)
                external_ref = f"hf-dataset://{repo_id}@{commit}/{path}"
                candidate = OntologyCatalogCandidate(
                    external_ref=external_ref, upstream_id=repo_id, commit=commit,
                    file_path=path, license=license_name,
                    description=description if isinstance(description, str) else None,
                    provenance=f"huggingface-dataset:{repo_id}@{commit}", lexical_score=score,
                )
                results.append(candidate)
        results.sort(key=lambda item: (-item.lexical_score, item.upstream_id, item.file_path))
        bounded = tuple(results[: self._config.max_results])
        self._metadata_cache.update({
            item.external_ref: OntologyCatalogMetadata(**item.model_dump()) for item in bounded
        })
        return bounded

    async def metadata(self, external_ref: str) -> OntologyCatalogMetadata:
        repo, commit, path = self._parse_ref(external_ref)
        cached = self._metadata_cache.get(external_ref)
        if cached is not None:
            return cached
        raw = await self._get_bytes(
            f"{self._config.api_base_url}/api/datasets/{repo}/revision/{commit}"
        )
        _, verified_commit, siblings, description, license_name = self._decode_metadata(
            raw, expected_repo=repo, expected_commit=commit
        )
        if path not in siblings:
            raise CatalogProviderError("catalog metadata does not contain requested ontology file")
        metadata = OntologyCatalogMetadata(
            external_ref=external_ref, upstream_id=repo, commit=commit, file_path=path,
            license=license_name, description=description,
            provenance=f"huggingface-dataset:{repo}@{verified_commit}",
        )
        self._metadata_cache[external_ref] = metadata
        return metadata

    async def fetch(self, external_ref: str) -> OntologyCatalogPayload:
        repo, commit, path = self._parse_ref(external_ref)
        encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
        content = await self._get_bytes(
            f"{self._config.api_base_url}/datasets/{repo}/resolve/{commit}/{encoded_path}"
        )
        return OntologyCatalogPayload(
            external_ref=external_ref, upstream_id=repo, commit=commit, file_path=path,
            content=content, sha256="sha256:" + sha256(content).hexdigest(),
        )

    def _parse_ref(self, external_ref: str) -> tuple[str, str, str]:
        try:
            repo, commit, path = parse_catalog_external_ref(external_ref)
        except ValueError as exc:
            raise CatalogProviderError("invalid external_ref") from exc
        if repo not in self._config.allowed_dataset_repo_ids or not self._valid_path(path):
            raise CatalogProviderError("invalid external_ref")
        if PurePosixPath(path).suffix.lower() not in _ONTOLOGY_SUFFIXES:
            raise CatalogProviderError("invalid external_ref")
        return repo, commit, path

    def _decode_metadata(
        self,
        raw: bytes,
        *,
        expected_repo: str,
        expected_commit: str | None = None,
    ) -> tuple[dict, str, list[str], str | None, str | None]:
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise CatalogProviderError(f"invalid metadata for configured repository {expected_repo}") from exc
        if not isinstance(document, dict):
            raise CatalogProviderError("catalog metadata must be a JSON object")
        if document.get("id") != expected_repo:
            raise CatalogProviderError("catalog metadata repository identity mismatch")
        commit = document.get("sha")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise CatalogProviderError("catalog metadata did not provide an exact commit")
        if expected_commit is not None and commit != expected_commit:
            raise CatalogProviderError("catalog metadata commit identity mismatch")
        description = document.get("description")
        if description is not None and not isinstance(description, str):
            raise CatalogProviderError("catalog metadata description must be a string")
        card_data = document.get("cardData") or {}
        if not isinstance(card_data, dict):
            raise CatalogProviderError("catalog metadata cardData must be an object")
        license_name = card_data.get("license")
        if license_name is not None and not isinstance(license_name, str):
            raise CatalogProviderError("catalog metadata license must be a string")
        raw_siblings = document.get("siblings")
        if not isinstance(raw_siblings, list):
            raise CatalogProviderError("catalog metadata siblings must be a list")
        siblings: list[str] = []
        for sibling in raw_siblings[: self._config.max_files_per_repository]:
            if not isinstance(sibling, dict) or not isinstance(sibling.get("rfilename"), str):
                raise CatalogProviderError("catalog metadata sibling must contain a string filename")
            siblings.append(sibling["rfilename"])
        return document, commit, siblings, description, license_name

    @staticmethod
    def _valid_path(path: str) -> bool:
        parsed = PurePosixPath(path)
        return bool(path) and not path.startswith("/") and ".." not in parsed.parts and "\\" not in path

    async def _get_bytes(self, url: str) -> bytes:
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout(self._config.request_timeout_seconds):
                request = self._client.build_request("GET", url)
                origin_host = request.url.host
                for redirect_count in range(self._config.max_redirects + 1):
                    self._validate_http_destination(request.url, redirect=redirect_count > 0)
                    cross_host = request.url.host != origin_host
                    if cross_host:
                        for header in ("authorization", "proxy-authorization", "cookie"):
                            request.headers.pop(header, None)
                    response = await self._client.send(
                        request, stream=True, follow_redirects=False,
                        auth=None if cross_host else httpx.USE_CLIENT_DEFAULT,
                    )
                    content = await self._read_bounded(response)
                    if not response.is_redirect:
                        response.raise_for_status()
                        return content
                    if redirect_count >= self._config.max_redirects:
                        raise CatalogProviderError("ontology catalog redirect limit exceeded")
                    location = response.headers.get("location")
                    if not location:
                        raise CatalogProviderError("ontology catalog redirect lacked a destination")
                    destination = response.url.join(location)
                    self._validate_http_destination(destination, redirect=True)
                    headers = dict(request.headers)
                    headers.pop("host", None)
                    await response.aclose()
                    response = None
                    request = self._client.build_request("GET", destination, headers=headers)
                raise CatalogProviderError("ontology catalog redirect limit exceeded")
        except TimeoutError as exc:
            raise CatalogProviderError("ontology catalog request timeout") from exc
        except CatalogProviderError:
            raise
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            raise CatalogProviderError(f"ontology catalog request failed: {exc}") from exc
        finally:
            if response is not None:
                await response.aclose()

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self._config.max_response_bytes:
                raise CatalogProviderError("catalog response exceeded configured byte cap")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _validate_http_destination(url: httpx.URL, *, redirect: bool) -> None:
        host = (url.host or "").lower()
        permitted_host = (
            host == "huggingface.co"
            or host.endswith(".huggingface.co")
            or host.endswith(".hf.co")
        )
        if url.scheme != "https" or not permitted_host:
            label = "redirect destination" if redirect else "request destination"
            raise CatalogProviderError(f"ontology catalog {label} is not approved")
