"""Signed static Semantier distribution; installation remains owned by Hermes."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tools.skills_hub_models import SkillBundle, SkillMeta, SkillSource, _parse_frontmatter, _referenced_support_paths, _validate_bundle_rel_path

BASE = 'https://assets.semantier.cn/skills/v1/'
_SLUG = re.compile(r'[a-z][a-z0-9-]{0,63}\Z')
_VERSION = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z')
_LOCK = threading.Lock()
_FILE_LOCK_HOLDER = threading.local()


class RegistryError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')

    def diagnostic(self):
        return {'source': 'semantier', 'code': self.code, 'retryable': self.code in {'timeout', 'unavailable'}, 'message': str(self)}


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def _timestamp(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError('UTC timestamp required')
    return dt


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate JSON member')
        out[key] = value
    return out


def _manifest(manifest):
    if not isinstance(manifest, dict) or not 1 <= len(manifest) <= 256 or 'SKILL.md' not in manifest:
        raise ValueError('complete manifest requires SKILL.md')
    seen, total = set(), 0
    for path, item in manifest.items():
        if _validate_bundle_rel_path(path) != path or '\\' in path:
            raise ValueError('noncanonical path')
        for part in path.split('/'):
            if (part.endswith(('.', ' ')) or any(c in part for c in '<>"|?*') or
                    any(ord(c) < 32 for c in part) or re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', unicodedata.normalize('NFKC', part.split('.')[0]))):
                raise ValueError('unsafe Windows path')
        folded = unicodedata.normalize('NFC', path).casefold()
        if folded in seen:
            raise ValueError('casefold collision')
        seen.add(folded)
        if set(item) != {'size', 'sha256'} or type(item['size']) is not int or not 0 <= item['size'] <= 10*1024*1024:
            raise ValueError('invalid file size or entry')
        if not re.fullmatch('[a-f0-9]{64}', item['sha256']):
            raise ValueError('invalid sha256')
        total += item['size']
    if total > 50*1024*1024:
        raise ValueError('release too large')
    if any(unicodedata.normalize('NFC', '/'.join(p.split('/')[:n])).casefold() in seen for p in manifest for n in range(1, len(p.split('/')))):
        raise ValueError('file/directory collision')


def _read(url, limit):
    # No redirects: neither credentials nor package requests can change authority.
    if urlsplit(url).hostname != 'assets.semantier.cn' or not url.startswith(BASE):
        raise RegistryError('invalid_catalog', 'off-host URL')
    from tools.url_safety import create_ssrf_safe_client
    from tools.website_policy import check_website_access
    if check_website_access(url):
        raise RegistryError('unavailable', 'registry blocked by website policy')
    try:
        with create_ssrf_safe_client(timeout=15, follow_redirects=False) as client, client.stream('GET', url) as response:
            if response.status_code != 200:
                raise RegistryError('unavailable', f'HTTP {response.status_code}')
            out = bytearray()
            for chunk in response.iter_bytes(chunk_size=65536):
                out.extend(chunk)
                if len(out) > limit:
                    raise RegistryError('integrity_failed', 'response exceeds bound')
            return bytes(out)
    except httpx.TimeoutException as exc:
        raise RegistryError('timeout', 'registry timed out') from exc
    except httpx.HTTPError as exc:
        raise RegistryError('unavailable', 'registry request failed') from exc


class SemantierSkillSource(SkillSource):
    SOURCE_ID = 'semantier'

    def __init__(self, *, trust=None, reader=None, cache_dir=None, now=None):
        self.trust = trust if trust is not None else json.loads(Path(__file__).with_name('semantier_skill_trust.json').read_text(encoding="utf-8-sig"))['keys']
        self.reader = reader or _read
        self.cache_dir = cache_dir
        self.now = now or (lambda: datetime.now(UTC))

    def _snapshot(self):
        if not self.trust:
            raise RegistryError('unavailable', 'Semantier publisher key has not been admitted')
        try:
            raw = self.reader(BASE + 'catalog.json', 2*1024*1024)
            if len(raw) > 2*1024*1024:
                raise ValueError('catalog too large')
            env = json.loads(raw, object_pairs_hook=_unique_object)
            if set(env) != {'schema', 'key_id', 'payload_b64', 'signature_b64'}:
                raise ValueError('unknown envelope fields')
            if type(env['schema']) is not int or env['schema'] != 1:
                raise RegistryError('unsupported_schema', 'catalog schema')
            key = self.trust.get(env['key_id'])
            if not key or key.get('revoked') or key['publisher_id'] != 'semantier':
                raise RegistryError('signature_invalid', 'unadmitted publisher key')
            if self.now() >= _timestamp(key['expires_at']):
                raise RegistryError('expired', 'publisher key expired')
            public = base64.b64decode(key['public_key_b64'], validate=True)
            if hashlib.sha256(public).hexdigest() != key['fingerprint']:
                raise RegistryError('signature_invalid', 'key fingerprint mismatch')
            payload = base64.b64decode(env['payload_b64'], validate=True)
            Ed25519PublicKey.from_public_bytes(public).verify(base64.b64decode(env['signature_b64'], validate=True), payload)
            data = json.loads(payload, object_pairs_hook=_unique_object)
            self._validate(data)
            digest = hashlib.sha256(payload).hexdigest()
            self._remember(env['key_id'], data['sequence'], digest, data['skills'])
            return data, {'catalog_digest': digest, 'key_id': env['key_id'], 'key_fingerprint': key['fingerprint'], 'publisher_verified': True, 'verified_at': self.now().isoformat()}
        except RegistryError:
            raise
        except InvalidSignature as exc:
            raise RegistryError('signature_invalid', 'signature rejected') from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise RegistryError('invalid_catalog', 'catalog contract rejected') from exc

    def _validate(self, data):
        if data['registry_id'] != 'semantier' or data['publisher']['id'] != 'semantier' or type(data['sequence']) is not int or data['sequence'] < 1:
            raise ValueError('registry identity or sequence')
        issued, expires = _timestamp(data['issued_at']), _timestamp(data['expires_at'])
        if issued > self.now() or expires <= issued:
            raise ValueError('catalog validity interval')
        if self.now() >= expires:
            raise RegistryError('expired', 'catalog expired')
        for field in ('name', 'repository', 'contact'):
            if not data['publisher'][field]:
                raise ValueError('publisher metadata')
        if not isinstance(data['skills'], list) or len(data['skills']) > 100:
            raise ValueError('skill count')
        slugs = set()
        for skill in data['skills']:
            slug = skill['slug']
            if not _SLUG.fullmatch(slug) or slug in slugs or skill['type'] != 'skill':
                raise ValueError('skill identity')
            slugs.add(slug)
            if not isinstance(skill['tags'], list) or any(not isinstance(tag, str) for tag in skill['tags']):
                raise ValueError('tags must be strings')
            if _validate_bundle_rel_path(skill['source_path']) != skill['source_path']:
                raise ValueError('source path')
            for field in ('name', 'description', 'category', 'license', 'homepage', 'source_repository', 'source_path'):
                if not isinstance(skill[field], str) or not skill[field]:
                    raise ValueError('skill metadata')
            if not re.fullmatch('(?:[a-f0-9]{40}|[a-f0-9]{64})', skill['source_commit']):
                raise ValueError('source pin')
            versions = set()
            if not 1 <= len(skill['versions']) <= 20:
                raise ValueError('version count')
            for release in skill['versions']:
                self._validate_release(release, versions)
            if skill['latest_version'] not in versions:
                raise ValueError('latest version absent')

    def _validate_release(self, release, versions):
        version = release['version']
        if not _VERSION.fullmatch(version) or version in versions or version in {'.', '..'}:
            raise ValueError('version identity')
        versions.add(version)
        if not re.fullmatch('(?:[a-f0-9]{40}|[a-f0-9]{64})', release['source_commit']):
            raise ValueError('release source pin')
        if _timestamp(release['released_at']) > self.now():
            raise ValueError('future release')
        if release['compatibility']:
            raise RegistryError('incompatible', 'unsupported compatibility requirements')
        _manifest(release['manifest'])

    def _remember(self, key_id, sequence, digest, skills):
        from tools.skills_hub import _index_cache_dir
        root = self.cache_dir if self.cache_dir is not None else _index_cache_dir()
        path = Path(root) / 'semantier-sequences.json'
        from hermes_cli.auth import _file_lock
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, _file_lock(path.with_suffix('.lock'), _FILE_LOCK_HOLDER, 10.0, 'Catalog state busy'):
            state = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
            old = state.get(key_id)
            if old and (sequence < old['sequence'] or sequence == old['sequence'] and digest != old['digest']):
                raise RegistryError('rollback_detected', 'catalog rollback or equivocation')
            releases = state.get('releases', {})
            for skill in skills:
                for release in skill['versions']:
                    identity = skill['slug'] + '@' + release['version']
                    pinned = manifest_digest(release)
                    if identity in releases and releases[identity] != pinned:
                        raise RegistryError('integrity_failed', 'immutable release changed')
                    releases[identity] = pinned
            present = {skill['slug'] + '@' + release['version'] for skill in skills for release in skill['versions']}
            if set(releases) - present:
                raise RegistryError('invalid_catalog', 'previous releases must remain discoverable')
            state['releases'] = releases
            state[key_id] = {'sequence': sequence, 'digest': digest}
            path.parent.mkdir(parents=True, exist_ok=True)
            from utils import atomic_json_write
            atomic_json_write(path, state)

    def _resolve(self, identifier):
        match = re.fullmatch(r'semantier/semantier/([a-z][a-z0-9-]{0,63})(?:@([A-Za-z0-9][A-Za-z0-9._-]{0,63}))?', identifier)
        if not match:
            return None
        data, proof = self._snapshot()
        for skill in data['skills']:
            if skill['slug'] == match[1]:
                version = match[2] or skill['latest_version']
                for release in skill['versions']:
                    if release['version'] == version:
                        return skill, release, proof
        raise RegistryError('not_found', 'release not found')

    def _meta(self, skill, release, proof, identifier=None):
        revision = manifest_digest(release['manifest'])
        extra = {**proof, 'publisher_id': 'semantier', 'publisher': 'Semantier', 'version': release['version'], 'source_revision': revision, 'source_commit': release['source_commit'], 'source_url': BASE + f"packages/semantier/{skill['slug']}/{release['version']}/{revision}/", 'license': skill['license'], 'released_at': release['released_at']}
        return SkillMeta(skill['name'], skill['description'], self.SOURCE_ID, identifier or f"semantier/semantier/{skill['slug']}", 'community', repo=skill['source_repository'], tags=skill['tags'], extra=extra)

    def search(self, query, limit=10, *, category=None, publisher=None):
        data, proof = self._snapshot()
        results = []
        for skill in sorted(data['skills'], key=lambda s: s['slug']):
            if publisher not in (None, 'semantier') or (category is not None and skill['category'] != category):
                continue
            if query.casefold() in ' '.join([skill['name'], skill['description'], *skill['tags']]).casefold():
                release = next(r for r in skill['versions'] if r['version'] == skill['latest_version'])
                results.append(self._meta(skill, release, proof))
        return results[:limit]

    def inspect(self, identifier):
        resolved = self._resolve(identifier)
        return self._meta(*resolved, identifier=identifier) if resolved else None

    def current_revision(self, identifier):
        meta = self.inspect(identifier)
        return meta.extra['source_revision'] if meta else ''

    def fetch(self, identifier):
        resolved = self._resolve(identifier)
        if not resolved:
            return None
        skill, release, _proof = resolved
        meta = self._meta(*resolved, identifier=identifier)
        files = {}
        for path, item in release['manifest'].items():
            raw = self.reader(meta.extra['source_url'] + quote(path, safe='/'), item['size'])
            if len(raw) != item['size'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise RegistryError('integrity_failed', 'artifact bytes differ from signed manifest')
            files[path] = raw
        try:
            text = files['SKILL.md'].decode('utf-8')
        except UnicodeDecodeError as exc:
            raise RegistryError('integrity_failed', 'SKILL.md must be UTF-8') from exc
        if _parse_frontmatter(text).get('name') != 'semantier--' + skill['slug']:
            raise RegistryError('invalid_catalog', 'release frontmatter must match source-qualified runtime name')
        refs = _referenced_support_paths(text)
        if refs is None or not refs.issubset(files):
            raise RegistryError('integrity_failed', 'referenced support files missing')
        return SkillBundle('semantier--' + skill['slug'], files, self.SOURCE_ID, identifier, 'community', {**meta.extra, 'integrity_verified': True, 'verified_files': release['manifest']})


def check_install_conflict(bundle, install_dir):
    """Only the recorded Semantier target may be replaced; force grants no origin change."""
    from tools.skills_hub import HubLockFile, _skills_dir
    from tools.skills_tool import _skill_catalog
    entries = HubLockFile().list_installed()
    owned = next((e for e in entries if e['name'] == bundle.name and e['source'] == bundle.source == 'semantier' and e['identifier'] == bundle.identifier and e['install_path'] == install_dir.relative_to(_skills_dir()).as_posix()), None)
    for entry in entries:
        if entry['name'].casefold() == bundle.name.casefold() and entry is not owned:
            raise RegistryError('name_conflict', 'runtime name belongs to ' + entry['source'])
    for skill in _skill_catalog(skip_disabled=True, include_hidden=True):
        if skill['name'].casefold() == bundle.name.casefold() and (not owned or Path(skill['path']).parent.resolve() != install_dir.resolve()):
            raise RegistryError('name_conflict', 'runtime name already exists')
    if install_dir.parent.exists():
        collisions = [path for path in install_dir.parent.iterdir() if path.name.casefold() == bundle.name.casefold() and path.name != bundle.name]
        if collisions:
            raise RegistryError('name_conflict', 'case-insensitive disk target exists')
    if install_dir.exists() and not owned:
        raise RegistryError('name_conflict', 'unmanaged target exists')
    if owned:
        old = owned.get('metadata', {})
        if old.get('version') != bundle.metadata.get('version'):
            # Release chronology is signed; versions are opaque and never lexically ordered.
            if not old.get('released_at') or _timestamp(old['released_at']) >= _timestamp(bundle.metadata['released_at']):
                raise RegistryError('incompatible', 'automatic downgrade refused')
