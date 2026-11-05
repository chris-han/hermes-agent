"""R2 publication target with separate private control custody."""
from contextlib import contextmanager, nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4
from xml.etree import ElementTree as ET

from .artifact_io import OperationError, safe_file
from .resumable_r2_readback import verify_resumable

from contracts.client_operations import canonical

MAX_BYTES = 10_000_000_000
MAX_RELEASES = 3
# Cover the simultaneous 60,000-byte retention projection, 16,000-byte upload
# journal, and bounded publication lock/head metadata throughout publication.
CONTROL_ALLOCATION_BYTES = 128 * 1024
LOCK_KEY = "_control/publication.lock"
# Explicitly approved one-time retirement; unknown keys outside this set are never inferred.
APPROVED_LEGACY_RETIREMENT = {
    'releases/win32/stable/stable.appinstaller': 466,
    'releases/win32/stable/SemantierAgent-0.0.68-win-x64.msix': 2261279093,
    'releases/win32/stable/semantier-setup-ch2.exe': 475136,
    'releases/channel-builds/00000000000000000000000000000001/qualification/canary.zip': 186,
    'releases/channel-builds/00000000000000000000000000000001/chunks/254bcc3fc4f27172636df4bf32de9f107f620d559b20d760197e452b97453917/000000-254bcc3fc4f27172636df4bf32de9f107f620d559b20d760197e452b97453917.part': 134217728,
}
RETENTION_KEY = "_control/retention.json"
BUILD = re.compile(r"^releases/channel-builds/([a-f0-9]{32})/")

# Exact local release verified by the Windows producer and independent
# SignTool checks. This is manual-download provenance, not CI qualification.
MANUAL_064 = {
    'build': '3d722e4ff1c44b48b908d1ebb606453a', 'version': '0.0.64.0',
    'source': '1bf40293e79e33057a2f23b3274ed57dd7a6b0cb',
    'controller': 'f12684540216d8af65fc6a4d1aefeb5752cf74b1',
    'producer_sha256': '723b68ea6b443e4abab48acff8705cf18f7874aee056926e017d1fdbb4e39ad0',
    'receipt_sha256': 'bef65bc1d100262c14564d2d12137f957ad41935e5092d0e99c9c7920837d56c',
    'graph_sha256': 'cfbf8c7582ecd2818f0c094acbc3177364bb61c74b1c0185ab0585e5eb3ca6d5',
    'zip_sha256': 'b970fd1b60fd181d8ae29d3eb88f48d217279ef3b55f2c454525683ea3d81e9e',
    'zip_size': 2261279211,
    'installer_sha256': 'edb2398304df68d69e40d6a1751566a94dbf245da10d4930713b4604504cec37',
    'installer_size': 476664, 'installer_source': '2bd9882204c9466ce353a298b3b4c0f5a1308d97',
    'installer_filename': 'SemantierAgent-0.0.64-Setup-1.1.1.exe',
    'previous_installer': {
        'url': 'https://update2.semantier.com/releases/channel-builds/3d722e4ff1c44b48b908d1ebb606453a/SemantierAgent-0.0.64-Setup-1.1.0.exe',
        'size': 476664, 'sha256': 'e7e3e61b387d2d454204c71a2029ef84158d4afdaa06bff911ad3c23826d8b92',
    },
}


# Exact independently signed monolithic 0.0.77 candidate and separate setup.
MANUAL_077 = {
    'build': '9e1ca1e9fad44d35ac9b13a2f09747e5', 'version': '0.0.77.0',
    'source': '95e1f63d9f20fd05c53f87b78468b27488ac0d15',
    'controller': 'b02455456b1385aa6b2cdfaac2ca0273cbd1f6be',
    'producer_sha256': '8ab358cc23150c1fdb5b8df3887abb9cd69e938489a18e1545911acc59a7237d',
    'receipt_sha256': '060e2b1e5827bbb7a4c0f04733f5429ab521b83e0e7c03b8f80960f4af0c74d8',
    'msix_sha256': '3c0fd642db1b35fa637c356cdd01871e057efcdd54e0b9b94646985d8e7c14a4',
    'zip_sha256': '59ad5f557babc3905835f239c502ea44ad5aa43207603ffff8cbcb757998c229',
    'zip_size': 2264240015, 'zip_filename': 'SemantierAgent-0.0.77-win-x64.zip',
    'installer_sha256': '9730dc39cb727c7306c3b834f63e067e9109bdd9d7cdbea083c767d88c8f4ec2',
    'installer_size': 479736, 'installer_source': '86fbadc18c286aa06fb7c0ea73613366cfffd976',
    'installer_filename': 'SemantierAgent-0.0.77-Setup.exe',
}


def validate_manual_077(incoming):
    """Admit only the reviewed local-signed monolithic archive and setup."""
    from zipfile import ZipFile, BadZipFile
    from cryptography import x509
    from datetime import timezone
    base='manual-local/'+MANUAL_077['build']+'/'
    names=('result.json','receipt.json','release.zip','installer.exe')
    paths={name:safe_file(incoming,base+name) for name in names}
    for name, field in (('result.json','producer_sha256'),('receipt.json','receipt_sha256'),
                        ('release.zip','zip_sha256'),('installer.exe','installer_sha256')):
        path=paths[name]
        if not path.is_file() or path.is_symlink():raise OperationError(422,'manual_artifact_missing')
        with path.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != MANUAL_077[field]:raise OperationError(422,'manual_artifact_hash_mismatch')
    if paths['release.zip'].stat().st_size != MANUAL_077['zip_size'] or paths['installer.exe'].stat().st_size != MANUAL_077['installer_size']:
        raise OperationError(422,'manual_artifact_size_mismatch')
    producer=json.loads(paths['result.json'].read_text(encoding='utf-8-sig'))
    receipt=json.loads(paths['receipt.json'].read_text(encoding='utf-8-sig'))
    if (producer.get('result')!='PASS' or producer.get('sourceCommit')!=MANUAL_077['source'] or
        producer.get('controllerCommit')!=MANUAL_077['controller'] or receipt.get('sourceCommit')!=MANUAL_077['source'] or
        receipt.get('packageVersion')!=MANUAL_077['version'] or receipt.get('packageName')!='Semantier.Agent.Acceptance' or
        receipt.get('publisher')!='CN=Semantier Technology, O=Semantier Technology' or receipt.get('signing')!='PASS' or
        not receipt.get('verification','').startswith('PASS:') or receipt.get('trustedTimestamp') is not False):
        raise OperationError(422,'manual_producer_mismatch')
    if receipt.get('signedSha256') != MANUAL_077['msix_sha256']:
        raise OperationError(422,'manual_package_mismatch')
    expected={'build-metadata.json','Install.bat','Install.ps1','README-Install.txt',
              'SemantierAgent-0.0.77-win-x64.msix','SemantierAgent-signing.cer','SHA256SUMS.txt'}
    try:
        with ZipFile(paths['release.zip']) as archive:
            entries=archive.namelist()
            if len(entries)!=len(expected) or set(entries)!=expected:
                raise OperationError(422,'manual_archive_inventory_mismatch')
            with archive.open('SemantierAgent-0.0.77-win-x64.msix') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=MANUAL_077['msix_sha256']:
                    raise OperationError(422,'manual_package_mismatch')
            cert_bytes=archive.read('SemantierAgent-signing.cer')
            if hashlib.sha256(cert_bytes).hexdigest()!='10a6b55e9a03bbc0dceb8d899c1776bf1fbad963c82050c16534da4983c9661d':
                raise OperationError(422,'manual_certificate_mismatch')
            cert=x509.load_der_x509_certificate(cert_bytes); now=datetime.now(timezone.utc)
            if not cert.not_valid_before_utc <= now < cert.not_valid_after_utc:
                raise OperationError(422,'manual_certificate_expired')
    except (BadZipFile, KeyError, ValueError) as error:
        raise OperationError(422,'manual_archive_invalid') from error
    return paths


def validate_manual_064(incoming):
    """Validate the exact previously signed local artifact, never a CI claim."""
    from .artifact_io import safe_file
    from zipfile import ZipFile
    base='manual-local/'+MANUAL_064['build']+'/'
    paths={name:safe_file(incoming,base+name) for name in ('result.json','receipt.json','native-graph.json','release.zip','installer.exe')}
    for name,field in (('result.json','producer_sha256'),('receipt.json','receipt_sha256'),('native-graph.json','graph_sha256'),('release.zip','zip_sha256'),('installer.exe','installer_sha256')):
        path=paths[name]
        if not path.is_file() or path.is_symlink():raise OperationError(422,'manual_artifact_missing')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=MANUAL_064[field]:raise OperationError(422,'manual_artifact_hash_mismatch')
    if paths['release.zip'].stat().st_size!=MANUAL_064['zip_size']:raise OperationError(422,'manual_artifact_size_mismatch')
    if paths['installer.exe'].stat().st_size!=MANUAL_064['installer_size']:raise OperationError(422,'manual_artifact_size_mismatch')
    producer=json.loads(paths['result.json'].read_text(encoding='utf-8-sig'))
    receipt=json.loads(paths['receipt.json'].read_text(encoding='utf-8-sig'))
    if (producer.get('result')!='PASS' or producer.get('sourceCommit')!=MANUAL_064['source'] or
        producer.get('controllerCommit')!=MANUAL_064['controller'] or receipt.get('sourceCommit')!=MANUAL_064['source'] or
        receipt.get('packageVersion')!=MANUAL_064['version'] or receipt.get('packageName')!='Semantier.Agent.Acceptance' or
        receipt.get('publisher')!='CN=Semantier Technology, O=Semantier Technology' or receipt.get('signing')!='PASS' or
        not receipt.get('verification','').startswith('PASS:') or receipt.get('trustedTimestamp') is not False):
        raise OperationError(422,'manual_producer_mismatch')
    with ZipFile(paths['release.zip']) as archive:
        if len(archive.namelist())!=len(set(archive.namelist())) or hashlib.sha256(archive.read('native-graph.json')).hexdigest()!=MANUAL_064['graph_sha256']:
            raise OperationError(422,'manual_archive_graph_mismatch')
        cert_bytes=archive.read('SemantierAgent-signing.cer')
        if hashlib.sha256(cert_bytes).hexdigest()!='10a6b55e9a03bbc0dceb8d899c1776bf1fbad963c82050c16534da4983c9661d':
            raise OperationError(422,'manual_certificate_mismatch')
        from cryptography import x509
        from datetime import timezone
        cert=x509.load_der_x509_certificate(cert_bytes);now=datetime.now(timezone.utc)
        if not (cert.not_valid_before_utc <= now < cert.not_valid_after_utc):raise OperationError(422,'manual_certificate_expired')
    return paths


def validate_manual_catalog(raw,objects):
    """Catalog entries protect their exact immutable objects from retention."""
    from urllib.parse import urlsplit
    try:
        data=json.loads(raw)
        if set(data)!={'schema','releases'} or data['schema']!=1 or not isinstance(data['releases'],list) or len(data['releases'])>3:raise ValueError('schema')
        builds=set();versions=[]
        for row in data['releases']:
            if set(row)!={'version','track','identity','provenance','publishedAt','installer','zip'} or row['track']!='acceptance' or row['identity']!='Semantier.Agent.Acceptance' or row['provenance'] not in ('local-signed','ci-qualified'):
                raise ValueError('identity')
            if not re.fullmatch(r'(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)',row['version']):raise ValueError('version')
            stamp=datetime.fromisoformat(row['publishedAt'].replace('Z','+00:00'))
            if stamp.utcoffset() is None or stamp.utcoffset().total_seconds()!=0:raise ValueError('date')
            row_build=None
            for name,extension in (('installer','.exe'),('zip','.zip')):
                item=row[name]
                if set(item)!={'url','size','sha256'} or type(item['size']) is not int or item['size']<=0 or not re.fullmatch('[a-f0-9]{64}',item['sha256']):raise ValueError('artifact')
                url=urlsplit(item['url']);key=url.path.lstrip('/')
                match=re.fullmatch(r'releases/channel-builds/([a-f0-9]{32})/[A-Za-z0-9_.-]+',key)
                if url.scheme!='https' or url.netloc!='update2.semantier.com' or url.query or url.fragment or not match or not key.endswith(extension) or objects.get(key)!=item['size']:raise ValueError('object')
                if row_build is not None and row_build!=match[1]:raise ValueError('build')
                row_build=match[1]
            if row_build in builds:raise ValueError('duplicate')
            builds.add(row_build);versions.append(tuple(map(int,row['version'].split('.'))))
        if versions!=sorted(versions,reverse=True) or len(versions)!=len(set(versions)):raise ValueError('order')
        return data,builds
    except (ValueError,TypeError,KeyError,AttributeError,json.JSONDecodeError) as error:
        raise OperationError(503,'manual_catalog_unresolved') from error


def capacity_plan(objects, manifests, protected, incoming_size=0, incoming_build=None, *, admission_order=None):
    """Conservative allocation includes duplicate/retry uploads and all unknown objects."""
    if type(incoming_size) is not int or incoming_size < 0:
        raise OperationError(422, "invalid_r2_allocation")
    groups = {}
    total = 0
    for key, size in objects.items():
        if type(size) is not int or size < 0:
            raise OperationError(503, "invalid_r2_inventory")
        total += size
        match = BUILD.match(key)
        if match:
            group = groups.setdefault(match[1], {"keys": [], "size": 0})
            group["keys"].append(key)
            group["size"] += size
    protected = set(protected) | ({incoming_build} if incoming_build else set())
    # Missing admission provenance still prevents deletion. It need not prevent
    # an allocation whose complete inventory fits without deleting anything.
    if len(set(groups) | protected) <= MAX_RELEASES and total + incoming_size < MAX_BYTES:
        return [], total + incoming_size
    admission_order = admission_order or {}
    def order(build):
        entry = admission_order.get(build, {})
        try:
            stamp = datetime.fromisoformat(entry['first_admitted_at'].replace('Z', '+00:00'))
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0 or entry['release_id'] != build:
                raise ValueError('invalid admission')
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            raise OperationError(503, 'r2_retention_order_unavailable') from error
        return stamp, build
    ordered = sorted(groups, key=order, reverse=True)
    keep = set(protected)
    for build in ordered:
        if len(keep) < MAX_RELEASES:
            keep.add(build)
    if len(keep) > MAX_RELEASES:
        raise OperationError(409, "r2_protected_release_count_exceeded")
    delete = []
    for build in reversed(ordered):
        if build not in keep or (total + incoming_size >= MAX_BYTES and build not in protected):
            delete.extend(groups[build]["keys"])
            total -= groups[build]["size"]
            keep.discard(build)
    if total + incoming_size >= MAX_BYTES:
        raise OperationError(413, "r2_storage_budget_exceeded")
    return delete, total + incoming_size


class R2Target:
    def __init__(self, transport, creds, base, bucket, *, control_target=None):
        self.transport, self.creds, self.base, self.bucket = transport, creds, base, bucket
        if control_target and (control_target.control_target or
                (control_target.base, control_target.bucket) == (base, bucket)):
            raise ValueError('R2 private control requires a separate non-nested bucket')
        self.control_target = control_target

    @classmethod
    def configured(cls, *, public_delivery=True):
        from agents.webapi.deps import ensure_repo_hermes_agent_on_path
        ensure_repo_hermes_agent_on_path()
        from scripts.releases import r2
        if any(not os.environ.get(key) for key in ('CLOUDFLARE_R2_ACCESS_KEY_ID','CLOUDFLARE_R2_SECRET_ACCESS_KEY','CLOUDFLARE_R2_ACCOUNT_ID','CLOUDFLARE_R2_BUCKET')):
            raise OperationError(503, 'delivery_target_credentials_unavailable')
        creds, base, bucket = r2.credentials()
        control = None
        if public_delivery:
            fields = ('CLOUDFLARE_R2_CONTROL_BUCKET', 'CLOUDFLARE_R2_CONTROL_ACCESS_KEY_ID',
                      'CLOUDFLARE_R2_CONTROL_SECRET_ACCESS_KEY')
            if any(not os.environ.get(key) for key in fields):
                raise OperationError(503, 'delivery_private_control_credentials_unavailable')
            control = cls(r2, {'access_key_id': os.environ[fields[1]], 'secret_key': os.environ[fields[2]]},
                          base, os.environ[fields[0]])
            r2.R2Scope.configured().bucket_url(base, control.bucket)
        return cls(r2, creds, base, bucket, control_target=control)

    def _retirement_head(self, key):
        try:
            response = self.request('HEAD', key)
            return {'size': int(response.header('content-length')), 'etag': response.header('etag')}
        except self.transport.R2RequestError as error:
            if error.status == 404:
                return None
            raise

    def snapshot_approved_retirement(self):
        """Bind the operator-approved exact keys to the current remote versions."""
        objects, _, heads = self.inventory()
        projection = self.retention()
        selected = {}
        for key, size in APPROVED_LEGACY_RETIREMENT.items():
            if objects.get(key) != size:
                raise OperationError(409, 'retirement_object_changed')
            match = BUILD.match(key)
            if match and match[1] in heads | set(projection['inflight']):
                raise OperationError(409, 'retirement_object_protected')
            actual = self._retirement_head(key)
            if not actual or actual['size'] != size or not actual['etag']:
                raise OperationError(409, 'retirement_object_changed')
            selected[key] = actual
        return selected

    def retire_approved_objects(self, owner, reservation, snapshot, authorize):
        """Idempotently retire exact approved versions under the existing lock."""
        if owner.get('kind') != 'approved-legacy-retirement' or reservation.get('owner') != owner:
            raise OperationError(409, 'retirement_reservation_invalid')
        if set(snapshot) != set(APPROVED_LEGACY_RETIREMENT):
            raise OperationError(409, 'retirement_snapshot_invalid')
        with self.locked(owner=owner, reservation=reservation) as held:
            objects, _, heads = self.inventory()
            projection = self.retention()
            # Every still-live manifest root must be examined before any delete.
            from urllib.parse import unquote
            retired_manifest = 'releases/win32/stable/stable.appinstaller'
            for root in objects:
                if root == retired_manifest or root.startswith('_control/'):
                    continue
                if root.endswith(('.json', '.appinstaller')):
                    raw, _ = self.read(root)
                    if raw is None:
                        raise OperationError(409, 'retirement_reference_changed')
                    text = unquote(raw.decode('utf-8'))
                    if any(key in text for key in snapshot):
                        raise OperationError(409, 'retirement_object_referenced')
            present = []
            for key, expected in snapshot.items():
                if set(expected) != {'size', 'etag'} or expected['size'] != APPROVED_LEGACY_RETIREMENT[key] or not expected['etag']:
                    raise OperationError(409, 'retirement_snapshot_invalid')
                match = BUILD.match(key)
                if match and match[1] in heads | set(projection['inflight']):
                    raise OperationError(409, 'retirement_object_protected')
                actual = self._retirement_head(key)
                if actual is not None and actual != expected:
                    raise OperationError(409, 'retirement_object_changed')
                if actual is not None:
                    present.append(key)
            # Remove the retired entry point before its payload; catalog stays intact.
            present.sort(key=lambda key: (key != retired_manifest, key))
            deleted = []
            for key in present:
                authorize(); self.verify_reservation(held)
                self.request('DELETE', key, headers={'If-Match': snapshot[key]['etag']})
                if self._retirement_head(key) is not None:
                    raise OperationError(503, 'retirement_readback_unconfirmed')
                deleted.append(key)
            return {'deleted_keys': deleted}

    def url(self, key):
        return self.transport.R2Scope.configured().object_url(self.base, self.bucket, key)

    def stage_manual_064(self, incoming, owner, reservation, authorize):
        return self._stage_manual(validate_manual_064(incoming), MANUAL_064, owner, reservation, authorize)

    def stage_manual_077(self, incoming, owner, reservation, authorize):
        return self._stage_manual(validate_manual_077(incoming), MANUAL_077, owner, reservation, authorize)

    def _stage_manual(self, sources, facts, owner, reservation, authorize):
        """Publish the exact reviewed artifacts before promoting the catalog."""
        build = facts['build']
        prefix = 'releases/channel-builds/' + build + '/'
        artifacts = {
            'zip': (prefix + facts.get('zip_filename', 'SemantierAgent-0.0.64-win-x64.zip'), sources['release.zip'], facts['zip_size'], facts['zip_sha256']),
            'installer': (prefix + facts.get('installer_filename', 'SemantierAgent-0.0.64-Setup.exe'), sources['installer.exe'], facts['installer_size'], facts['installer_sha256']),
        }
        if owner.get('kind') != 'manual_download' or owner.get('release_id') != build or reservation.get('owner') != owner:
            raise OperationError(409, 'manual_reservation_invalid')
        authorize()
        with self.locked(owner=owner, reservation=reservation) as held:
            objects, manifests, heads = self.inventory()
            projection = self.retention()
            catalog_key = 'releases/manual/catalog.json'
            raw, etag = self.read(catalog_key)
            catalog = validate_manual_catalog(raw, objects)[0] if raw is not None else {'schema': 1, 'releases': []}
            for key, _, size, _ in artifacts.values():
                if key in objects and objects[key] != size:
                    raise OperationError(409, 'manual_remote_conflict')
            # Three multipart attempts can coexist if an abort is unconfirmed.
            needed = sum((3 if name == 'zip' else 1) * size for name, (key, _, size, _) in artifacts.items() if key not in objects) + CONTROL_ALLOCATION_BYTES
            delete, projected = capacity_plan(objects, manifests, heads | set(projection['inflight']), needed, build, admission_order=projection['admissions'])
            # Reclaim only the unprotected, provenance-ordered keys selected by
            # the shared retention planner, under this publication's lock.
            for key in delete:
                authorize(); self.verify_reservation(held)
                self.request('DELETE', key)
            if delete:
                self.verify_reservation(held)
                objects, manifests, heads = self.inventory()
                projection = self.retention()
                remaining, projected = capacity_plan(
                    objects, manifests, heads | set(projection['inflight']), needed, build,
                    admission_order=projection['admissions'])
                if remaining:
                    raise OperationError(409, 'manual_capacity_changed_during_cleanup')
            projection['inflight'] = sorted(set(projection['inflight']) | {build})
            authorize(); self.request('PUT', RETENTION_KEY, canonical(projection))
            for name, (key, path, size, digest) in artifacts.items():
                authorize(); self.verify_reservation(held)
                if key not in objects:
                    def before_complete():
                        authorize(); self.verify_reservation(held)
                    options = {'expected_sha256': digest}
                    if name == 'zip':
                        options.update(on_multipart_created=lambda upload_id: self.journal_upload(held, key, upload_id), before_multipart_complete=before_complete)
                    self.transport.put_object(self.creds, self.base, self.bucket, key, str(path), self.transport.amz_timestamp(), self.transport.content_type_for(key), {'If-None-Match': '*'}, **options)
                self.verify_reservation(held)
                if facts is MANUAL_077 and name == 'zip':
                    verify_resumable(self.transport, self.url(key), self.creds, size, digest)
                else:
                    self.transport.verify_remote_artifact(self.url(key), self.creds, self.transport.amz_timestamp(), size, digest, algorithm='sha256', encoding='hex')
                objects[key] = size
            item = {'version': facts['version'], 'track': 'acceptance', 'identity': 'Semantier.Agent.Acceptance', 'provenance': 'local-signed',
                    'publishedAt': datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z')}
            for name, (key, _, size, digest) in artifacts.items():
                item[name] = {'url': 'https://update2.semantier.com/' + key, 'size': size, 'sha256': digest}
            prior = next((r for r in catalog['releases'] if r['version'] == item['version']), None)
            changed = prior is None
            if prior:
                same_release = {k:v for k,v in prior.items() if k not in {'publishedAt', 'installer'}} == {k:v for k,v in item.items() if k not in {'publishedAt', 'installer'}}
                if not same_release:
                    raise OperationError(409, 'manual_catalog_conflict')
                if prior['installer'] != item['installer']:
                    # Only the reviewed predecessor may advance to this pinned revision.
                    # Its original immutable object and the application ZIP stay unchanged.
                    if prior['installer'] != facts.get('previous_installer'):
                        raise OperationError(409, 'manual_catalog_conflict')
                    item['publishedAt'] = prior['publishedAt']
                    catalog['releases'][catalog['releases'].index(prior)] = item
                    changed = True
                else:
                    item = prior
            else:
                if len(catalog['releases']) >= MAX_RELEASES:
                    raise OperationError(409, 'manual_catalog_full')
                catalog['releases'].append(item)
            if changed:
                catalog['releases'].sort(key=lambda r: tuple(map(int, r['version'].split('.'))), reverse=True)
                body = canonical(catalog)
                validate_manual_catalog(body, objects)
                authorize(); self.verify_reservation(held)
                self.request('PUT', catalog_key, body, {**({'If-Match': etag} if etag else {'If-None-Match': '*'}), 'Content-Type': 'application/json', 'Cache-Control': 'no-store'})
                saved, _ = self.read(catalog_key)
                if saved != body:
                    raise OperationError(503, 'manual_catalog_readback_failed')
            projection['admissions'].setdefault(build, {'release_id': build, 'first_admitted_at': item['publishedAt']})
            projection['inflight'] = [value for value in projection['inflight'] if value != build]
            authorize(); self.verify_reservation(held); self.request('PUT', RETENTION_KEY, canonical(projection))
            return {'release_id': build, 'version': facts['version'], 'state': 'manual_catalog_published',
                    'provenance': 'local-signed', 'source_commit': facts['source'], 'controller_commit': facts['controller'],
                    'graph_sha256': facts.get('graph_sha256'), 'installer_source_commit': facts['installer_source'],
                    'installer_sha256': facts['installer_sha256'], 'installer_size': facts['installer_size'],
                    'size': facts['zip_size'], 'sha256': facts['zip_sha256'], 'projected_bytes': projected,
                    'retention_deleted_keys': delete}

    def request(self, method, key, body=None, headers=None, query=""):
        if self.control_target and key.startswith('_control/'):
            return self.control_target.request(method, key, body, headers, query)
        if method == 'PUT' and key.startswith('releases/channels/'):
            headers = {**(headers or {}), 'Cache-Control': 'no-store', 'Content-Type': 'application/json'}
        try:
            return self.transport.signed_request(
                method, self.url(key) + query, creds=self.creds, now=self.transport.amz_timestamp(),
                body=body, body_hash=hashlib.sha256(body or b"").hexdigest(),
                content_length=len(body) if body is not None else None, extra_headers=headers)
        except self.transport.R2RequestError as error:
            if error.status in {404,412}:
                raise
            raise OperationError(503, 'delivery_target_unavailable') from error

    def listing(self, query):
        try:
            return self.transport.signed_request('GET', self.base+'/'+self.bucket+'?'+query,
                creds=self.creds, now=self.transport.amz_timestamp())
        except self.transport.R2RequestError as error:
            raise OperationError(503, 'delivery_target_unavailable') from error

    def read(self, key):
        try:
            result = self.request("GET", key)
            if len(result._body) > 4 * 1024 * 1024:
                raise OperationError(503, "r2_metadata_too_large")
            return result._body, result.header("etag")
        except self.transport.R2RequestError as error:
            if error.status == 404:
                return None, None
            raise

    @contextmanager
    def locked(self, owner=None, reservation=None):
        token = reservation or {"owner": owner or {"kind": "publisher"}, "id": uuid4().hex}
        if owner is not None and token.get("owner") != owner:
            raise OperationError(409, "r2_lock_intent_changed")
        body = canonical(token)
        try:
            self.request("PUT", LOCK_KEY, body, {"If-None-Match": "*"})
        except self.transport.R2RequestError as error:
            if error.status == 412:
                raise OperationError(409, "r2_publisher_busy") from error
            raise
        raw, _ = self.read(LOCK_KEY)
        if raw != body:
            raise OperationError(503, 'r2_lock_readback_failed')
        try:
            yield token
        finally:
            try:
                raw, etag = self.read(LOCK_KEY)
                if raw == body:
                    self.finish_upload_journal(token)
                    self.request("DELETE", LOCK_KEY, headers={"If-Match": etag})
            except self.transport.R2RequestError as error:
                raise OperationError(503, "r2_lock_release_unconfirmed") from error

    def upload_journal_key(self, reservation):
        token = reservation.get('id', '')
        if not re.fullmatch('[a-f0-9]{32}', token):
            raise OperationError(409, 'r2_upload_owner_invalid')
        return '_control/uploads/'+token+'.json'

    def journal_upload(self, reservation, key, upload_id):
        self.verify_reservation(reservation)
        journal_key = self.upload_journal_key(reservation)
        raw, _ = self.read(journal_key)
        journal = json.loads(raw) if raw else {'reservation': reservation, 'uploads': []}
        if journal.get('reservation') != reservation or not isinstance(upload_id, str) or not upload_id:
            raise OperationError(409, 'r2_upload_owner_invalid')
        match = BUILD.match(key)
        if not match or match[1] != reservation.get('owner', {}).get('release_id'):
            raise OperationError(409, 'r2_upload_scope_invalid')
        journal['uploads'].append({'key': key, 'upload_id': upload_id})
        body = canonical(journal)
        if len(body) > 16000:
            raise OperationError(413, 'r2_upload_journal_full')
        self.request('PUT', journal_key, body)
        if self.read(journal_key)[0] != body:
            raise OperationError(503, 'r2_upload_journal_unconfirmed')

    def pending_uploads(self):
        pending = self.listing('uploads=')
        root = ET.fromstring(pending._body)
        if root.findtext('{*}IsTruncated') == 'true':
            raise OperationError(503, 'r2_upload_inventory_incomplete')
        return [(row.findtext('{*}Key'), row.findtext('{*}UploadId')) for row in root.findall('{*}Upload')]

    def finish_upload_journal(self, reservation, *, abort=False):
        key = self.upload_journal_key(reservation)
        raw, etag = self.read(key)
        if raw is None:
            if self.pending_uploads():
                raise OperationError(409, 'r2_incomplete_upload_requires_recovery')
            return
        journal = json.loads(raw)
        if journal.get('reservation') != reservation:
            raise OperationError(409, 'r2_upload_owner_invalid')
        entries = journal.get('uploads')
        if not isinstance(entries, list) or len(entries) > 32:
            raise OperationError(503, 'r2_upload_journal_invalid')
        for item in entries:
            match = BUILD.match(item.get('key', ''))
            if (not match or match[1] != reservation.get('owner', {}).get('release_id')
                    or not isinstance(item.get('upload_id'), str) or not 0 < len(item['upload_id']) <= 2048):
                raise OperationError(503, 'r2_upload_journal_invalid')
        owned = {(item['key'], item['upload_id']) for item in entries}
        pending = set(self.pending_uploads())
        if not pending <= owned:
            raise OperationError(409, 'r2_unknown_upload_owner')
        if pending and not abort:
            raise OperationError(503, 'r2_upload_recovery_required')
        for upload_key, upload_id in sorted(pending):
            self.request('DELETE', upload_key, query='?'+self.transport.canonical_query({'uploadId': upload_id}))
        if self.pending_uploads():
            raise OperationError(503, 'r2_upload_abort_unconfirmed')
        self.request('DELETE', key, headers={'If-Match': etag})

    def verify_reservation(self, reservation):
        raw, _ = self.read(LOCK_KEY)
        if not reservation or raw != canonical(reservation):
            raise OperationError(409, 'r2_reservation_changed')

    def recover_owned_lock(self, reservation, owner, current_instance=None):
        raw, etag = self.read(LOCK_KEY)
        if raw is None:
            if not reservation:
                if self.pending_uploads():
                    raise OperationError(409, 'r2_unknown_upload_owner')
                return
            journal, _ = self.read(self.upload_journal_key(reservation))
            if journal is None:
                if self.pending_uploads():
                    raise OperationError(409, 'r2_unknown_upload_owner')
                return
        if not reservation or (raw is not None and raw != canonical(reservation)) or reservation.get('owner') != owner:
            raise OperationError(409, 'r2_lock_ownership_unresolved')
        import socket
        import psutil
        if owner.get('host') != socket.gethostname():
            raise OperationError(409, 'r2_lock_owner_unreachable')
        try:
            process = psutil.Process(owner['pid'])
            if process.create_time() == owner['start_time_seconds']:
                exact_instance = current_instance and all(owner.get(key) == value for key, value in current_instance.items())
                if not exact_instance:
                    raise OperationError(409, 'r2_lock_owner_still_running')
        except psutil.NoSuchProcess:
            pass
        # Unknown unfinished uploads require explicit transport-level recovery;
        # age alone never permits releasing a lock or abandoning their storage.
        self.finish_upload_journal(reservation, abort=True)
        self.inventory()
        if raw is not None:
            self.request('DELETE', LOCK_KEY, headers={'If-Match': etag})

    def retention(self):
        raw, _ = self.read(RETENTION_KEY)
        if raw is None:
            return {'schema': 1, 'admissions': {}, 'inflight': []}
        try:
            value = json.loads(raw)
            if value.get('schema') != 1 or not isinstance(value.get('admissions'), dict) or not isinstance(value.get('inflight'), list):
                raise ValueError('invalid retention projection')
            return value
        except (ValueError, TypeError, AttributeError) as error:
            raise OperationError(503, 'r2_retention_order_unavailable') from error

    def inventory(self):
        objects, manifests, heads = {}, {}, set()
        token = None
        while True:
            params = {"list-type": "2", "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            page = self.listing(self.transport.canonical_query(params))
            root = ET.fromstring(page._body)
            for item in root.findall("{*}Contents"):
                key = item.findtext("{*}Key")
                objects[key] = int(item.findtext("{*}Size"))
            token = root.findtext("{*}NextContinuationToken")
            if root.findtext("{*}IsTruncated") != "true":
                break
            if not token or len(objects) > 10000:
                raise OperationError(503, "r2_inventory_incomplete")
        if self.control_target:
            if any(key.startswith('_control/') for key in objects):
                # Public visibility is not control authorization. Preserve and
                # hold legacy records until an exact owned migration occurs.
                raise OperationError(503, 'r2_public_control_objects_require_migration')
            private_objects, _, _ = self.control_target.inventory()
            if any(not key.startswith('_control/') for key in private_objects):
                raise OperationError(503, 'r2_private_control_inventory_invalid')
            objects.update(private_objects)
        # An abandoned upload may still consume storage; never allocate through that uncertainty.
        if self.pending_uploads():
            raise OperationError(409, "r2_incomplete_upload_requires_recovery")
        for key in objects:
            if re.fullmatch(r"releases/channel-builds/[a-f0-9]{32}/build.json", key):
                raw, _ = self.read(key)
                manifest = json.loads(raw)
                build = BUILD.match(key)[1]
                if manifest.get("request", {}).get("buildId") != build or type(manifest["request"].get("sequence")) is not int:
                    raise OperationError(503, "r2_manifest_inventory_invalid")
                manifests[build] = manifest
            elif re.fullmatch(r"releases/channels/[a-z0-9-]+.json", key):
                raw, _ = self.read(key)
                record = json.loads(raw)
                for field in ('head','nativeHead','destinationHead','lastHead','nativeArchive','nativeRetirePending'):
                    if record.get(field):heads.add(record[field]['buildId'])
        self.verify_native_inventory(objects,manifests)
        if 'releases/manual/catalog.json' in objects:
            raw,_=self.read('releases/manual/catalog.json')
            _,manual_builds=validate_manual_catalog(raw,objects)
            heads.update(manual_builds)
        return objects, manifests, heads

    def verify_native_inventory(self, objects, manifests):
        from .package_graph import graph_descriptor,decode_package_graph
        for build,manifest in manifests.items():
            if manifest.get('schema')!=2:continue
            try:
                packages=manifest['packages']
                if len(packages)!=1:raise ValueError('ambiguous native package')
                package=packages[0];descriptor=graph_descriptor(package,build)
                body,_=self.read(descriptor['key'])
                graph=decode_package_graph(body,package,build)
                if any(objects.get(row['key'])!=row['size'] for row in [*graph['objects'],descriptor]):
                    raise ValueError('native object inventory missing or changed')
            except (OperationError,ValueError,TypeError,KeyError) as error:
                raise OperationError(503,'native_graph_inventory_unresolved') from error

    def initialize(self, record):
        key = "releases/channels/" + record["name"] + ".json"
        with self.locked():
            raw, _ = self.read(key)
            if raw is None:
                objects, manifests, heads = self.inventory()
                capacity_plan(objects, manifests, heads, len(canonical(record)), admission_order=self.retention()["admissions"])
                self.request("PUT", key, canonical(record), {"If-None-Match": "*"})
            elif any(json.loads(raw).get(field) != record.get(field) for field in ("name", "repository", "identity")):
                raise OperationError(503, "r2_channel_provenance_mismatch")
            # A DB-uncertain committed promotion is reconciled by promote, not overwritten here.

    def files(self, release, public_root):
        files = {}
        from .package_graph import package_objects
        for package in release['manifest']['packages']:
            for item in package_objects(package,release['manifest']['request']['buildId'],public_root):
                files[item['key']]=safe_file(public_root,item['key'])
        files[release['manifest_key']] = Path(public_root)/release['manifest_key']
        return files

    def preflight(self, release, public_root, *, reservation=None):
        # Read-only previews are advisory. Publication repeats this inventory
        # under its retained exclusive reservation before any mutation.
        with nullcontext(reservation) as held:
            if held:
                self.verify_reservation(held)
            objects, manifests, heads = self.inventory()
            projection = self.retention()
            build = release['manifest']['request']['buildId']
            admission = release.get('retention_admission') or {
                'release_id': release['release_id'], 'first_admitted_at': release['received_at']}
            previous = projection['admissions'].get(build)
            if previous and previous != admission:
                raise OperationError(409, 'r2_first_admission_conflict')
            projection['admissions'][build] = admission
            missing = []
            for key, path in self.files(release, public_root).items():
                if key in objects:
                    self.transport.verify_remote_artifact(self.url(key), self.creds, self.transport.amz_timestamp(),
                        path.stat().st_size, self.transport.file_sha256(str(path)), algorithm='sha256', encoding='hex')
                else:
                    missing.append(key)
            # Deterministic bounded control/lock/head allowance, independent of
            # ephemeral lock ids; every actual object remains in inventory bytes.
            lock_bytes = objects.get(LOCK_KEY, 0)
            control = sum(size for key, size in objects.items() if key.startswith('_control/')) - lock_bytes
            allocation = max(0, sum(self.files(release, public_root)[key].stat().st_size for key in missing) + CONTROL_ALLOCATION_BYTES - control)
            incoming = max(0, allocation - lock_bytes)
            delete, projected = capacity_plan(objects, manifests, heads | set(projection['inflight']), incoming, build,
                                              admission_order=projection['admissions'])
            retained = {BUILD.match(key)[1] for key in objects if BUILD.match(key) and key not in delete} | {build}
            return {'delete_keys': sorted(delete), 'projected_bytes': projected, 'current_bytes': sum(objects.values()) - lock_bytes,
                    'incoming_allocation_bytes': allocation, 'retained_build_count': len(retained),
                    'protected_builds': sorted(heads | set(projection['inflight'])), 'max_bytes_exclusive': MAX_BYTES,
                    'missing_keys': sorted(missing), 'retention': projection}

    def promote(self, release, public_root, previous, record, *, reservation=None, authorize=lambda: None):
        key = 'releases/channels/'+record['name']+'.json'
        with nullcontext(reservation) if reservation else self.locked() as held:
            self.verify_reservation(held)
            raw, etag = self.read(key)
            actual = json.loads(raw) if raw else None
            recovering = actual == record
            if not recovering and actual != previous:
                raise OperationError(409, 'r2_channel_revision_conflict')
            files = self.files(release, public_root)
            if not recovering:
                plan = self.preflight(release, public_root, reservation=held)
                authorize()
                projection = plan['retention']
                build = release['manifest']['request']['buildId']
                projection['inflight'] = sorted(set(projection['inflight']) | {build})
                if len(canonical(projection)) > 60000:
                    raise OperationError(413, 'r2_retention_projection_full')
                self.request('PUT', RETENTION_KEY, canonical(projection))
                for old in plan['delete_keys']:
                    self.request('DELETE', old)
            from .remote_publication import publish_objects_then_head
            def upload(file_key, file_path):
                if file_key in plan['missing_keys']:
                    self.transport.put_object(self.creds, self.base, self.bucket, file_key, str(file_path),
                        self.transport.amz_timestamp(), self.transport.content_type_for(file_key), {'If-None-Match': '*'},
                        on_multipart_created=lambda upload_id: self.journal_upload(held, file_key, upload_id))
            def verify(file_key, file_path):
                self.transport.verify_remote_artifact(self.url(file_key), self.creds, self.transport.amz_timestamp(),
                    file_path.stat().st_size, self.transport.file_sha256(str(file_path)), algorithm='sha256', encoding='hex')
            publish_objects_then_head(files, recovering=recovering,
                assert_reservation=lambda: self.verify_reservation(held), upload=upload, verify=verify,
                authorize=authorize, commit_head=lambda: self.request('PUT', key, canonical(record), {'If-Match': etag}),
                read_head=lambda: self.read(key)[0], intended_bytes=canonical(record), readback_error='r2_head_readback_failed')
            # A confirmed head remains Published even if subsequent maintenance fails.
            try:
                projection = self.retention()
                build = release['manifest']['request']['buildId']
                projection['inflight'] = [item for item in projection['inflight'] if item != build]
                self.request('PUT', RETENTION_KEY, canonical(projection))
                objects, manifests, heads = self.inventory()
                delete, _ = capacity_plan(objects, manifests, heads | set(projection['inflight']),
                                           admission_order=projection['admissions'])
                for old in delete:
                    self.request('DELETE', old)
                return {'maintenance_status': 'none'}
            except (OperationError, OSError, self.transport.R2RequestError):
                return {'maintenance_status': 'failed'}

    def maintain_retention(self, owner, *, reservation):
        """Backend backstop: protect current/in-flight builds; never change a head."""
        with self.locked(owner, reservation=reservation):
            objects, manifests, heads = self.inventory()
            projection = self.retention()
            delete, projected = capacity_plan(objects, manifests, heads | set(projection['inflight']),
                                               admission_order=projection['admissions'])
            for key in delete:
                self.verify_reservation(reservation)
                self.request('DELETE', key)
            return {'deleted_keys': delete, 'projected_bytes': projected}

    def cleanup(self, confirmed_build, authorize, *, reservation):
        """Retention-only recovery: never write a channel head."""
        with nullcontext(reservation):
            self.verify_reservation(reservation)
            objects, manifests, heads = self.inventory()
            projection = self.retention()
            projection['inflight'] = [build for build in projection['inflight'] if build != confirmed_build]
            delete, _ = capacity_plan(objects, manifests, heads | set(projection['inflight']),
                                       admission_order=projection['admissions'])
            authorize()
            self.request('PUT', RETENTION_KEY, canonical(projection))
            for key in delete:
                self.request('DELETE', key)
