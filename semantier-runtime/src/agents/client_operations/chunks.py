"""Release-time chunk preparation in private custody; publication remains head-last."""
import hashlib
import re
import tempfile
from pathlib import Path

from .artifact_io import OperationError, immutable_copy, safe_file

CHUNK_BYTES = 128 * 1024 * 1024
_BLOCK_BYTES = 1024 * 1024


def prepare_chunks(source, root, build_id, artifact):
    """Verify the signed source before exposing immutable chunk objects in custody.

    The caller must own source/root and include these bytes in target admission.
    This does not upload, promote, or qualify a native release.
    """
    if (not isinstance(build_id, str) or not re.fullmatch(r'[a-f0-9]{32}', build_id)
            or type(artifact.get('size')) is not int or not 0 < artifact['size'] < 10_000_000_000
            or not isinstance(artifact.get('sha256'), str)
            or not re.fullmatch(r'[a-f0-9]{64}', artifact['sha256'])):
        raise OperationError(422, 'invalid_chunk_artifact')
    source, root = Path(source), Path(root).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise OperationError(422, 'artifact_symlink_refused')
    if source.is_symlink() or not source.is_file():
        raise OperationError(422, 'artifact_symlink_refused')
    root.mkdir(parents=True, exist_ok=True)
    chunks = []
    complete = hashlib.sha256()
    total = 0
    # Private temporary staging prevents a failed full digest from producing
    # apparently valid release chunks. Each copy uses the existing fsync and
    # atomic no-replace owner; conflicting custody bytes are never overwritten.
    with tempfile.TemporaryDirectory(prefix='.chunking-', dir=root) as temporary:
        with source.open('rb') as incoming:
            while total < artifact['size']:
                offset = total
                length = min(CHUNK_BYTES, artifact['size'] - total)
                checksum = hashlib.sha256()
                staged = Path(temporary) / str(len(chunks))
                with staged.open('xb') as output:
                    remaining = length
                    while remaining:
                        body = incoming.read(min(_BLOCK_BYTES, remaining))
                        if not body:
                            raise OperationError(422, 'artifact_integrity_mismatch')
                        output.write(body)
                        checksum.update(body)
                        complete.update(body)
                        total += len(body)
                        remaining -= len(body)
                chunks.append({'key': f"releases/channel-builds/{build_id}/chunks/{artifact['sha256']}/{len(chunks):06d}.part",
                               'offset': offset, 'size': length, 'sha256': checksum.hexdigest()})
            if incoming.read(1) or complete.hexdigest() != artifact['sha256']:
                raise OperationError(422, 'artifact_integrity_mismatch')
        # Resolve all paths before copying so traversal/symlink failures do not
        # publish a partial descriptor. Chunks are inert until the head commits.
        destinations = [safe_file(root, chunk['key']) for chunk in chunks]
        for index, (chunk, destination) in enumerate(zip(chunks, destinations)):
            immutable_copy(Path(temporary) / str(index), destination, chunk)
    return {'schema': 1, 'buildId': build_id,
            'artifact': {'size': total, 'sha256': artifact['sha256']},
            'chunkSize': CHUNK_BYTES, 'chunks': chunks}


def chunk_entries(artifact, build_id):
    """Validate the versioned inline descriptor before custody or remote IO."""
    descriptor = artifact.get('chunks')
    if descriptor is None:
        return []
    facts = {'size': artifact['size'], 'sha256': artifact['sha256']}
    if (not isinstance(descriptor, dict) or type(descriptor.get('schema')) is not int
            or descriptor['schema'] != 1 or descriptor.get('buildId') != build_id
            or descriptor.get('artifact') != facts or descriptor.get('chunkSize') != CHUNK_BYTES
            or type(facts['size']) is not int or not 0 < facts['size'] < 10_000_000_000
            or not isinstance(descriptor.get('chunks'), list)
            or len(descriptor['chunks']) != (facts['size'] + CHUNK_BYTES - 1)//CHUNK_BYTES):
        raise OperationError(422, 'invalid_chunk_descriptor')
    offset = 0
    for index, chunk in enumerate(descriptor['chunks']):
        expected = f"releases/channel-builds/{build_id}/chunks/{facts['sha256']}/{index:06d}.part"
        if (not isinstance(chunk, dict) or chunk.get('key') != expected
                or type(chunk.get('offset')) is not int or chunk['offset'] != offset
                or type(chunk.get('size')) is not int or chunk['size'] != min(CHUNK_BYTES, facts['size']-offset)
                or not isinstance(chunk.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', chunk['sha256'])):
            raise OperationError(422, 'invalid_chunk_descriptor')
        offset += chunk['size']
    return descriptor['chunks']
