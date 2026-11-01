"""Shared immutable-object verification followed by the sole conditional head."""
from .artifact_io import OperationError


def publish_objects_then_head(files, *, recovering, assert_reservation, upload,
                              verify, authorize, commit_head, read_head,
                              intended_bytes, readback_error):
    """Provider hooks retain allocation, transport and private recovery ownership."""
    for key, path in files.items():
        if not recovering:
            assert_reservation()
            upload(key, path)
        verify(key, path)
    if not recovering:
        authorize()
        assert_reservation()
        commit_head()
    if read_head() != intended_bytes:
        raise OperationError(503, readback_error)
