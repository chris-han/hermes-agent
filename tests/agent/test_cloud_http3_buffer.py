from agent.cloud_http3_transport import _BodyBuffer


def test_full_buffer_pauses_without_consuming_and_resumes_after_read():
    resumed=[]
    body=_BodyBuffer(4,lambda:resumed.append(True))
    assert body.write(b'abcd') == 4
    assert body.write(b'ef') == body.PAUSE
    assert body.buffered_bytes == 4
    assert body.read() == b'abcd'
    assert resumed == [True]
    # libcurl replays the unconsumed chunk after resume.
    assert body.write(b'ef') == 2
    body.finish()
    assert body.read() == b'ef'
    assert body.read() is None


def test_close_discards_buffer_and_rejects_more_bytes():
    body=_BodyBuffer(4,lambda:None)
    body.write(b'ab')
    body.close()
    assert body.buffered_bytes == 0
    assert body.read() is None
    assert body.write(b'cd') == body.ERROR


def test_close_wakes_waiting_reader():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    body=_BodyBuffer(4,lambda:None)
    entered=Event()
    def read():
        entered.set()
        return body.read()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result=pool.submit(read)
        assert entered.wait(1)
        body.close()
        assert result.result(timeout=1) is None


def test_oversized_callback_fails_without_exceeding_capacity():
    import httpx
    import pytest
    body=_BodyBuffer(4,lambda:None)
    assert body.write(b'12345') == body.ERROR
    assert body.buffered_bytes == 0
    with pytest.raises(httpx.ReadError):
        body.read()


def test_idle_buffer_read_timeout():
    import time
    import httpx
    import pytest
    buffer = _BodyBuffer(32768, lambda: None, read_timeout=0.02)
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        buffer.read()
    assert time.monotonic() - started < 0.5
