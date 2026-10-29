"""Native-only installed qualification refuses an actual non-Windows host."""
import importlib.util
from pathlib import Path
import pytest


@pytest.mark.platforms('posix')
def test_native_payload_probe_refuses_the_actual_host_before_fixture_io():
    file=Path(__file__).resolve().parents[1]/'install/e2e-assets/native-payload-probe.py'
    spec=importlib.util.spec_from_file_location('native_payload_probe',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='requires Windows x64'):module.register(None)
