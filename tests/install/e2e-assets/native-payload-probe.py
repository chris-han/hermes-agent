"""CI Plugin Doctor fixture: exercise the installed native runtime without network."""
import ctypes
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys


def register(_context):
    if sys.platform != 'win32' or sys.maxsize <= 2**32:
        raise ValueError('Native payload probe requires Windows x64')
    from ctypes import wintypes
    api=ctypes.WinDLL('kernel32',use_last_error=True).GetCurrentPackageFullName
    api.argtypes=[ctypes.POINTER(wintypes.UINT),wintypes.LPWSTR];api.restype=wintypes.LONG
    length=wintypes.UINT()
    if api(ctypes.byref(length),None)!=122:
        raise ValueError('Native payload probe requires registered package identity')
    buffer=ctypes.create_unicode_buffer(length.value)
    if api(ctypes.byref(length),buffer)!=0:raise ValueError('Actual package identity unavailable')
    spec=json.loads((Path(__file__).parent/'probe.json').read_text(encoding='utf-8-sig'))
    if buffer.value!=spec['package_full_name']:raise ValueError('Native probe package identity mismatch')
    package=Path(spec['package_root']).resolve()
    def installed(path):
        resolved=Path(path).resolve()
        if not resolved.is_relative_to(package):raise ValueError('Native runtime escaped installed package')
        return str(resolved)
    executable=installed(sys.executable)
    from pm.paths import repo_root
    repo=Path(installed(repo_root()))
    stamp=json.loads((package/'app/resources/install-stamp.json').read_text(encoding='utf-8-sig'))
    if stamp['commit']!=spec['commit'] or stamp['channelBuild']['buildId']!=spec['build_id']:
        raise ValueError('Native runtime source/build mismatch')
    import numpy
    import polars
    import sherpa_onnx
    import soundfile
    modules={module.__name__:installed(module.__file__) for module in (numpy,polars,sherpa_onnx,soundfile)}
    total=polars.DataFrame({'value':[1,2,3]}).select(polars.col('value').sum()).item()
    if total!=6:raise ValueError('Installed Polars computation failed')
    provider=repo/'semantier-product/semantier_product/sensevoice_stt/provider.py'
    installed(provider)
    loader=importlib.util.spec_from_file_location('native_sensevoice_probe',provider)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    voice=module.SenseVoiceProvider()
    if not voice.is_available():raise ValueError('Installed SenseVoice assets/runtime unavailable')
    assets=installed(voice._asset_dir())
    recognizer=voice._recognizer()
    stream=recognizer.create_stream()
    stream.accept_waveform(16000,numpy.zeros(16000,dtype=numpy.float32))
    recognizer.decode_stream(stream)
    skills=sorted((repo/'skills').rglob('SKILL.md'))
    if not skills:raise ValueError('Installed Skills missing')
    skill_bytes=0
    for skill in skills:
        installed(skill)
        body=skill.read_bytes()
        if not body:raise ValueError('Installed Skill empty')
        body.decode('utf-8-sig');skill_bytes+=len(body)
    native={name:installed(item.__file__) for name,item in sys.modules.copy().items()
            if getattr(item,'__file__',None) and str(item.__file__).lower().endswith('.pyd')}
    if not native:raise ValueError('No installed native modules were loaded')
    proof=dict(producedAt=datetime.now(UTC).isoformat(),packageFullName=buffer.value,
        sourceCommit=stamp['commit'],buildId=stamp['channelBuild']['buildId'],executable=executable,
        python=sys.version,modules=modules,nativeModules=native,polarsSum=total,
        sensevoice=dict(assetDir=assets,modelLoaded=True,silenceDecodeCompleted=True,accuracy='NOT_MEASURED'),
        skills=dict(filesRead=len(skills),bytesRead=skill_bytes),
        scope='actual package identity, installed native imports/computation/model decode and Skill reads; no update or preservation claim')
    out=Path(spec['output'])
    with out.open('x',encoding='utf-8') as target:json.dump(proof,target,indent=2);target.write('\n')


if __name__=='__main__':register(None)
