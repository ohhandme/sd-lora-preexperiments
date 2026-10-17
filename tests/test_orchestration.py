import os,subprocess,sys
from spec_research import common

def test_inherited_lock_excludes_other_process(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path);(tmp_path/'manifests').mkdir()
    with common.gpu_lock() as fd:
        code="from pathlib import Path; from spec_research import common; common.ROOT=Path(%r); " % str(tmp_path)
        inherited=code+"\nwith common.gpu_lock(%d): print('inherited exclusive lock verified')" % fd
        result=subprocess.run([sys.executable,'-c',inherited],pass_fds=(fd,),capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        competitor=code+"\ntry:\n with common.gpu_lock(): raise AssertionError('competing lock acquired')\nexcept BlockingIOError: print('competitor correctly excluded')"
        result=subprocess.run([sys.executable,'-c',competitor],capture_output=True,text=True)
        assert result.returncode==0 and 'correctly excluded' in result.stdout,result.stderr
