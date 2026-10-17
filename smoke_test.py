import argparse,pathlib,subprocess,traceback,xml.etree.ElementTree as ET
from spec_research.common import *
from spec_research.data import prepare,smoke_records
from spec_research.environment import check_environment
from spec_research.evaluation import evaluate_requests
from spec_research.teacher import generate,validate_teacher
from spec_research.training import train
from spec_research.gate import create_gate

def smoke(run_id):
    assert_spec();path=ROOT/'results/smoke/report.json';report={'state':'running','run_id':run_id,'prefix':sys.prefix,'provenance':provenance(),'stages':{},'scales':{},'commands':[],'ids':[]}
    if path.exists():
        import shutil
        previous=read(path);archive=ROOT/'results/smoke'/previous['run_id']/'prior_attempt_evidence';archive.mkdir(parents=True,exist_ok=True)
        for old in [path,ROOT/'results/smoke/environment.json',ROOT/'results/smoke/scripted.json',ROOT/'results/smoke/scripted.xml',ROOT/'logs/smoke/scripted.log']:
            if old.exists():
                dest=archive/old.name
                assert not dest.exists(),f'Prior evidence already archived: {dest}'
                shutil.copy2(old,dest)
    atomic(path,report)
    def completed(name,result):report['stages'][name]={'state':'passed','result':result,'at':now()};atomic(path,report)
    def run(args,log):
        report['commands'].append(args);atomic(path,report)
        with open(log,'w') as f:r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,f'{args}: exit {r.returncode}; see {log}'
        return r.returncode
    try:
        status(phase='p0_environment',state='running',training_started=False,current_job=None,latest_error=None)
        env=check_environment();completed('environment',env['state'])
        args=[SPEC+'/bin/python','-m','pytest','-q','tests','--junitxml=results/smoke/scripted.xml']
        rc=run(args,ROOT/'logs/smoke/scripted.log');xml=ET.parse(ROOT/'results/smoke/scripted.xml');num=sum(int(x.attrib.get('tests','0')) for x in xml.getroot().iter('testsuite'))
        atomic(ROOT/'results/smoke/scripted.json',{'exit_code':rc,'tests':num,'prefix':sys.prefix,'provenance':provenance()});assert num>=54;completed('scripted',num)
        status(phase='p0_data_preparation',state='running');data=prepare();rr=smoke_records();report['ids']=[r['id'] for r in rr];assert len(rr)==192;completed('data',data['counts'])
        # Short preliminary real-model guard saves GPU hours if backend or cache semantics fail.
        preliminary=[rr[0],rr[64],rr[128]]
        for scale in ('0.6B','4B'):
            evaluate_requests(scale,preliminary,None,ROOT/'results/smoke'/run_id/scale/'preliminary',64,(4,8,12))
        completed('preliminary_real_sd',True)
        status(phase='smoke_teacher_running',state='running');tt=generate(rr,ROOT/'logs/smoke'/f'{run_id}.teacher.jsonl','smoke_teacher_running');assert len(tt)==192;completed('teacher',192)
        for scale in ('0.6B','4B'):
            status(phase='smoke_training_running',state='running',current_job=scale)
            out=ROOT/'adapters/smoke'/scale/run_id
            tr=train(scale,'mix16',tt,out,ROOT/'logs/smoke'/f'{run_id}.{scale}.train.jsonl',smoke=True)
            adapter=out/'epoch1';primary=ROOT/'results/smoke'/run_id/scale/'primary';extra=ROOT/'results/smoke'/run_id/scale/'extra'
            sm={'training_complete':str(out/'complete.json'),'adapter':str(adapter),'primary':str(primary),'extra':str(extra)}
            report['scales'][scale]=sm;atomic(path,report)
            status(phase='smoke_decode_running',state='running',current_job=scale)
            evaluate_requests(scale,rr,adapter,primary,64,(8,))
            evaluate_requests(scale,preliminary,adapter,extra,64,(4,12))
            completed('scale_'+scale,True)
        assert report['provenance']['source_hashes']==sources(),'Code changed during P0'
        guard_command=[SPEC+'/bin/python','validate_gate.py'];report['commands'].append(guard_command)
        report['state']='passed';report['finished_at']=now();atomic(path,report)
        gate=create_gate(path)
        with (ROOT/'logs/smoke/independent_guard.log').open('w') as log:
            independent=subprocess.run(guard_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert independent.returncode==0,'Independent guard failed; formal launch blocked'
        status(phase='p0_passed',state='complete',training_started=False,current_job=None);return gate
    except BaseException as e:
        report.update(state='failed',error=repr(e),traceback=traceback.format_exc(),finished_at=now());atomic(path,report)
        status(phase='p0_failed',state='blocked',training_started=False,latest_error=repr(e),next_action='Inspect saved evidence; repair within the 6-attempt limit and reverify relevant checks. Formal pipeline not launched.')
        append(ROOT/'autoresearch/loop-261016-2120/attempts.jsonl',{'at':now(),'status':'failed','error':repr(e),'traceback':report['traceback']});raise
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    with gpu_lock():smoke(a.run_id)
