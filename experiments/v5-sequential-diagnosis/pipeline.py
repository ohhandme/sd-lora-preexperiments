"""Execute all four preregistered stages, including negative scientific results."""
import argparse
import fcntl
import math
import signal
from common import *
from training_jobs import stage1_train,stage4_train
from reports import stage1_curves,stage2,stage3,stage4
from spec_research.decoding import ar,speculative,compare


def ledger(phase,state,detail):
    LEDGER.mkdir(parents=True,exist_ok=True)
    path=LEDGER/'handoff.json'
    doc=read(path) if path.exists() else {'version':'2.2.2','source':'classic','created':now(),'phases':[],
            'config':{'scope':str(EXP),'iterations':4,'metric':'mismatch char_weighted routed acceptance minus old MIX80',
                      'verify':f'{sys.executable} {EXP}/pipeline.py verify','guard':f'{sys.executable} {EXP}/pipeline.py guard'}}
    doc.update(status=state,updated=now(),current_phase=phase)
    doc['phases'].append({'phase':phase,'at':now(),'state':state,'detail':detail})
    atomic(path,doc)
    metric=None
    if (OUT/'stage3.json').exists():metric=read(OUT/'stage3.json')['comparisons']['mismatch_char_weighted_minus_old_mix80']['mean']
    tsv=LEDGER/'loop-results.tsv'
    if not tsv.exists():tsv.write_text('# metric_direction: higher_is_better\nphase\ttimestamp\tmetric\tstatus\tdescription\n')
    with tsv.open('a') as f:f.write(f'{phase}\t{now()}\t{metric if metric is not None else "NA"}\t{state}\t{json.dumps(detail)}\n')


def correctness(splits,binding):
    path=OUT/'correctness.json'
    if path.exists():return read(path)
    gpu_idle();target=load_model('8B').eval()
    raw=target.generation_config.eos_token_id;eos={raw} if isinstance(raw,int) else set(raw)
    structure=read(OUT/'stage4.json');model=None;models={}
    for pool in ('mismatch','rank32','wide'):
        chosen=structure['pools'][pool]['val']['fixed_index']
        adapter=EXP/'adapters'/f'{pool}_{chosen}'/'s32'
        if model is None:model=PeftModel.from_pretrained(load_model('0.6B'),adapter,adapter_name=pool).eval()
        else:model.load_adapter(adapter,adapter_name=pool,is_trainable=False)
        models[pool]=str(adapter)
    rows=[next(r for r in splits['eval'] if r['domain']==d) for d in DOMAINS]
    details=[]
    for row in rows:
        reference=ar(target,row['prompt_ids'],64,eos)
        assert reference['tokens']==row['response_ids'][:len(reference['tokens'])]
        item={'id':row['id'],'AR':reference,'methods':{}}
        for pool in models:
            model.set_adapter(pool);model.eval()
            result=speculative(target,model,row['prompt_ids'],64,eos,8)
            failure=compare(reference,result)
            if failure:
                atomic(OUT/'correctness_failure.json',{'id':row['id'],'pool':pool,'failure':failure})
                raise AssertionError('Complete target token mismatch')
            assert (result['tokens'][-1] in eos)==(reference['tokens'][-1] in eos)
            item['methods'][pool]=result
        details.append(item)
        atomic(OUT/'online'/key(row),item)
        status('correctness',requests=len(details),total=5)
    summary={'all_exact':True,'requests':5,'comparisons':15,'binding':binding,'models':models,
             'scope':'single timing repeat, fixed expert per structure; correctness smoke, not speed evidence',
             'diagnostic_speed_vs_AR':{pool:math.exp(mean(math.log(x['AR']['latency_ms']/x['methods'][pool]['latency_ms']) for x in details)) for pool in models}}
    atomic(path,summary)
    del model,target;cleanup()
    return summary


def guard():
    protected_count=protect_check();splits,binding=data()
    train=read(OUT/'stage1_training.json');cap=read(OUT/'stage4_training.json')
    for method,item in {**train,**cap}.items():
        assert item['binding']==binding and item['base_frozen']
        for step,cp in item['checkpoints'].items():
            assert digest(EXP/'adapters'/method/f's{step}'/'adapter_model.safetensors')==cp['weight_sha']
        expected=640 if method in ('pooled','weighted') else 128
        assert item['requests']==expected and item['steps']==expected//4
    for p in POOLS:
        for field in ('input_tokens','response_tokens','steps','requests'):
            assert sum(train[f'{p}_{i}'][field] for i in range(5))==train['pooled'][field]
    for p in ('rank32','wide'):
        for i in range(5):
            for field in ('input_tokens','response_tokens','steps','requests'):
                assert cap[f'{p}_{i}'][field]==train[f'mismatch_{i}'][field]
    online=read(OUT/'correctness.json');assert online['all_exact'] and online['comparisons']==15
    for path in (OUT/'online').glob('*.json'):
        r=read(path)
        assert all(v['tokens']==r['AR']['tokens'] for v in r['methods'].values())
    return {'state':'passed','protected_files':protected_count,'trained_adapters':len(train)+len(cap),
            'exact_comparisons':15,'matched_token_exposure':True,'binding':binding,'at':now()}


def finish():
    audit=guard();atomic(OUT/'guard.json',audit)
    curves=read(OUT/'stage1_curves.json');s2=read(OUT/'stage2.json');s3=read(OUT/'stage3.json');s4=read(OUT/'stage4.json')
    report={'state':'all_four_stages_complete','at':now(),'guard':audit,
            'stage2_complementarity_gate':s2['complementarity_gate'],'stage3_routing_gate':s3['routing_gate'],
            'stage4_capacity_gate':s4['capacity_gate'],
            'metric':s3['comparisons']['mismatch_char_weighted_minus_old_mix80']['mean'],
            'routing_vs_semantic':s3['comparisons']['mismatch_char_weighted_minus_semantic'],
            'limits':['seed42 only','reused development partitions, final ID/external not opened',
                      'one epoch, 128 samples per expert','public-prefix counterfactual except correctness smoke',
                      'capacity variants have no fitted routers','no WAN or formal speed evidence'],
            'source_hashes':{p.name:digest(p) for p in EXP.glob('*.py')}}
    atomic(OUT/'final_report.json',report)
    lines=['# V5 四阶段预实验结果','',f'状态：{report["state"]}；旧文件保护：{audit["protected_files"]} 个；在线一致性：15/15。','',
           '## 1. 训练学习曲线（50条开发验证请求）','', '| 池 | 步数/专家 | Oracle | 有专家差异窗口 | 总窗口 |','|---|---:|---:|---:|---:|']
    for pool in POOLS:
        for step,v in curves[pool].items():lines.append(f'| {pool} | {step} | {v["prefix_oracle"]:.4f} | {v["different_acceptance_windows"]} | {v["windows"]} |')
    lines+=['','## 2. 冻结专家池互补（120条开发评估请求）','', '| 池 | 固定专家 | 请求Oracle | 前缀Oracle |','|---|---:|---:|---:|']
    for p,v in s2['pools'].items():lines.append(f'| {p} | {v["fixed"]:.4f} | {v["request_oracle"]:.4f} | {v["prefix_oracle"]:.4f} |')
    lines+=['','## 3. 路由监督与实际选择','', '| 池 | 有区分度窗口/训练窗口 | Hash | Char | Char加权 |','|---|---:|---:|---:|---:|']
    for p,v in s3['pools'].items():lines.append(f'| {p} | {v["informative_windows"]}/{v["training_windows"]} | {v["routers"]["hash"]["mean"]:.4f} | {v["routers"]["char"]["mean"]:.4f} | {v["routers"]["char_weighted"]["mean"]:.4f} |')
    lines+=['',f'主候选相对MIX80：{s3["comparisons"]["mismatch_char_weighted_minus_old_mix80"]}。',
            f'相对相同路由器的语义池：{report["routing_vs_semantic"]}。','',
            '## 4. 结构消融（均最终32步，不做容量变体路由）','', '| 结构 | Oracle | 固定专家 | 活跃适配参数 | 总适配参数 |','|---|---:|---:|---:|---:|']
    for p,v in s4['pools'].items():lines.append(f'| {p} | {v["eval"]["prefix_oracle"]:.4f} | {v["eval"]["fixed"]:.4f} | {v["active_adapter_parameters"]} | {v["pool_adapter_parameters"]} |')
    lines+=['',f'结构比较CI：{s4["comparisons"]}。','',
            '## 范围与限制','', '所有四步均执行。来源与预算检查通过不代表研究假设成立。单seed、开发集复用，Oracle不等于可部署速度；未解锁最终测试、WAN或多seed。',
            '离线生成在首失配后提前结束，记录的是接受行为及首失配前缀，不能据此报告完整拒绝后缀的多样性。',
            '所有负结果保留；每步原始请求级数据和置信区间见 results/stage*.json。']
    (EXP/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ledger(4,'COMPLETE',report);status('all_four_complete',metric=report['metric'],routing_gate=report['stage3_routing_gate'])


def selftest():
    a=np.array([[7,2],[2,7]],float)
    assert a.max(axis=1).mean()==7 and a.mean(axis=0).max()==4.5
    assert paired([1,2],[1,2])['ci95']==[0.,0.]
    row={'response_ids':list(range(100))}
    validate({'windows':[{'position':0,'a':2,'proposal':[0,1,9]},
                         {'position':64,'a':8,'proposal':list(range(64,72))}],'mean_a':5.},row)
    try:validate({'windows':[{'position':0,'a':2,'proposal':[0,1]}],'mean_a':2.},row)
    except AssertionError:pass
    else:raise AssertionError('Truncated prefix without rejection accepted')
    assert not (0.0>1e-9)
    print('selftest passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','guard','verify','selftest'))
    args=parser.parse_args();boot()
    if args.mode=='selftest':selftest();return
    if args.mode=='verify':
        report=read(OUT/'final_report.json');assert report['state']=='all_four_stages_complete'
        assert report['guard']['state']=='passed' and math.isfinite(report['metric'])
        print(report['metric']);return
    if args.mode=='guard':print(json.dumps(guard()));return
    with (ROOT/'manifests/gpu_pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle()
        def deadline(*_):raise TimeoutError('Six-hour continuous-run limit; checkpoints preserved')
        signal.signal(signal.SIGALRM,deadline);signal.alarm(21600)
        splits,binding=data();assert read(OUT/'prepared.json')['binding']==binding
        atomic(OUT/'execution_sources.json',{p.name:digest(p) for p in EXP.glob('*.py')})
        ledger(1,'RUNNING',{'train_requests':640,'per_expert':128,'route_fit':200,'val':50,'eval':120})
        stage1_train(splits,binding)
        if not (OUT/'stage1_curves.json').exists():stage1_curves(splits,binding)
        ledger(1,'COMPLETE',read(OUT/'selection.json'))
        if not (OUT/'stage2.json').exists():stage2(splits,binding)
        ledger(2,'COMPLETE',{'scientific_gate':read(OUT/'stage2.json')['complementarity_gate']})
        if not (OUT/'stage3.json').exists():stage3(splits,binding)
        ledger(3,'COMPLETE',{'scientific_gate':read(OUT/'stage3.json')['routing_gate']})
        stage4_train(splits,binding)
        if not (OUT/'stage4.json').exists():stage4(splits,binding)
        correctness(splits,binding);finish();signal.alarm(0)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=repr(error))
        ledger('error','ERROR',{'error':repr(error)})
        raise
