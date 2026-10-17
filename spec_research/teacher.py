import pathlib
from .common import *
from .decoding import ar

def teacher_config():
    from .numerics import AR_POLICY
    # This generator uses full B1 prefill followed by one-token B1 queries.
    # Version2 changes only multi-query/request norms, leaving this AR path
    # unchanged. Preserve its exact configuration and existing teacher records.
    return {'model':model_info('8B'),'system':protocol()['system'],'enable_thinking':False,'do_sample':False,'processors':[],'max_new_tokens':512,'precision':'bfloat16','backend':'sdpa','tokenizer_revision':model_info('8B')['revision'],'numerical_policy':dict(AR_POLICY)}
def teacher_path(r):return ROOT/'data/teacher'/r['domain']/(objhash(r['id'])+'.json')
def validate_teacher(records,require_all=True):
    cfg=teacher_config();found=[]
    for r in records:
        p=teacher_path(r)
        if not p.exists():
            if require_all:raise RuntimeError('Missing teacher '+r['id'])
            continue
        t=read(p)
        assert t['id']==r['id'] and t['prompt_ids']==r['prompt_ids'] and t['config_sha']==objhash(cfg)
        assert t['response_ids'] and len(t['response_ids'])<=512
        assert t['record_sha']==objhash({k:v for k,v in t.items() if k!='record_sha'})
        assert t['eos']==(t['response_ids'][-1] in t['eos_ids'])
        assert t['truncated']==(len(t['response_ids'])==512 and not t['eos'])
        found.append(t)
    return found

def generate(records,log,phase):
    assert_spec();cfg=teacher_config();outdir=ROOT/'data/teacher';outdir.mkdir(parents=True,exist_ok=True)
    meta=outdir/'config.json'
    if meta.exists():assert read(meta)==cfg,'Teacher config conflict'
    else:atomic(meta,cfg)
    existing=validate_teacher(records,False);done={r['id'] for r in existing}
    pending=[r for r in records if r['id'] not in done]
    if not pending:return existing
    model=load_model('8B');tok=tokenizer();eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    assert eos
    for n,r in enumerate(records):
        if r['id'] in done:continue
        try:
            result=ar(model,r['prompt_ids'],512,set(eos));response=result['tokens']
            assert response,'Empty teacher output'
            item={**r,'response_ids':response,'eos':response[-1] in eos,'truncated':len(response)==512 and response[-1] not in eos,'eos_ids':eos,'config_sha':objhash(cfg),'generation':cfg,'code_commit':revision()}
            item['record_sha']=objhash(item);atomic(teacher_path(r),item)
            append(log,{'at':now(),'id':r['id'],'order':n,'tokens':len(response),'seconds':result['latency_ms']/1000,'status':'complete'})
            status(phase=phase,state='running',counts={'teacher_completed_for_current_request':len(done)+1,'teacher_requested':len(records)},current_job=r['id'],training_started=False)
            done.add(r['id'])
        except Exception as e:
            append(log,{'id':r['id'],'status':'failed','error':repr(e)});raise
    del model
    import gc,torch
    gc.collect();torch.cuda.empty_cache()
    return validate_teacher(records)
