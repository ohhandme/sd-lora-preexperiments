import collections, math, random, time
import torch
from peft import LoraConfig,TaskType,get_peft_model
from .common import *
from .teacher import validate_teacher

def attach(model,rank=16,name='default'):
    cfg=LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,bias='none',target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM)
    return get_peft_model(model,cfg,adapter_name=name)
def geometry(model,rank,name='default'):
    found=[]
    for key,mod in model.named_modules():
        if hasattr(mod,'lora_A') and name in mod.lora_A:
            a,b=mod.lora_A[name].weight,mod.lora_B[name].weight
            assert a.shape[0]==rank and b.shape[1]==rank
            assert key.endswith(('q_proj','v_proj'));found.append(key)
    assert len(found)==2*model.config.num_hidden_layers,(len(found),model.config.num_hidden_layers)
    return {'modules':len(found),'rank':rank,'adapter_params':sum(p.numel() for n,p in model.named_parameters() if f'.{name}.' in n and 'lora_' in n),'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_dtypes':dict(collections.Counter(str(p.dtype) for p in model.parameters() if p.requires_grad))}
def batch_record(r,device):
    prompt=r['prompt_ids'];response=r['response_ids'];assert prompt and response
    ids=prompt+response;assert len(ids)<=2048
    labels=[-100]*len(prompt)+response
    x=torch.tensor([ids],dtype=torch.long,device=device);y=torch.tensor([labels],dtype=torch.long,device=device)
    assert int((y[:,1:]!=-100).sum())==len(response)
    return x,y

def order(records,seed,epoch,mixed):
    rng=random.Random(seed+epoch)
    if not mixed:
        rr=list(records);rng.shuffle(rr);return rr
    groups={d:[r for r in records if r['domain']==d] for d in DOMAINS}
    assert len(set(len(g) for g in groups.values()))==1
    for g in groups.values():rng.shuffle(g)
    rr=[]
    for i in range(len(groups['math'])):
        domains=list(DOMAINS);rng.shuffle(domains);rr.extend(groups[d][i] for d in domains)
    return rr

def train(scale,method,records,out,log,smoke=False):
    assert_spec();out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    rank=48 if method=='mix48' else 16;cfg={**protocol()['train'],'rank':rank,'alpha':2*rank,'scale':scale,'method':method,'seed':42,'smoke':smoke,'compute_autocast':'bfloat16','epochs':1 if smoke else 2,'source_ids':[r['id'] for r in records],'teacher_shas':[r['record_sha'] for r in records],'provenance':provenance()}
    cfgfile=out/'run.json'
    if cfgfile.exists():
        prior=read(cfgfile)
        # Provenance must match on resume, rather than silently training changed code.
        assert prior['provenance']['source_hashes']==cfg['provenance']['source_hashes'],'Code changed on resume'
        assert {k:v for k,v in prior.items() if k!='provenance'}=={k:v for k,v in cfg.items() if k!='provenance'}
        cfg=prior
    else:atomic(cfgfile,cfg)
    if (out/'complete.json').exists():return read(out/'complete.json')
    seed_all(42);model=attach(load_model(scale),rank);geo=geometry(model,rank)
    model.train();model.config.use_cache=False;model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad];frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    total_steps=math.ceil(len(records)/16)*cfg['epochs'];warmup=math.ceil(total_steps*.05)
    def lr_lambda(step):
        if step<warmup:return step/max(1,warmup)
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    step=0;start_epoch=0;start_offset=0;tokens=0;response_tokens=0;exposures=collections.Counter();elapsed=0.;nonzero=0
    resume=out/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['config_sha']==digest(cfgfile)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model,state['adapter']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        step=state['step'];start_epoch=state['epoch'];start_offset=state['offset'];tokens=state['tokens'];response_tokens=state['response_tokens'];exposures.update(state['exposures']);elapsed=state['elapsed'];nonzero=state['nonzero']
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    from peft import get_peft_model_state_dict
    t0=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch,cfg['epochs']):
        rr=order(records,42,epoch,method.startswith('mix'));offset=start_offset if epoch==start_epoch else 0
        for base in range(offset,len(rr),16):
            chunk=rr[base:base+16];loss_sum=0.
            for r in chunk:
                x,y=batch_record(r,model.device)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False).loss
                assert torch.isfinite(loss).item(),f'Nonfinite loss {r["id"]}'
                (loss/len(chunk)).backward();loss_sum+=float(loss.detach());tokens+=x.numel();response_tokens+=len(r['response_ids']);exposures[r['id']]+=1
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            assert torch.isfinite(norm).item()
            if float(norm)>0:nonzero+=1
            assert any(p.grad is not None for p in params)
            used_lr=optimizer.param_groups[0]['lr']
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            assert all(p._version==frozen[n] and p.grad is None for n,p in model.named_parameters() if n in frozen),'Frozen base mutated'
            assert all(torch.isfinite(p).all().item() for p in params),'Nonfinite adapter weights'
            entry={'at':now(),'scale':scale,'method':method,'smoke':smoke,'step':step,'epoch':epoch+1,'offset':base+len(chunk),'loss':loss_sum/len(chunk),'gradient_norm':float(norm),'lr':scheduler.get_last_lr()[0],'training_tokens':tokens,'response_tokens':response_tokens,'seconds':elapsed+time.perf_counter()-t0,'peak_allocated':torch.cuda.max_memory_allocated(),'actual_optimizer_update':bool(used_lr>0 and float(norm)>0),'update_lr':used_lr}
            append(log,entry);print(json.dumps(entry),flush=True)
            if not smoke and entry['actual_optimizer_update']:status(phase='training_running',state='running',training_started=True,current_job=f'{scale}/{method}',counts={'optimizer_steps_current_job':step,'training_tokens_current_job':tokens})
            state={'adapter':{k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'config_sha':digest(cfgfile),'epoch':epoch,'offset':base+len(chunk),'step':step,'tokens':tokens,'response_tokens':response_tokens,'exposures':dict(exposures),'elapsed':elapsed+time.perf_counter()-t0,'nonzero':nonzero,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
            torch.save(state,out/'resume.pt.tmp');os.replace(out/'resume.pt.tmp',resume)
        model.save_pretrained(out/f'epoch{epoch+1}',safe_serialization=True)
        atomic(out/f'epoch{epoch+1}'/'metadata.json',{'step':step,'tokens':tokens,'config_sha':digest(cfgfile),'code_commit':revision()})
    assert step==total_steps and nonzero>0 and all(n==cfg['epochs'] for n in exposures.values()) and len(exposures)==len(records)
    result={'state':'complete','steps':step,'training_tokens':tokens,'response_tokens':response_tokens,'nonzero_gradient_updates':nonzero,'exposure_hash':objhash(dict(exposures)),'exposure_counts':dict(exposures),'geometry':geo,'optimizer_state_dtypes':dict(collections.Counter(str(v.dtype) for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))),'config_sha':digest(cfgfile),'flops_proxy_6_base_params_tokens':6*sum(p.numel() for p in model.parameters())*tokens,'seconds':elapsed+time.perf_counter()-t0}
    atomic(out/'complete.json',result);del model,optimizer,scheduler,params,state
    import gc
    gc.collect();torch.cuda.empty_cache();return result
