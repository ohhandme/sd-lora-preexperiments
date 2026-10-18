"""Budget-matched continuation and capacity ablations, with resumable checkpoints."""
import math
from common import *
from peft import LoraConfig,TaskType,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file
from spec_research.training import batch_record


def initialize(rank=16,wide=False):
    if rank==16 and not wide:
        return PeftModel.from_pretrained(load_model('0.6B'),selected('mix16'),is_trainable=True)
    modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if wide else ['q_proj','v_proj']
    model=get_peft_model(load_model('0.6B'),LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=.05,
                  bias='none',target_modules=modules,task_type=TaskType.CAUSAL_LM))
    old=load_file(str(selected('mix16')/'adapter_model.safetensors'))
    state=get_peft_model_state_dict(model)
    with torch.no_grad():
        for k,v in state.items():
            if 'lora_B' in k:v.zero_()
        for k,v in old.items():
            assert k in state
            new=state[k]
            if 'lora_A' in k:
                assert new.shape[1]==v.shape[1] and new.shape[0]>=v.shape[0]
                new[:v.shape[0]].copy_(v.to(new.device,new.dtype))
                assert torch.equal(new[:v.shape[0]].cpu(),v.to(new.dtype))
            elif 'lora_B' in k:
                assert new.shape[0]==v.shape[0] and new.shape[1]>=v.shape[1]
                new[:,:v.shape[1]].copy_(v.to(new.device,new.dtype))
                assert torch.equal(new[:,:v.shape[1]].cpu(),v.to(new.dtype))
            else:raise AssertionError(k)
        assert all(torch.count_nonzero(v)==0 for k,v in state.items() if 'lora_B' in k and k not in old)
    set_peft_model_state_dict(model,state)
    return model


def subset(method,splits):
    if method in ('pooled','weighted'):return splits['train']
    group,index=method.rsplit('_',1)
    if group in ('rank32','wide'):group='mismatch'
    mapping=read(OUT/'assignments.json')['maps'][group]
    return [r for r in splits['train'] if mapping[r['id']]==int(index)]


def train_job(method,splits,binding):
    directory=EXP/'adapters'/method
    done=directory/'complete.json'
    if done.exists():
        item=read(done);assert item['binding']==binding
        for s,record in item['checkpoints'].items():
            assert digest(directory/('s'+s)/'adapter_model.safetensors')==record['weight_sha']
        return item
    gpu_idle();seed_all(42)
    rows=subset(method,splits);ordered=list(rows);random.Random(42).shuffle(ordered)
    rank=32 if method.startswith('rank32') else 16
    wide=method.startswith('wide')
    snapshots=[10,80,160] if method in ('pooled','weighted') else [32] if rank==32 or wide else [2,16,32]
    cfg={'method':method,'binding':binding,'script_sha':digest(__file__),'rank':rank,'wide':wide,
         'row_ids':[r['id'] for r in ordered],'lr':5e-5,'accumulation':4,'epochs':1,'snapshots':snapshots}
    cfgpath=directory/'config.json'
    if cfgpath.exists():assert read(cfgpath)==cfg
    else:atomic(cfgpath,cfg)
    model=initialize(rank,wide).eval()
    old=load_matrix(rows,'old_mix16','train',binding)
    warmcheck=diagnostic(model,rows[0])
    initialization={'algebraic_warmstart_preserved':True,'initial_proposal_matches_mix16':warmcheck==old[rows[0]['id']],
                    'row_id':rows[0]['id'],'rank':rank,'wide':wide}
    warm={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    model.train();model.config.use_cache=False;model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad]
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    step=0;input_tokens=0;response_tokens=0;weighted_tokens=0;elapsed=0.;checkpoints={};losses=[]
    resume=directory/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['config_sha']==digest(cfgpath)
        set_peft_model_state_dict(model,state['adapter']);optimizer.load_state_dict(state['optimizer'])
        step=state['step'];input_tokens=state['input_tokens'];response_tokens=state['response_tokens']
        weighted_tokens=state['weighted_tokens'];elapsed=state['seconds'];checkpoints=state['checkpoints'];losses=state['losses']
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
        del state
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for offset in range(step*4,len(ordered),4):
        chunk=ordered[offset:offset+4];optimizer.zero_grad(set_to_none=True)
        batchloss=[]
        for original in chunk:
            row={**original,'response_ids':original['response_ids'][:128]}
            x,y=batch_record(row,model.device)
            input_tokens+=x.numel();response_tokens+=len(row['response_ids'])
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits
                ce=torch.nn.functional.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
                      y[:,1:].reshape(-1),ignore_index=-100,reduction='none')
                weight=(y[:,1:].reshape(-1)!=-100).float()
                if method=='weighted':
                    hot=set()
                    for w in old[row['id']]['windows']:
                        if w['a']<8:hot.update(range(w['position']+w['a'],min(w['position']+w['a']+3,len(row['response_ids']))))
                    for t in hot:weight[len(row['prompt_ids'])-1+t]=2.
                    weighted_tokens+=len(hot)
                loss=(ce*weight).sum()/weight.sum()
            assert torch.isfinite(loss)
            (loss/len(chunk)).backward();batchloss.append(float(loss.detach()))
            del logits,ce,weight,loss,x,y
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert math.isfinite(float(norm)) and float(norm)>0
        optimizer.step();step+=1;losses.append(mean(batchloss))
        assert all(p._version==frozen[n] and p.grad is None for n,p in model.named_parameters() if n in frozen)
        assert all(torch.isfinite(p).all().item() for p in params)
        if step%4==0 or step in snapshots:
            status('training',method=method,step=step,total_steps=len(ordered)//4,loss=losses[-1])
        if step in snapshots:
            path=directory/f's{step}';model.save_pretrained(path,safe_serialization=True)
            state=get_peft_model_state_dict(model)
            delta=math.sqrt(sum(float(((v.detach().cpu().float()-warm[k].float())**2).sum()) for k,v in state.items()))
            checkpoint={'step':step,'input_tokens':input_tokens,'response_tokens':response_tokens,
                        'samples':offset+len(chunk),'weight_sha':digest(path/'adapter_model.safetensors'),
                        'delta_l2_from_warmstart':delta,'seconds':elapsed+time.perf_counter()-start}
            checkpoints[str(step)]=checkpoint;atomic(path/'metadata.json',checkpoint)
            saved={'config_sha':digest(cfgpath),'adapter':{k:v.detach().cpu() for k,v in state.items()},
                   'optimizer':optimizer.state_dict(),'step':step,'input_tokens':input_tokens,
                   'response_tokens':response_tokens,'weighted_tokens':weighted_tokens,
                   'seconds':elapsed+time.perf_counter()-start,'checkpoints':checkpoints,'losses':losses,
                   'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
            torch.save(saved,directory/'resume.tmp');os.replace(directory/'resume.tmp',resume)
            del state,saved
    result={'state':'complete','binding':binding,'method':method,'steps':step,'requests':len(rows),
            'input_tokens':input_tokens,'response_tokens':response_tokens,'weighted_tokens':weighted_tokens,
            'checkpoints':checkpoints,'base_frozen':True,'initialization':initialization,
            'trainable_parameters':sum(p.numel() for p in params),'rank':rank,'wide':wide,
            'peak_gib':torch.cuda.max_memory_allocated()/1024**3,'seconds':elapsed+time.perf_counter()-start,
            'first_four_step_loss':mean(losses[:4]),'last_four_step_loss':mean(losses[-4:])}
    atomic(done,result)
    del model,optimizer,params,warm;cleanup()
    return result


def stage1_train(splits,binding):
    results={m:train_job(m,splits,binding) for m in [f'{p}_{i}' for p in POOLS for i in range(5)]+['pooled','weighted']}
    for p in POOLS:
        for field in ('input_tokens','response_tokens','requests','steps'):
            assert sum(results[f'{p}_{i}'][field] for i in range(5))==results['pooled'][field]
    for field in ('input_tokens','response_tokens','requests','steps'):
        assert results['weighted'][field]==results['pooled'][field]
    atomic(OUT/'stage1_training.json',results)


def stage4_train(splits,binding):
    results={m:train_job(m,splits,binding) for m in [f'{p}_{i}' for p in ('rank32','wide') for i in range(5)]}
    for p in ('rank32','wide'):
        for i in range(5):
            old=read(EXP/'adapters'/f'mismatch_{i}'/'complete.json')
            for field in ('input_tokens','response_tokens','requests','steps'):
                assert results[f'{p}_{i}'][field]==old[field]
    atomic(OUT/'stage4_training.json',results)
