import collections,importlib.metadata,platform,subprocess,time
import torch
from .common import *
from .training import attach,geometry,batch_record

def command(args):
    r=subprocess.run(args,text=True,capture_output=True);assert r.returncode==0,(args,r.stderr);return r.stdout

def check_environment():
    assert_spec();report={'state':'running','provenance':provenance(),'stages':{},'prefix':sys.prefix}
    path=ROOT/'results/smoke/environment.json';atomic(path,report)
    def stage(name,value):report['stages'][name]=value;atomic(path,report)
    try:
        stage('hardware',{'nvidia_smi':command(['nvidia-smi','-q']),'platform':platform.platform(),'cpu':platform.processor(),'ram':command(['free','-b']),'container':pathlib.Path('/proc/1/cgroup').read_text(),'gpu_processes':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])})
        assert not report['stages']['hardware']['gpu_processes'].strip(),'Other GPU process present'
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        stage('libraries',{'versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','peft','accelerate','datasets','numpy','scikit-learn','pyarrow','safetensors']},'torch_build_cuda':torch.version.cuda,'compute_capability':torch.cuda.get_device_capability(),'compiled_arches':torch.cuda.get_arch_list(),'bf16_supported':torch.cuda.is_bf16_supported()})
        assert tuple(map(int,torch.version.cuda.split('.')[:2]))>=(12,8)
        a=torch.randn(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);(a@a.T).float().mean().backward();assert torch.isfinite(a.grad).all()
        q=torch.randn(1,4,32,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            y=torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True);y.float().square().mean().backward();torch.cuda.synchronize()
        kernels=[x.key for x in prof.key_averages() if 'attention' in x.key or 'flash' in x.key or 'scaled_dot' in x.key]
        assert torch.isfinite(q.grad).all();stage('gpu_kernels',{'cuda_matmul_forward_backward':True,'sdpa_forward_backward':True,'observed_ops':kernels,'enabled':{'flash':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled()}})
        del a,q,y
        tok=tokenizer();vocab=tok.get_vocab();special=tok.special_tokens_map;template=tok.chat_template;test=['Hello world!','def f(x):\n    return x + 1','计算 2+2 = ?','<|im_end|>']
        from transformers import AutoTokenizer
        compatible={}
        for scale in ['8B','0.6B','4B']:
            t=AutoTokenizer.from_pretrained(model_info(scale)['path'],local_files_only=True,trust_remote_code=False)
            assert t.get_vocab()==vocab and t.special_tokens_map==special and t.chat_template==template,scale
            for s in test:assert t.encode(s)==tok.encode(s) and prompt_ids(t,s)==prompt_ids(tok,s)
            compatible[scale]={'vocab_size':len(vocab),'vocab_sha':objhash(vocab),'special_sha':objhash(special),'template_sha':objhash(template)}
        stage('tokenizers',compatible)
        forward={}
        for scale in ['8B','0.6B','4B']:
            m=load_model(scale);ids=prompt_ids(tok,'What is 2 + 2?');x=torch.tensor([ids],device='cuda')
            with torch.inference_mode():o=m(input_ids=x,use_cache=False)
            assert o.logits.dtype==torch.bfloat16 and torch.isfinite(o.logits).all()
            info={'logits_dtype':str(o.logits.dtype),'shape':list(o.logits.shape),'backend':m.config._attn_implementation,'weight_dtypes':dict(collections.Counter(str(p.dtype) for p in m.parameters()))}
            from transformers import GenerationConfig
            from .decoding import ar,sync
            eos=m.generation_config.eos_token_id;eos_set=set([eos] if isinstance(eos,int) else eos)
            gcfg=GenerationConfig(do_sample=False,max_new_tokens=16,use_cache=True,eos_token_id=eos,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id)
            with torch.inference_mode():
                m.generate(x,attention_mask=torch.ones_like(x),generation_config=gcfg,do_sample=False,use_model_defaults=False)
                sync();started=time.perf_counter();hf=m.generate(x,attention_mask=torch.ones_like(x),generation_config=gcfg,do_sample=False,use_model_defaults=False);sync();hf_ms=(time.perf_counter()-started)*1000
                cached=ar(m,ids,16,eos_set)
            assert hf[0,len(ids):].tolist()==cached['tokens'],'Cached AR differs from HF generate'
            info['cached_ar_vs_hf']={'token_match':True,'hf_ms':hf_ms,'cached_ar_ms':cached['latency_ms'],'tokens':len(cached['tokens'])}
            del hf,cached
            if scale!='8B':
                m=attach(m,16);g16=geometry(m,16)
                from peft import LoraConfig,TaskType
                for name,rank in [('code',16),('general',16),('mix48',48)]:
                    m.add_adapter(name,LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=.05,bias='none',target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM))
                g48=geometry(m,48,'mix48');assert g48['modules']==g16['modules']
                activations={}
                for name in ['default','code','general','mix48']:
                    m.set_adapter(name);m.eval()
                    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):z=m(input_ids=x,use_cache=False).logits
                    assert torch.isfinite(z).all();activations[name]=True
                m.set_adapter('default');m.train();m.config.use_cache=False
                frozen={n:(p._version,p.detach().flatten()[0].item()) for n,p in m.named_parameters() if not p.requires_grad}
                before={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
                r={'prompt_ids':ids,'response_ids':tok.encode('4.',add_special_tokens=False)+[tok.eos_token_id]};xx,labels=batch_record(r,'cuda')
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    out=m(input_ids=xx,labels=labels,use_cache=False)
                manual=torch.nn.functional.cross_entropy(out.logits[:,:-1].float().reshape(-1,out.logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
                assert torch.allclose(out.loss.float(),manual,rtol=1e-5,atol=1e-5)
                out.loss.backward();pars=[p for p in m.parameters() if p.requires_grad];norm=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);assert norm>0
                opt=torch.optim.AdamW(pars,lr=1e-4);opt.step()
                assert any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if n in before)
                assert all(p._version==v and p.flatten()[0].item()==first and p.grad is None for n,p in m.named_parameters() if n in frozen for v,first in [frozen[n]])
                info.update({'r16':g16,'r48':g48,'resident_activations':activations,'assistant_shifted_ce_checked':True,'gradient_norm':float(norm),'base_unchanged':True,'adapter_updated':True})
                del opt,out,manual,pars,before,xx,labels,z
            forward[scale]=info;stage('models',forward);del m,o,x
            import gc
            gc.collect();torch.cuda.empty_cache()
        report['state']='passed';atomic(path,report);return report
    except BaseException as e:
        report.update(state='failed',error=repr(e));atomic(path,report);raise
