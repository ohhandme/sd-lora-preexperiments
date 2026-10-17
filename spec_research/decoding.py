"""Greedy cached AR and draft/verify. No token filtering or sampling."""
import time
import torch
from transformers import DynamicCache

class CachedModel:
    def __init__(self,model):self.model=model;self.cache=DynamicCache(config=model.config);self.calls=0
    def length(self):return self.cache.get_seq_length()
    def crop(self,n):
        assert n>=0;self.cache.crop(n);assert self.length()==n
    def feed(self,ids):
        assert ids
        start=self.length();x=torch.tensor([ids],dtype=torch.long,device=self.model.device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=self.model.device.type=='cuda'):
            out=self.model(input_ids=x,past_key_values=self.cache,use_cache=True,cache_position=torch.arange(start,start+len(ids),device=x.device),position_ids=torch.arange(start,start+len(ids),device=x.device).unsqueeze(0),attention_mask=torch.ones((1,start+len(ids)),dtype=torch.long,device=x.device))
        self.cache=out.past_key_values;self.calls+=1
        assert self.length()==start+len(ids)
        return out.logits[0].float()
    def next(self,committed):
        n=self.length();assert n<len(committed),(n,len(committed))
        return self.feed(committed[n:])[-1]

def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()
def top(logits):return int(logits.argmax().item())
def margin(logits):
    v=logits.topk(2);return {'top_ids':v.indices.tolist(),'top_logits':v.values.tolist(),'margin':float(v.values[0]-v.values[1])}

@torch.inference_mode()
def ar(model,prompt,max_new,eos):
    sync();t0=time.perf_counter(); c=CachedModel(model);committed=list(prompt);out=[];first=None; traces=[]
    for _ in range(max_new):
        logits=c.next(committed);token=top(logits);committed.append(token);out.append(token)
        traces.append(margin(logits))
        if first is None:first=time.perf_counter()
        if token in eos:break
    sync();end=time.perf_counter()
    return {'tokens':out,'latency_ms':(end-t0)*1000,'ttft_ms':((first or end)-t0)*1000,'decode_seconds':end-(first or end),'target_calls':c.calls,'margins':traces}

@torch.inference_mode()
def speculative(target,draft,prompt,max_new,eos,L):
    sync();t0=time.perf_counter();tc=CachedModel(target);dc=CachedModel(draft);committed=list(prompt);out=[];rounds=[];traces=[];first=None
    # Use the identical full prompt prefill as cached AR. Reuse its next-token
    # logits for the first verification, rather than splitting the last token.
    first_logits=tc.feed(prompt)[-1]
    prefill_calls=tc.calls
    while len(out)<max_new:
        base=len(committed);proposal=[];draft_prefix=list(committed)
        for _ in range(min(L,max_new-len(out))):
            token=top(dc.next(draft_prefix));proposal.append(token);draft_prefix.append(token)
            if token in eos:break
        if first_logits is not None:
            assert tc.length()==base
            logits=torch.cat((first_logits.unsqueeze(0),tc.feed(proposal)),dim=0)
            first_logits=None
        else:
            assert tc.length()==base-1
            logits=tc.feed([committed[-1]]+proposal)
        accepted=0;emitted=[];done=False
        for i,token in enumerate(proposal):
            prediction=top(logits[i]);traces.append(margin(logits[i]))
            if token==prediction:
                accepted+=1;emitted.append(token)
            else:emitted.append(prediction)
            if emitted[-1] in eos or len(out)+len(emitted)>=max_new:done=True;break
            if token!=prediction:break
        else:
            if len(out)+len(emitted)<max_new:
                token=top(logits[len(proposal)]);emitted.append(token);traces.append(margin(logits[len(proposal)]))
                if token in eos:done=True
        assert emitted
        out.extend(emitted);committed.extend(emitted)
        tc.crop(len(committed)-1)
        # Keep only cache entries belonging to the newly committed sequence.
        # A full acceptance+bonus can leave two uncached tokens for the next proposal.
        dc.crop(min(dc.length(),base+accepted,len(committed)-1))
        rounds.append({'Lr':len(proposal),'ar':accepted,'nr':len(emitted),'prefix_len':base,'target_cache':tc.length(),'draft_cache':dc.length()})
        if first is None:first=time.perf_counter()
        if done:break
    sync();end=time.perf_counter()
    return {'tokens':out,'latency_ms':(end-t0)*1000,'ttft_ms':((first or end)-t0)*1000,'decode_seconds':end-(first or end),'rounds':rounds,'target_prefill_calls':prefill_calls,'target_verify_calls':tc.calls-prefill_calls,'target_calls':tc.calls,'draft_calls':dc.calls,'margins':traces}

@torch.inference_mode()
def public_prefix(draft,prompt,reference,L,positions=(0,32,64,128,192)):
    windows=[]
    for j in positions:
        if len(reference)-j<L+1:continue
        cache=CachedModel(draft);context=list(prompt)+reference[:j];proposal=[]
        for _ in range(L):
            token=top(cache.next(context));proposal.append(token);context.append(token)
        a=0
        for x,y in zip(proposal,reference[j:]):
            if x!=y:break
            a+=1
        windows.append({'position':j,'a':a,'proposal':proposal})
    return {'windows':windows,'mean_a':sum(w['a'] for w in windows)/len(windows) if windows else None,'survival':[sum(w['a']>=k for w in windows)/len(windows) for k in range(1,L+1)] if windows else None}

def compare(reference,candidate):
    a,b=reference['tokens'],candidate['tokens']
    if a==b:return None
    j=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    return {'earliest_divergence':j,'ar_token':a[j] if j<len(a) else None,'sd_token':b[j] if j<len(b) else None,'ar_margin':reference['margins'][j] if j<len(reference['margins']) else None,'sd_margin':candidate['margins'][j] if j<len(candidate['margins']) else None,'ar_tokens':a,'sd_tokens':b}
