"""Analytical independent oracle with token-dependent context and actual cache storage."""
import types
import pytest,torch
from transformers import Qwen3Config
from spec_research.decoding import ar,speculative

class Toy(torch.nn.Module):
    def __init__(self,fn):
        super().__init__();self.fn=fn;self.device=torch.device('cpu');self.config=Qwen3Config(num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,hidden_size=8,head_dim=8);self.calls=[]
    def forward(self,input_ids,past_key_values,cache_position,position_ids,attention_mask,**kwargs):
        old=past_key_values.get_seq_length();assert cache_position.tolist()==list(range(old,old+input_ids.shape[1]));assert position_ids.tolist()==[cache_position.tolist()];assert attention_mask.shape[1]==old+input_ids.shape[1]
        prev=[] if old==0 else past_key_values.layers[0].keys[0,0,:,0].long().tolist()
        ids=input_ids[0].tolist();logits=[]
        for token in ids:
            prev.append(token);v=torch.full((32,),-20.);v[self.fn(prev)]=20.;logits.append(v)
        kv=input_ids.float().reshape(1,1,-1,1);past_key_values.update(kv,kv,0)
        self.calls.append((old,len(ids)));return types.SimpleNamespace(logits=torch.stack(logits)[None],past_key_values=past_key_values)
def target_fn(p):return (sum((i+1)*x for i,x in enumerate(p))+len(p)*3)%23+1
@pytest.mark.parametrize('match',[0,2,1000])
@pytest.mark.parametrize('prompt',[[2],[4,6,3,9],[8,1,7,6,3,2,9,5,1,4,8,7]])
@pytest.mark.parametrize('cap',[1,3,5,9,19])
def test_boundaries(match,prompt,cap):
    def draft_fn(p):
        expected=target_fn(p)
        return expected if (len(p)-len(prompt))%4<match else expected%23+1
    oracle=[];p=list(prompt)
    for _ in range(cap):oracle.append(target_fn(p));p.append(oracle[-1])
    t=Toy(target_fn);d=Toy(draft_fn);sd=speculative(t,d,prompt,cap,{31},4)
    assert sd['tokens']==oracle
    assert ar(Toy(target_fn),prompt,cap,{31})['tokens']==oracle
    assert all(r['nr']<=r['Lr']+1 and r['ar']<=r['Lr'] and r['target_cache']==r['prefix_len']+r['nr']-1 for r in sd['rounds'])
    assert all(n<=5 for old,n in t.calls[1:]),'Target recomputed prompt'
    if match==0:assert sd['rounds'][0]['ar']==0
    if match==2 and cap>=3:assert sd['rounds'][0]['ar']==2 and sd['rounds'][0]['nr']==3
    if match==1000 and cap>=5:assert sd['rounds'][0]['nr']==5
@pytest.mark.parametrize('eos_offset',[0,1,3,4,7])
def test_eos(eos_offset):
    prompt=[3,4];eos=30
    def f(p):return eos if len(p)-len(prompt)==eos_offset else target_fn(p)
    sd=speculative(Toy(f),Toy(f),prompt,20,{eos},4)
    assert len(sd['tokens'])==eos_offset+1 and sd['tokens'][-1]==eos
    assert sd['tokens']==ar(Toy(f),prompt,20,{eos})['tokens']
def test_mask_shift():
    from spec_research.training import batch_record
    x,y=batch_record({'prompt_ids':[1,2,3],'response_ids':[4,5]},'cpu')
    assert y.tolist()==[[-100,-100,-100,4,5]]
    assert y[:,1:].tolist()==[[-100,-100,4,5]]
