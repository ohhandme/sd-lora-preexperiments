import torch
from spec_research.training import batch_record,order

def test_partial_accumulation_matches_mean():
    # The final three examples must have the same update as their mean loss,
    # rather than being divided by the nominal accumulation size16.
    p=torch.nn.Parameter(torch.tensor([2.]));q=torch.nn.Parameter(p.detach().clone());examples=[(1.,3.),(2.,1.),(4.,7.)]
    for x,y in examples:((p*x-y).square()/len(examples)).backward()
    torch.stack([(q*x-y).square() for x,y in examples]).mean().backward()
    assert torch.allclose(p.grad,q.grad)
def test_assistant_ce_prediction_boundary():
    x,labels=batch_record({'prompt_ids':[11,12,13],'response_ids':[14,15]},'cpu')
    logits=torch.zeros(1,5,20,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,20),labels[:,1:].reshape(-1),ignore_index=-100);loss.backward()
    assert torch.count_nonzero(logits.grad[0,:2])==0
    assert logits.grad[0,2,14]<0 and logits.grad[0,3,15]<0
    assert torch.count_nonzero(logits.grad[0,4])==0

def test_equal_source_exposure_and_shuffle():
    rr=[{'id':f'{d}{i}','domain':d} for d in ('math','code','general') for i in range(5)]
    for epoch in (0,1):
        mixed=order(rr,42,epoch,True)
        assert len(mixed)==len(rr) and sorted(r['id'] for r in mixed)==sorted(r['id'] for r in rr)
        assert mixed==order(rr,42,epoch,True)
        assert all({r['domain'] for r in mixed[i:i+3]}=={'math','code','general'} for i in range(0,len(mixed),3))
