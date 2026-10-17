"""Regress actual large-model reduction counterexample, not only tiny models."""
import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from spec_research.common import ROOT
from spec_research.numerics import _install_norm

@pytest.mark.parametrize('mode', ['requests', 'cached_tokens', 'unchanged_ar'])
def test_real_rmsnorm_counterexample(mode):
    assert torch.cuda.is_available(), 'P0 requires actual CUDA verification'
    data=torch.load(ROOT/'tests/fixtures/rmsnorm_counterexample.pt',weights_only=True)
    x=data['input'].cuda();module=Qwen3RMSNorm(4096,eps=data['eps']).to(device='cuda',dtype=torch.bfloat16).eval()
    module.weight.data.copy_(data['weight'].cuda());original=module.forward
    context={'cached': mode!='unchanged_ar'};_install_norm(module,context)
    with torch.inference_mode():
        if mode=='requests':
            reference=torch.cat([original(x[b:b+1].contiguous()) for b in range(8)],0)
            assert torch.equal(module(x),reference)
        elif mode=='cached_tokens':
            xx=x.transpose(0,1).contiguous()
            reference=torch.cat([original(xx[:,i:i+1].contiguous()) for i in range(8)],1)
            assert torch.equal(module(xx),reference)
        else:
            xx=x.transpose(0,1).contiguous()
            assert torch.equal(module(xx),original(xx))
            context['cached']=True
            assert torch.equal(module(x[:1]),original(x[:1]))
