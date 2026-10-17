"""BF16 inference arithmetic with fixed linear rows and causal SDPA rows.

No weights, logits, acceptance rules or training precision are changed. Every
comparison must use this policy for AR, SD, teacher and model.generate alike.
"""
import types
import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

AR_POLICY = {'version': 1, 'linear_rows': 32, 'cached_attention': 'causal_query_rows_sdpa',
          'precision': 'bfloat16', 'reduced_precision_reduction': False}
POLICY = {**AR_POLICY, 'version': 2,
          'normalization': 'requestwise_prefill_tokenwise_cached_rmsnorm'}


def _install_norm(module, context):
    original = module.forward
    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            return original(x)
        if x.shape[0] > 1:
            return torch.cat([forward(self, x[b:b+1].contiguous())
                              for b in range(x.shape[0])], dim=0)
        if context['cached'] and x.shape[1] > 1:
            return torch.cat([original(x[:, i:i+1].contiguous())
                              for i in range(x.shape[1])], dim=1)
        # Original full-prompt and single-token AR arithmetic stays identical.
        return original(x)
    module.forward = types.MethodType(forward, module)
    module._stable_rmsnorm = True


def _linear(self, x):
    if self.training or torch.is_grad_enabled():
        return F.linear(x, self.weight, self.bias)
    flat = x.reshape(-1, x.shape[-1])
    chunks = []
    for row in range(0, flat.shape[0], POLICY['linear_rows']):
        part = flat[row:row + POLICY['linear_rows']]
        n = part.shape[0]
        if n < POLICY['linear_rows']:
            part = F.pad(part, (0, 0, 0, POLICY['linear_rows'] - n))
        chunks.append(F.linear(part, self.weight, self.bias)[:n])
    return torch.cat(chunks, dim=0).reshape(*x.shape[:-1], self.out_features)


def _attention(module, query, key, value, attention_mask, dropout=0., scaling=None,
               is_causal=None, **kwargs):
    if not getattr(module, '_stable_bf16', False) or module.training or torch.is_grad_enabled():
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    qn, kn = query.shape[-2], key.shape[-2]
    # Equal-length prompt prefill is identical in AR and SD. Retain fast SDPA.
    if qn == kn:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    assert query.shape[0] == 1 and kn > qn and dropout == 0
    assert not kwargs.get('sliding_window'), 'Only full causal Qwen3 attention supported'
    out = []
    for i in range(qn):
        end = kn - qn + i + 1
        if attention_mask is not None:
            assert attention_mask.ndim == 4
            allowed = attention_mask[:, :, i:i+1, :end]
            allowed = allowed if allowed.dtype == torch.bool else allowed == 0
        else:
            allowed = torch.ones((query.shape[0], 1, 1, end), dtype=torch.bool, device=query.device)
        a, _ = sdpa_attention_forward(module, query[:, :, i:i+1].contiguous(),
                                     key[:, :, :end].contiguous(), value[:, :, :end].contiguous(),
                                     allowed.contiguous(), dropout=0., scaling=scaling, is_causal=False, **kwargs)
        out.append(a)
    return torch.cat(out, dim=1), None


def install(model):
    assert model.config.model_type == 'qwen3'
    assert model.config._attn_implementation == 'sdpa'
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ALL_ATTENTION_FUNCTIONS.register('sdpa', _attention)
    if getattr(model, '_stable_norm_context', None) is None:
        context = {'cached': False}
        def before_forward(module, args, kwargs):
            cache = kwargs.get('past_key_values')
            context['cached'] = cache is not None and cache.get_seq_length() > 0
        model.register_forward_pre_hook(before_forward, with_kwargs=True)
        model._stable_norm_context = context
    context = model._stable_norm_context
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and not getattr(module, '_stable_bf16', False):
            module.forward = types.MethodType(_linear, module)
            module._stable_bf16 = True
        if module.__class__.__name__ == 'Qwen3Attention':
            module._stable_bf16 = True
        if module.__class__.__name__ == 'Qwen3RMSNorm' and not getattr(module, '_stable_rmsnorm', False):
            _install_norm(module, context)
    model._numerical_policy = dict(POLICY)
    return model
