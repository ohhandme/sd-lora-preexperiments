"""Causal BF16 logits/KV must not depend on future candidate tokens or block length."""
import copy
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from spec_research.decoding import CachedModel
from spec_research.numerics import install


@pytest.mark.parametrize('prefix_len', [31, 32, 33])
def test_bf16_cached_future_invariance(prefix_len):
    assert torch.cuda.is_available(), 'P0 requires the real spec GPU; do not silently skip'
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    model = install(Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval())
    with torch.inference_mode():
        seed = CachedModel(model)
        seed.feed([2 + i % 40 for i in range(prefix_len)])
        ref = CachedModel(model); ref.cache = copy.deepcopy(seed.cache)
        logits = ref.feed([7])[0]
        for future in (4, 8, 12):
            for value in (9, 17):
                candidate = CachedModel(model); candidate.cache = copy.deepcopy(seed.cache)
                out = candidate.feed([7] + [value] * future)[0]
                assert torch.equal(logits, out), 'Future candidates changed earlier BF16 logits'
                for a, b in zip(ref.cache.layers, candidate.cache.layers):
                    assert torch.equal(a.keys, b.keys[:, :, :prefix_len+1])
                    assert torch.equal(a.values, b.values[:, :, :prefix_len+1])
                del candidate, out
        assert logits.dtype == torch.float32
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    del model, seed, ref
    torch.cuda.empty_cache()
