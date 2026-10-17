"""Real resident BF16 component microbenchmarks; separate from throughput runs."""
import argparse,statistics,time
import torch
from spec_research.common import *
from spec_research.data import partition
from spec_research.decoding import CachedModel,sync
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--scale',choices=['0.6B','4B','8B'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with gpu_lock(),torch.inference_mode():
        m=load_model(a.scale);library=[r['prompt_ids'] for d in DOMAINS for r in partition(d,'expert_train')]
        # Join varied real prompts only for component profiling, never training/acceptance.
        real=[t for ids in library for t in ids]
        for context in (256,1024,2048):
            for op in ('prefill','decode'):
                times=[]
                for rep in range(110):
                    offset=(rep*2053)%(len(real)-context-1);prefix=real[offset:offset+context];c=CachedModel(m)
                    if op=='decode':c.feed(prefix)
                    sync();t=time.perf_counter();c.feed(prefix if op=='prefill' else [real[offset+context]]);sync();ms=(time.perf_counter()-t)*1000
                    if rep>=10:times.append(ms)
                append(a.output,{'scale':a.scale,'context':context,'operation':op,'warmup':10,'repeats':100,'median_ms':statistics.median(times),'p95_ms':sorted(times)[94],'backend':m.config._attn_implementation,'precision':'bfloat16','weights_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'provenance':provenance()})
