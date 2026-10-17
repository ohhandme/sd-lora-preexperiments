"""Prompt-only local readers; no dataset scripts or benchmark execution."""
import collections, pathlib, random, re, unicodedata
from .common import *

def normalize(s): return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def shingles(s):
    w=re.findall(r'\w+|[^\w\s]',normalize(s)); n=5
    return {' '.join(w[i:i+n]) for i in range(max(1,len(w)-n+1))}
class Deduper:
    def __init__(self):
        from datasketch import MinHashLSH
        self.lsh=MinHashLSH(threshold=.85,num_perm=128);self.sets={};self.exact={};self.serial=0
    def check(self,text,add=True):
        from datasketch import MinHash
        norm=normalize(text);h=objhash(norm)
        if h in self.exact:return 'exact'
        s=shingles(text);m=MinHash(num_perm=128,seed=20261015)
        m.update_batch([x.encode() for x in sorted(s)])
        for key in self.lsh.query(m):
            t=self.sets[key]
            if len(s&t)/len(s|t)>=.85:return 'near'
        if add:
            key=str(self.serial);self.serial+=1;self.lsh.insert(key,m);self.sets[key]=s;self.exact[h]=key
        return None

def parquet(paths,columns):
    import pyarrow.parquet as pq
    for path in sorted(paths):
        pf=pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512,columns=columns):
            yield from batch.to_pylist()
def source_records(domain,split):
    a=assets()['datasets']
    if domain=='math':
        ds='openai/gsm8k'; it=parquet(pathlib.Path(a[ds]['path']).glob('main/'+split+'*.parquet'),['question'])
        for i,r in enumerate(it):yield ds,split,str(i),r['question']
    elif domain=='code':
        ds='ise-uiuc/Magicoder-OSS-Instruct-75K'
        for i,r in enumerate(rows(pathlib.Path(a[ds]['path'])/'data-oss_instruct-decontaminated.jsonl')):
            if r['lang'].lower()=='python':yield ds,'train',str(i),r['problem']
    elif domain=='general':
        ds='HuggingFaceH4/ultrachat_200k';seen=set()
        for r in parquet(pathlib.Path(a[ds]['path']).glob('data/'+split+'*.parquet'),['prompt_id','messages']):
            key=str(r['prompt_id'])
            if key in seen:continue
            seen.add(key)
            msgs=[x['content'] for x in r['messages'] if x['role']=='user']
            if msgs:yield ds,split,key,msgs[0]
def external_inputs():
    yield from (x[3] for x in source_records('math','test'))
    a=assets()['datasets']
    for r in parquet(pathlib.Path(a['google-research-datasets/mbpp']['path']).glob('sanitized/test*.parquet'),['prompt']):yield r['prompt']
    for r in parquet(pathlib.Path(a['openai/openai_humaneval']['path']).rglob('*.parquet'),['prompt']):yield r['prompt']

def prepare():
    assert_spec(); p=ROOT/'data/processed';p.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'manifests/data.json'
    if manifest.exists():validate_data();return read(manifest)
    tok=tokenizer();d=Deduper();excluded=collections.Counter();candidates={}; general_test=[]
    for text in external_inputs(): d.check(text)
    # Reserve independent general test inputs before training; no answers accessed.
    for ds,split,sid,text in source_records('general','test_sft'):
        reason=d.check(text)
        if reason:excluded['general_test_'+reason]+=1;continue
        ids=prompt_ids(tok,text)
        if len(ids)>1024:excluded['general_test_length']+=1;continue
        general_test.append(record(ds,split,sid,text,ids,'general'))
    rng=random.Random(20261015);rng.shuffle(general_test)
    if len(general_test)<300:raise RuntimeError('Insufficient disjoint general test prompts')
    for domain in DOMAINS:
        pool=[]
        for ds,split,sid,text in source_records(domain,'train_sft' if domain=='general' else 'train'):
            if not isinstance(text,str) or not text.strip():excluded[domain+'_empty']+=1;continue
            reason=d.check(text)
            if reason:excluded[domain+'_'+reason]+=1;continue
            ids=prompt_ids(tok,text)
            if len(ids)>1024:excluded[domain+'_length']+=1;continue
            pool.append(record(ds,split,sid,text,ids,domain))
        rng=random.Random(20261015);rng.shuffle(pool);candidates[domain]=pool
        print(json.dumps({'domain':domain,'eligible':len(pool),'excluded':dict(excluded)}),flush=True)
        need=2700 if domain=='general' else 3000
        if len(pool)<need:raise RuntimeError(f'{domain}: only {len(pool)} eligible, need {need}; protocol amendment required; no duplicate padding')
    files={};counts={}
    for domain,pool in candidates.items():
        offset=0
        for partition,n in [('expert_train',2000),('adapter_val',200),('router_train',400),('router_val',100)]+([] if domain=='general' else [('id_test',300)]):
            selected=pool[offset:offset+n];offset+=n
            write_partition(p,domain,partition,selected,files,counts)
    write_partition(p,'general','id_test',general_test[:300],files,counts)
    result={'state':'complete','seed':20261015,'counts':counts,'files':files,'eligible':{k:len(v) for k,v in candidates.items()},'excluded':dict(excluded),'dedup':protocol()['dedup'],'general_test_reserved_inputs':len(general_test),'provenance':provenance(),'final_test_locked':True}
    atomic(manifest,result);validate_data();return result

def record(ds,split,sid,text,ids,domain):
    return {'id':ds+':'+split+':'+sid,'dataset':ds,'revision':assets()['datasets'][ds]['revision'],'split':split,'source_id':sid,'domain':domain,'prompt':text,'prompt_ids':ids,'prompt_hash':objhash(normalize(text)),'prompt_tokens':len(ids)}
def write_partition(p,domain,partition,selected,files,counts):
    path=p/(domain+'.'+partition+'.jsonl')
    if path.exists():raise RuntimeError(f'Conflicting unfinished preparation: {path}')
    with path.open('x') as f:
        for r in selected:r['partition']=partition;f.write(json.dumps(r)+'\n')
    key=domain+'.'+partition;files[str(path.relative_to(ROOT))]=digest(path);counts[key]=len(selected)
def validate_data():
    m=read(ROOT/'manifests/data.json');seen=set();hashes=set();groups=set()
    expected={f'{d}.{p}' for d in DOMAINS for p in ('expert_train','adapter_val','router_train','router_val','id_test')}
    assert set(m['counts'])==expected and len(m['files'])==15,'Missing partition counts/files'
    for path,sha in m['files'].items():
        assert digest(ROOT/path)==sha,path
        rr=list(rows(ROOT/path));key=pathlib.Path(path).stem
        assert len(rr)==m['counts'][key]
        expected={'expert_train':2000,'adapter_val':200,'router_train':400,'router_val':100,'id_test':300}[key.split('.')[1]]
        assert len(rr)==expected,(key,len(rr))
        for r in rr:
            assert r['id'] not in seen and r['prompt_hash'] not in hashes
            assert 0<len(r['prompt_ids'])<=1024
            if r['domain']=='general':
                assert r['source_id'] not in groups,'UltraChat prompt group leaked'
                groups.add(r['source_id'])
            seen.add(r['id']);hashes.add(r['prompt_hash'])
    return m
def partition(domain,name):
    assert name in ('expert_train','adapter_val','router_train','router_val'), 'Final tests are locked'
    return list(rows(ROOT/'data/processed'/f'{domain}.{name}.jsonl'))
def smoke_records():return [r for d in DOMAINS for r in partition(d,'expert_train')[:64]]
