import contextlib, datetime, hashlib, json, os, pathlib, subprocess, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = '/root/miniconda3/envs/spec'
DOMAINS = ('math', 'code', 'general')
def assert_spec():
    assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(SPEC), (sys.prefix, SPEC)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    for key,subdir in {'HF_HOME':'hf','TORCH_HOME':'torch','CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton'}.items():
        os.environ[key]=str(ROOT/'cache'/subdir)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def objhash(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def read(path): return json.loads(pathlib.Path(path).read_text())
def atomic(path, obj):
    p=pathlib.Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.')
    with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def append(path,obj):
    p=pathlib.Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f: f.write(json.dumps(obj,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
def rows(path):
    with pathlib.Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def protocol(): return read(ROOT/'configs/protocol.json')
def assets(): return read(ROOT/'configs/assets.local.json')
def model_info(scale): return assets()['models']['Qwen/Qwen3-'+scale]
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_info('8B')['path'],local_files_only=True,trust_remote_code=False)
def prompt_ids(tok,prompt):
    return tok.apply_chat_template([{'role':'system','content':protocol()['system']},{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
def load_model(scale):
    import torch
    from transformers import AutoModelForCausalLM
    m=AutoModelForCausalLM.from_pretrained(model_info(scale)['path'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    from .numerics import install
    m.requires_grad_(False); m.eval(); return install(m)
def revision():
    r=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else 'UNCOMMITTED'
def sources():
    paths=list((ROOT/'spec_research').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list(ROOT.glob('*.py'))+list((ROOT/'configs').glob('*.json'))+[ROOT/'program.md',ROOT/'scripts/run_seed42.sh']
    paths.extend((ROOT/'tests/fixtures').glob('*.pt'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths) if p.exists()}
def provenance():
    from .numerics import POLICY
    return {'prefix':sys.prefix,'interpreter':sys.executable,'code_commit':revision(),'source_hashes':sources(),'config_sha':digest(ROOT/'configs/protocol.json'),'at':now(),'models':assets()['models'],'numerical_policy':dict(POLICY)}
def status(**kw):
    p=ROOT/'manifests/execution_status.json'; old=read(p) if p.exists() else {}
    old.update(kw,updated_at=now(),interpreter=sys.executable,prefix=sys.prefix,pid=os.getpid()); atomic(p,old)
@contextlib.contextmanager
def gpu_lock(inherited_fd=None):
    import fcntl
    p=ROOT/'manifests/gpu_pipeline.lock'; p.parent.mkdir(exist_ok=True)
    if inherited_fd is not None:
        assert os.fstat(inherited_fd).st_ino==p.stat().st_ino,'Invalid inherited GPU lock'
        fcntl.flock(inherited_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield inherited_fd
        return
    with p.open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); f.write(str(os.getpid()));f.flush();yield f.fileno()

def seed_all(seed):
    import random,numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
