import fcntl
from common import *
boot()
with (ROOT/'manifests/gpu_pipeline.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_idle()
    if not (OUT/'protected.json').exists():atomic(OUT/'protected.json',protected())
    splits,binding=data()
    source(splits,binding)
    atomic(OUT/'prepared.json',{'at':now(),'binding':binding,'protected_files':protect_check()})
