"""Download official frozen base models on a fresh restore machine."""
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path('/root/autodl-tmp/sd-lora-preexp')
models=[('0.6B','draft','c1899de289a04d12100db370d81485cdf75e47ca'),('4B','draft','1cfa9a7208912126459214e8b04321603b3df60c'),('8B','target','b968826d9c46dd6066d109eabc6255188de91218')]
for size,kind,revision in models:
    dest=root/'models'/kind/('Qwen3-'+size)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'Refusing to overwrite populated directory: {dest}')
    snapshot_download(repo_id='Qwen/Qwen3-'+size,revision=revision,local_dir=dest)
