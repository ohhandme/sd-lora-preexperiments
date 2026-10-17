"""Verify exported bytes without importing models or acquiring GPU resources."""
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
m=json.loads((root/'SHA256SUMS.json').read_text())
for name,expected in m.items():
    p=root/name
    if not p.is_file():raise SystemExit('Missing: '+name)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise SystemExit('SHA mismatch: '+name)
print(f'OK: {len(m)} files')
