#!/root/miniconda3/envs/spec/bin/python
"""Audit frozen v2 prompts for cross-partition word-shingle near duplicates."""

import argparse
import hashlib
import json
import pathlib
import re
import time

from datasketch import MinHash, MinHashLSH


def shingles(prompt):
    words = re.findall(r'\w+', prompt.lower())
    if len(words) < 5:
        return {' '.join(words)}
    return {' '.join(words[i:i+5]) for i in range(len(words)-4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x]
    lsh = MinHashLSH(threshold=.75, num_perm=128)
    seen = {}
    found = []
    start = time.monotonic()
    for idx, row in enumerate(rows):
        terms = shingles(row['prompt'])
        sketch = MinHash(num_perm=128)
        for term in terms:
            sketch.update(term.encode())
        for prior in lsh.query(sketch):
            old_row, old_terms = seen[prior]
            score = len(terms & old_terms) / len(terms | old_terms)
            if score >= .85:
                found.append({'left_id': old_row['record_id'], 'right_id': row['record_id'],
                              'left_partition': old_row['partition'], 'right_partition': row['partition'],
                              'jaccard': score})
        key = str(idx)
        lsh.insert(key, sketch)
        seen[key] = row, terms
    result = {'state': 'passed' if not found else 'failed', 'count': len(rows),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'threshold': .85, 'word_shingles': 5, 'num_perm': 128,
              'pairs': found, 'seconds': time.monotonic()-start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temporary.replace(args.out)
    print(json.dumps({'state': result['state'], 'count': len(rows),
                      'near_duplicate_pairs': len(found), 'seconds': result['seconds']}))
    if found:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
