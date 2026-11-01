"""Download one already-complete model cell; preserve failed partial transfers."""
import base64
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import urllib.request

manifest, root, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
files = json.loads(base64.b64decode(''.join(manifest.read_text().split())))

def download(item):
    name = item['file_name']
    if not name.startswith(prefix + '/'):
        raise ValueError('unexpected output prefix')
    rel = Path(name[len(prefix) + 1:])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('unsafe output path')
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('refusing to overwrite completed download')
    part = target.with_name(target.name + '.partial')
    if part.exists():
        raise ValueError('preserving existing partial')
    with urllib.request.urlopen(item['url'], timeout=180) as response, part.open('xb') as out:
        while True:
            block = response.read(1048576)
            if not block:
                break
            out.write(block)
    os.replace(part, target)
    return name, target.stat().st_size

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(download, x): x['file_name'] for x in files}
    failed = []
    for future in concurrent.futures.as_completed(futures):
        try:
            name, size = future.result()
            print('OK', name, size, flush=True)
        except Exception as exc:
            failed.append(futures[future])
            print('ERROR', futures[future], type(exc).__name__, str(exc).split('?')[0], flush=True)
    print('SUMMARY', len(files) - len(failed), len(failed), flush=True)
    if failed:
        sys.exit(1)
