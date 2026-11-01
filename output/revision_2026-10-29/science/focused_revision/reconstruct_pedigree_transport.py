"""Reconstruct exactly two original pedigree DBs from an approved chunk manifest."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def reconstruct(manifest_path,manifest_hash,chunks_dir,out):
    assert digest(manifest_path)==manifest_hash
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['format']=='gzip-mtime-0-chunks-v1'
    files=manifest['files'];assert [x['name'] for x in files]==['first','second']
    expected_names=[c['name'] for item in files for c in item['chunks']]
    assert len(expected_names)==len(set(expected_names))
    assert {p.name for p in chunks_dir.iterdir() if p.is_file()}==set(expected_names)
    assert not out.exists();out.mkdir(parents=True)
    for item in files:
        packed=out/(item['name']+'.gz');offset=0
        with packed.open('xb') as dest:
            for i,c in enumerate(item['chunks']):
                assert c['name']==f"{item['name']}-{i:03d}.bin" and c['offset']==offset
                p=chunks_dir/c['name'];assert p.stat().st_size==c['length'] and digest(p)==c['sha256']
                with p.open('rb') as src:shutil.copyfileobj(src,dest)
                offset+=c['length']
        assert offset==item['gzip_bytes'] and digest(packed)==item['gzip_sha256']
        raw=out/(item['name']+'.sqlite')
        with gzip.open(packed,'rb') as src,raw.open('xb') as dest:shutil.copyfileobj(src,dest,4*1024*1024)
        assert raw.stat().st_size==item['original_bytes'] and digest(raw)==item['original_sha256']
        packed.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--chunks-dir',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    reconstruct(a.manifest,a.manifest_sha256,a.chunks_dir,a.out)
    print(json.dumps({'status':'pass','reconstructed':2}))
