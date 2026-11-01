"""Prepare deterministic byte-preserving chunks of the two pedigree-only DBs.

This reads raw files and the existing train/validation provenance manifest only;
it never opens a database or reads a test observation. It submits no jobs.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=HERE/'private_data/pedigree_transport'
CHUNK=600_000
sources=[('first',ROOT/'local_jra_worker/data/raw_jbis_laptop.db','local_jra_worker\\data\\raw_jbis_laptop.db'),
         ('second',ROOT/'tmp/pedigree_patch.db','tmp\\pedigree_patch.db')]
expected=json.loads((HERE/'private_data/parent_manifest.json').read_text())['source_databases_sha256']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

assert not OUT.exists()
OUT.mkdir(parents=True)
records=[]
for label,source,key in sources:
    original_hash=sha(source);assert original_hash==expected[key]
    compressed=OUT/(label+'.gz')
    with source.open('rb') as src, compressed.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=9,mtime=0) as dest:
            shutil.copyfileobj(src,dest,4*1024*1024)
    chunks=[];offset=0
    with compressed.open('rb') as src:
        while block:=src.read(CHUNK):
            name=f'{label}-{len(chunks):03d}.bin';path=OUT/name;path.write_bytes(block)
            chunks.append({'name':name,'offset':offset,'length':len(block),'sha256':hashlib.sha256(block).hexdigest()})
            offset+=len(block)
    assert offset==compressed.stat().st_size
    h=hashlib.sha256();size=0
    with gzip.open(compressed,'rb') as src:
        for block in iter(lambda:src.read(4*1024*1024),b''):
            h.update(block);size+=len(block)
    assert h.hexdigest()==original_hash and size==source.stat().st_size
    records.append({'name':label,'original_bytes':size,'original_sha256':original_hash,
                    'gzip_bytes':offset,'gzip_sha256':sha(compressed),'chunks':chunks})
manifest={'format':'gzip-mtime-0-chunks-v1','chunk_bytes_max':CHUNK,'test_observations_read':False,
          'database_queries_executed':False,'files':records}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'status':'pass','files':[{'name':r['name'],'original_bytes':r['original_bytes'],
    'gzip_bytes':r['gzip_bytes'],'gzip_sha256':r['gzip_sha256'],'chunks':len(r['chunks'])} for r in records],
    'manifest_sha256':sha(OUT/'manifest.json')}))
