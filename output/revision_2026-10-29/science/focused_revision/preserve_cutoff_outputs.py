"""Verify already completed superseded assembly artifacts; never fit or launch."""
from pathlib import Path
import gzip
import hashlib
import json

HERE=Path(__file__).resolve().parent
SCIENCE=HERE.parent
ROOT=SCIENCE/'private_inputs/harvest/assembly_jobs'


def sha(path, compressed=False):
    h=hashlib.sha256();op=gzip.open if compressed else open
    with op(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    results=[]
    for seed in (0,1,2):
        folder=ROOT/f'seed{seed}'
        marker=folder/'ASSEMBLY_PASS.json';m=json.loads(marker.read_text())
        assert m['status']=='pass' and m['seed']==seed
        assert m['protocol_sha256']=='47fa77bce8702c4787538f68717b4c4d486576dd8e991ffe6748d703d5cce6ae'
        # The frozen remote wrapper reads UTF-8 text with universal newlines,
        # then writes that text on Linux. The Windows file bytes use CRLF.
        normalized=(SCIENCE/'cohort_fingerprint.json').read_text(encoding='utf-8').encode('utf-8')
        assert m['fingerprint_sha256']==hashlib.sha256(normalized).hexdigest()
        for regime in ('old','new'):
            meta=folder/f'{regime}-seed{seed}.csv.json';data=folder/f'{regime}-seed{seed}.csv.gz'
            a=json.loads(meta.read_text());assert a['regime']==regime and a['seed']==seed
            assert a['rows']==266812 and a['splits']==['train','validation']
            assert sha(data,compressed=True)==a['data_sha256']
            results.append({'seed':seed,'regime':regime,'rows':a['rows'],
                            'downloaded_gzip_sha256':sha(data),'decompressed_payload_sha256':a['data_sha256'],
                            'sidecar_sha256':sha(meta),'marker_sha256':sha(marker)})
    dest=HERE/'superseded_cutoff_preservation.json'
    assert not dest.exists()
    dest.write_text(json.dumps({'status':'preserved_and_verified','superseded':True,'artifacts':results},indent=2)+'\n',encoding='utf-8')
    print('PASS preserved 3 markers, 6 sidecars and 6 exact decompressed assembly payloads')


if __name__=='__main__':main()
