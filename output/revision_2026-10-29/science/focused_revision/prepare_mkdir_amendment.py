"""Prepare one exact transport-only parent-job correction; never submit."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;PRIVATE=HERE/'private_data'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original=PRIVATE/'parent_prepare_v4.py';old_record=PRIVATE/'parent_prepare_v4.json'
    out=PRIVATE/'parent_prepare_v4_mkdir1.py';record=out.with_suffix('.json')
    assert not out.exists() and not record.exists()
    previous=json.loads(old_record.read_text());assert sha(original)==previous['script_sha256']
    source=original.read_bytes();old=b'inp.mkdir();bundle.mkdir()';new=b'inp.mkdir(parents=True);bundle.mkdir()'
    assert source.count(old)==1 and new not in source
    changed=source.replace(old,new);compile(changed.decode('utf-8'),'<mkdir-amendment>','exec')
    assert changed.replace(new,old)==source
    updated=dict(previous,script_sha256=hashlib.sha256(changed).hexdigest(),source_bytes=len(changed),
                 transport_amendment='create missing scratch parent directory',
                 superseded_source_sha256=sha(original),superseded_record_sha256=sha(old_record))
    out.write_bytes(changed);record.write_text(json.dumps(updated,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'source_sha256':sha(out),'record_sha256':sha(record),'bytes':len(changed),
                      'old_source_sha256':sha(original),'scientific_bytes_otherwise_identical':True}))


if __name__=='__main__':main()
