"""Apply approved one-substitution transport amendment to one private job."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent;PRIVATE=HERE/'private_data'
AMEND='a84dd39d0af49f08c45a3fd59eb898d9aee128a0aab2008bd7a7b772c5e46cb3'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    assert all(x.resolve().is_relative_to(PRIVATE.resolve()) for x in (a.source,a.out))
    old_record=a.source.with_suffix('.json');new_record=a.out.with_suffix('.json');assert not a.out.exists() and not new_record.exists()
    assert sha(HERE/'TRANSPORT_MKDIR_AMENDMENT.md')==AMEND
    record=json.loads(old_record.read_text());assert record['prefit_sha256']==sha(HERE/'PREFIT_MANIFEST_V4.json')
    source=a.source.read_bytes();assert hashlib.sha256(source).hexdigest()==record['script_sha256']
    old=b'inp.mkdir();bundle.mkdir()';new=b'inp.mkdir(parents=True);bundle.mkdir()'
    assert source.count(old)==1 and new not in source
    changed=source.replace(old,new);assert changed.replace(new,old)==source;compile(changed.decode('utf-8'),'<scratch-amendment>','exec')
    revised=dict(record,script_sha256=hashlib.sha256(changed).hexdigest(),source_bytes=len(changed),
       transport_amendment_sha256=AMEND,superseded_source_sha256=sha(a.source),superseded_record_sha256=sha(old_record))
    a.out.write_bytes(changed);new_record.write_text(json.dumps(revised,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'stage':record['stage'],'original_source_sha256':sha(a.source),'corrected_source_sha256':sha(a.out),
                      'original_record_sha256':sha(old_record),'corrected_record_sha256':sha(new_record),'amendment_sha256':AMEND,'bytes':len(changed)}))


if __name__=='__main__':main()
