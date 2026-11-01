"""Generate but do not submit private, exact-byte pedigree transport jobs."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/'private_data/pedigree_transport'
OUT=BASE/'job_sources'
MANIFEST_HASH='509413e101ab97c68220bb4ac29583f0fecab409c9ed97da07a035ab99e899a1'

def sha(b):return hashlib.sha256(b).hexdigest()

raw=(BASE/'manifest.json').read_bytes();assert sha(raw)==MANIFEST_HASH
manifest=json.loads(raw);assert manifest['format']=='gzip-mtime-0-chunks-v1'
assert not OUT.exists();OUT.mkdir()
records=[]
for item in manifest['files']:
    for chunk in item['chunks']:
        name=chunk['name'];payload=(BASE/'chunks'/name).read_bytes()
        assert len(payload)==chunk['length'] and sha(payload)==chunk['sha256']
        encoded=base64.b64encode(payload).decode('ascii')
        source=f'''from pathlib import Path
import base64,hashlib,json
payload={encoded!r}
data=base64.b64decode(payload,validate=True)
assert len(data)=={len(payload)} and hashlib.sha256(data).hexdigest()=={chunk['sha256']!r}
out=Path('/kaggle/working/pedigree-chunk');out.mkdir(parents=True,exist_ok=False)
dest=out/{name!r};dest.write_bytes(data)
assert dest.stat().st_size=={len(payload)} and hashlib.sha256(dest.read_bytes()).hexdigest()=={chunk['sha256']!r}
(out/'PASS.json').write_text(json.dumps({{'status':'pass','name':{name!r},'bytes':len(data),'sha256':{chunk['sha256']!r},'manifest_sha256':{MANIFEST_HASH!r}}},sort_keys=True)+'\\n')
'''
        data=source.encode('utf-8');tree=ast.parse(source);compile(source,name,'exec')
        assignments=[x for x in tree.body if isinstance(x,ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='payload']
        assert len(assignments)==1 and base64.b64decode(ast.literal_eval(assignments[0].value),validate=True)==payload
        assert len(data)<1_000_000
        script=OUT/(name+'.py');script.write_bytes(data)
        records.append({'name':name,'offset':chunk['offset'],'bytes':len(payload),'payload_sha256':chunk['sha256'],
                        'source_name':script.name,'source_bytes':len(data),'source_sha256':sha(data)})
assert len(records)==12 and len({r['name'] for r in records})==12
report={'status':'prepared_not_submitted','manifest_sha256':MANIFEST_HASH,'jobs':records}
(OUT/'inventory.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'status':report['status'],'jobs':len(records),'largest_source_bytes':max(r['source_bytes'] for r in records),
                  'inventory_sha256':sha((OUT/'inventory.json').read_bytes())}))
