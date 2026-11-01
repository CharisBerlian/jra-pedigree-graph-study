"""Summarize fixed-grid validation round limits from an audited model freeze."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIVATE=HERE/'private_data'
source=PRIVATE/'FOCUSED_MODEL_FREEZE_V1.json'
output=HERE/'CAP_DIAGNOSTICS_V1.json'
assert not output.exists()
raw=source.read_bytes();m=json.loads(raw);spec=json.loads((HERE/'EXPERIMENT_SPEC.json').read_text())
cap=spec['tuning']['max_rounds'];assert cap==1200
assert m['status']=='frozen_for_root_review' and m['tuning_cells']==24 and m['candidate_count']==96 and m['model_count']==72
cells=[];refits=0
for item in m['models']:
    evid=item['candidate_evidence'];assert len(evid)==4
    selected=next(e for e in evid if e['candidate_id']==item['selected_candidate'])
    assert selected['best_round']==item['selected_round']
    for refit in item['refits']:
        folder=PRIVATE/'model_harvest'/Path(refit['relative_dir'])
        cfg=json.loads((folder/'config.json').read_text());assert cfg['selected_round']==item['selected_round']
        refits+=1
    cells.append({'family':item['family'],'surface':item['surface'],'arm':item['arm'],
        'candidate_curves_at_cap':sum(bool(e['hit_cap']) for e in evid),
        'selected_curve_at_cap':bool(selected['hit_cap']),'selected_best_round':item['selected_round']})
assert len(cells)==24 and refits==72
out={'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),'scope':'validation_only','max_rounds':cap,
     'candidate_curves_at_cap':sum(c['candidate_curves_at_cap'] for c in cells),
     'candidate_count':96,'selected_curves_at_cap':sum(c['selected_curve_at_cap'] for c in cells),
     'selected_count':24,'selected_best_round_at_cap':sum(c['selected_best_round']==cap for c in cells),
     'refit_count':72,'refit_selected_round_at_cap':sum(3 for c in cells if c['selected_best_round']==cap),
     'max_selected_best_round':max(c['selected_best_round'] for c in cells),'cells':cells,
     'interpretation':'hit_cap describes a candidate evaluation curve reaching the round limit, not the selected best round.'}
assert out['candidate_curves_at_cap']==m['candidate_cap_count']
output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('cells','interpretation')}))
