"""Synthetic frozen-map application, including novel relatives and surfaces."""
from pathlib import Path
import hashlib,json,math,tempfile
from parent_control import compute
from apply_frozen_parent import load,apply
from test_core import naive


def main():
    parents={('a','sire'):'S',('b','sire'):'S',('b','dam'):'D',('a','dam'):'D',('D','sire'):'DS',('new','sire'):'U'}
    def r(rid,h,date,split,label,surface='turf'):
        return {'key':(rid,h,'1'),'horse':h,'date':date,'split':split,'label':label,'surface':surface}
    rows=[r('r1','a','20100101','train',1),r('r2','b','20100101','train',0),
          r('r3','a','20100102','train',0),r('r4','b','20100103','train',1,'dirt'),
          r('r5','a','20220101','validation',None),r('r6','new','20231231','validation',None)]
    features,state=compute(rows,parents)
    serial={'cutoff':'2021-12-31','surface_specific':True,'prior':0.2,'strength':20,
            'total':[[*k,*v] for k,v in sorted(state['total'].items())],
            'byhorse':[[*k[0],k[1],*v] for k,v in sorted(state['byhorse'].items())]}
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'maps.json';p.write_text(json.dumps(serial));sha=hashlib.sha256(p.read_bytes()).hexdigest();maps=load(p,sha)
        for row in rows:
            if row['split']=='validation':
                got=apply(row['horse'],row['surface'],parents,maps);expected=naive(rows,parents,row)
                assert got==features[row['key']]
                assert all(math.isclose(got[k],v,abs_tol=1e-15) for k,v in expected.items())
        before=(dict(maps[0]),dict(maps[1]));apply('novel','turf',parents,maps);assert before==(maps[0],maps[1])
        try:load(p,'0'*64)
        except AssertionError:pass
        else:raise AssertionError('unverified map accepted')
    print('PASS frozen P map against independent naive reference and unknown parent')


if __name__=='__main__':main()
