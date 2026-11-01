"""Independent AP/ranking and duplicate-race bootstrap fixtures."""
import math
import numpy as np
from metrics import prepare,paired,historical


def naive_ap(labels,scores):
    ordered=sorted(zip(scores,labels),reverse=True);positives=sum(labels);seen=0;tp=0;ap=0;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        group=sum(y for _,y in ordered[i:j]);seen+=j-i;tp+=group
        ap+=group/positives*tp/seen;i=j
    return ap


def main():
    rows=[{'race_id':f'r{r}','horse_id_netkeiba':f'h{r}{h}','horse_number':h,
           'field_size':3,'label':int(h in ((1,2) if r==1 else (2,3)))} for r in (1,2) for h in (1,2,3)]
    scores=[.8,.8,.2,.9,.4,.4];ap,hits=prepare(rows,scores)
    weights=np.array([2,1]);score_copy=scores[:3]*2+scores[3:];labels=[x['label'] for x in rows]
    label_copy=labels[:3]*2+labels[3:]
    assert math.isclose(historical._weighted_ap(ap,weights),naive_ap(label_copy,score_copy),abs_tol=1e-15)
    assert list(hits)==[2/3,2/3]
    tie=[{'race_id':'tie','horse_id_netkeiba':f'h{x}','horse_number':x,'field_size':4,'label':int(x==4)} for x in (4,2,1,3)]
    _,tie_hits=prepare(tie,[.5]*4);assert tie_hits[0]==0 # number 4 loses the score tie
    result,contrasts=paired(rows,{x:[scores,scores,scores] for x in ('H','Gmixed','Gancestor','P')},20,20261029)
    assert len(result)==4 and len(contrasts)==6 and all(x['point']==0 for x in contrasts.values())
    assert contrasts['Gancestor-H/hit']['additional_hits_per_100_races']==[0,0,0]
    for invalid in (scores[:-1],scores[:-1]+[float('nan')],scores[:-1]+[1.1]):
        try:prepare(rows,invalid)
        except AssertionError:pass
        else:raise AssertionError('invalid scores passed')
    broken=[dict(x) for x in rows];broken[0]['field_size']=4
    try:prepare(broken,scores)
    except AssertionError:pass
    else:raise AssertionError('incomplete roster passed')
    print('PASS full-roster rank, tied AP, duplicate races, intervals and score rejection')


if __name__=='__main__':main()
