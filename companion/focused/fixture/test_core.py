"""Independent synthetic tests of genealogy and strict temporal aggregation."""
import math
from ancestry import encode
from parent_control import compute,ids,ROLES
import feature_adapter as features


def naive(rows,parents,target):
    result={}
    for role,parent in zip(ROLES,ids(target['horse'],parents)):
        qualifying=[]
        for r in rows:
            if r['split']!='train' or r['horse']==target['horse'] or r['surface']!=target['surface']:continue
            if target['split']=='train' and r['date']>=target['date']:continue
            if parent and ids(r['horse'],parents)[ROLES.index(role)]==parent:qualifying.append(r)
        n=len(qualifying);s=sum(r['label'] for r in qualifying)
        result[role+'_available']=int(bool(parent));result[role+'_prior_count']=n
        result[role+'_prior_rate']=(s+4.0)/(n+20)
    return result


def main():
    assert [len(features.feature_columns(s)) for s in features.SETTINGS]==[43,112,112,52]
    graphcats=features.categorical_columns(features.feature_columns('Gancestor'))
    assert 'embedding_status' in graphcats and 'graph_cluster' in graphcats
    assert not set(features.P_COLUMNS)&set(features.categorical_columns(features.feature_columns('P')))
    rows_cat=[{'split':'train','embedding_status':'fallback','graph_cluster':'0'} for _ in range(5)]
    maps=features.fit_category_maps(rows_cat,('embedding_status','graph_cluster'))
    assert features.original.encode_category('direct',maps['embedding_status'])==1
    assert features.original.encode_category('missing',maps['embedding_status'])==1
    assert features.original.encode_category('',maps['embedding_status'])==0
    rare=features.fit_category_maps(rows_cat+[{'split':'train','embedding_status':'direct','graph_cluster':'7'}],('embedding_status','graph_cluster'))
    assert features.original.encode_category('direct',rare['embedding_status'])==2
    centers=[(0.0,0.0)]*64
    vectors={'h':(99,99),'a':(1,3),'b':(3,1),'c':(8,8)}
    p={('h','sire'):'a',('h','dam'):'b',('a','sire'):'c',('b','sire'):'c',('c','dam'):'h',
       ('x','sire'):'u',('u','sire'):'z',('z','dam'):'c',('dup','sire'):'a',('dup','dam'):'a'}
    one=encode('h',p,vectors,centers,2)
    assert one['vector']==(2.0,2.0) and one['fallback_depth']==1 and one['ancestor_count']==2 and one['graph_cluster']=='0'
    dup=encode('dup',p,vectors,centers,2);assert dup['vector']==(1,3) and dup['ancestor_count']==1
    three=encode('x',p,vectors,centers,2);assert three['fallback_depth']==3 and three['vector']==(8,8)
    missing=encode('unknown',p,vectors,centers,2);assert missing['embedding_status']=='missing' and missing['fallback_depth'] is None and missing['graph_cluster']=='MISSING' and missing['vector']==(0.,0.)
    cyclic=encode('h',{('h','sire'):'h'},vectors,centers,2);assert cyclic['embedding_status']=='missing'
    pedigree={('a','sire'):'S',('b','sire'):'S',('c','sire'):'S',('d','sire'):'S',
              ('a','dam'):'D',('b','dam'):'D',('c','dam'):'D',('D','sire'):'DS',('e','sire'):'UNSEEN'}
    def r(key,horse,date,split,label,surface='turf'):
        return dict(key=(key,horse,1),horse=horse,date=date,split=split,label=label,surface=surface)
    rows=[r('r1','a','20100101','train',1),r('r2','b','20100101','train',1),
          r('r3','a','20100102','train',0),r('r4','c','20100102','train',1),
          r('r5','d','20100103','train',0,'dirt'),r('r6','e','20100103','train',0),
          r('r7','a','20220101','validation',None),r('r8','c','20230101','validation',None)]
    actual,state=compute(rows,pedigree)
    for row in rows:
        expected=naive(rows,pedigree,row)
        for k,v in expected.items():assert math.isclose(actual[row['key']][k],v,abs_tol=1e-15),(row,k,v,actual[row['key']][k])
    assert actual[rows[0]['key']]['sire_prior_count']==0 # same-day exclusion
    assert actual[rows[2]['key']]['sire_prior_count']==1 # focal a excluded, b retained
    assert actual[rows[-2]['key']]['sire_prior_count']==2 # a excluded in frozen map
    assert actual[rows[4]['key']]['sire_prior_count']==0 # separate surface
    assert actual[rows[5]['key']]['dam_available']==0 and actual[rows[5]['key']]['sire_available']==1
    assert actual[rows[5]['key']]['sire_prior_count']==0 and actual[rows[5]['key']]['sire_prior_rate']==0.2
    assert state['total'][('sire','S','turf')]==(4,3)
    _,train_state=compute([r for r in rows if r['split']=='train'],pedigree)
    novel=r('r9','novel','20230102','validation',1)
    extended_pedigree=pedigree|{('novel','sire'):'NOVEL'}
    changed=[dict(row,label=1 if row['split']=='validation' else row['label']) for row in rows]+[novel]
    _,validation_state=compute(changed,extended_pedigree)
    assert train_state==validation_state, 'validation lookup mutated the frozen training maps'
    print('PASS independent ancestry and temporal P fixtures')


if __name__=='__main__':main()
