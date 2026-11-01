"""Small, local-only synthetic staging equivalence checks."""
from pathlib import Path
import csv,gzip,sqlite3,tempfile
from remote_workspace import surrogate,deterministic_gzip,digest


def main():
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp);pilot=p/'pilot.csv';pilot.write_text('child,parent,relation\na,s,sire\na,d,dam\n',encoding='utf-8')
        target=p/'extra.csv.gz'
        with gzip.open(target,'wt',encoding='utf-8',newline='') as f:f.write('child,parent,relation\na,s,sire\nd,ds,sire\n')
        dates=p/'dates.csv.gz'
        with gzip.open(dates,'wt',encoding='utf-8',newline='') as f:f.write('race_id,race_date\nr1,20100101\nr2,20231231\n')
        info=surrogate(pilot,target,dates,p/'work');assert info=={'relationship_keys':3,'races':2}
        db=sqlite3.connect(p/'work/local_jra_worker/data/raw_jbis_laptop.db')
        assert db.execute('select count(*) from pedigree_edges').fetchone()[0]==3;db.close()
        a=p/'a.gz';b=p/'b.gz';deterministic_gzip(pilot,a);deterministic_gzip(pilot,b)
        assert digest(a)==digest(b) and digest(a,True)==digest(pilot)
        try:surrogate(pilot,target,dates,p/'work')
        except AssertionError:pass
        else:raise AssertionError('surrogate overwrite allowed')
    print('PASS private staging surrogate and deterministic gzip fixture')


if __name__=='__main__':main()
