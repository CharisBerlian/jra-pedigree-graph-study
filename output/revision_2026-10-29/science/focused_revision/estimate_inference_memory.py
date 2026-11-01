"""Synthetic retained-row memory estimate; does not open any test inputs."""
from pathlib import Path
import csv,io,json,sys,tracemalloc
import feature_adapter as features


def main():
    cols=list(dict.fromkeys(['race_id','horse_id_netkeiba','horse_number','split','surface','sire','dam','odds','popularity']+
                            list(features.feature_columns('Gmixed'))+list(features.P_COLUMNS)))
    count=5000;buffer=io.StringIO(newline='');w=csv.writer(buffer);w.writerow(cols)
    for i in range(count):w.writerow([f'{(i+1)*(j+17)%100000000:08d}' for j in range(len(cols))])
    text=buffer.getvalue();stream=io.StringIO(text);tracemalloc.start()
    rows=list(csv.DictReader(stream));current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();assert len(rows)==count
    per_row=current/count;projected=per_row*89137*6
    result={'scope':'synthetic CSV rows only; no train/test feature values or outcomes',
            'python':sys.version.split()[0],'synthetic_rows':count,'columns':len(cols),
            'traced_retained_bytes_per_row':per_row,'traced_read_peak_bytes':peak,
            'six_times_89137_retained_estimated_gib':projected/(1024**3),
            'limitations':'Synthetic string lengths and missingness differ; excludes roster/content digests, NumPy/Pandas frames, model memory, interpreter and OS overhead. Not a peak RSS or bound.',
            'prior_kaggle_graph_smoke_proc_memtotal_kib':32869292,
            'kaggle_memory_note':'Historical /proc/meminfo host MemTotal; not proof of the inference container cgroup limit or current allocation.'}
    out=Path(__file__).resolve().parent/'INFERENCE_MEMORY_ESTIMATE.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
