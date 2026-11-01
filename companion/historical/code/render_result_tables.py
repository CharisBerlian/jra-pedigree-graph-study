"""Render included aggregate records as Markdown, without private data."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    'model_summaries_extended.csv': ['family','surface','setting','entry_average_precision_mean','race_mean_hit_at_3_mean'],
    'exploratory_history_contrasts.csv': ['family','surface','history','contrast','delta','ci_low','ci_high'],
    'exploratory_history_interactions.csv': ['family','surface','history','contrast','difference_of_deltas','ci_low','ci_high'],
    'history_subgroup_counts.csv': ['surface','history','entries','positives','prevalence'],
    'graph_coverage.csv': ['split','surface','entries','direct','parent','generation2','generation3','missing'],
    'b0_metrics_corrected.csv': ['surface','expected_random_hit_at_3','race_mean_hit_at_3','ndcg_binary_all_races'],
}

output = ['# Aggregate result tables']
for name, fields in TABLES.items():
    with (ROOT/'tables'/name).open(newline='',encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    assert records and set(fields) <= records[0].keys()
    output += ['\n## '+name, '\n| '+' | '.join(fields)+' |','|'+'---|'*len(fields)]
    for record in records:
        values = []
        for field in fields:
            value = record[field]
            try:
                if '.' in value or 'e-' in value:
                    value = f'{float(value):.6f}'
            except ValueError:
                pass
            values.append(value)
        output.append('| '+' | '.join(values)+' |')
(ROOT/'RESULT_TABLES.md').write_text('\n'.join(output)+'\n',encoding='utf-8')
print('Wrote', ROOT/'RESULT_TABLES.md')
