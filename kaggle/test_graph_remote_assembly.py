"""No-fit checks for compact graph tables and remote assembly."""
import csv
import gzip
import json
import tempfile
from pathlib import Path

from assemble_prediction_seed import (GRAPH_COLUMNS, assemble,
                                       load_graph_table, verify_graph_metadata)
from build_graph_feature_tables import write_table


def _base_row(race, horse, number):
    row = {"race_id": race, "horse_id_netkeiba": horse, "horse_number": str(number),
           "split": "train", "label": "0", "venue": "V"}
    row.update({column: "old" for column in GRAPH_COLUMNS})
    return row


def _write_base(path, rows):
    columns = tuple(rows[0])
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(
        json.dumps({"data_sha256": _sha(path), "splits": ["train", "validation"]}, indent=2),
        encoding="utf-8")


def _sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        base = root / "base.csv.gz"
        _write_base(base, [_base_row("r1", "h1", 1), _base_row("r2", "h1", 1)])
        graph = root / "graph.csv.gz"
        row = {"horse_id_netkeiba": "h1", "embedding_status": "direct",
               "fallback_depth": "0", "ancestor_count": "0",
               "embedding_missing": "0", "graph_cluster": "7"}
        row.update({f"embedding_{i}": str(i / 10) for i in range(64)})
        write_table(graph, [row], {"seed": 1})
        output = root / "assembled.csv.gz"
        result = assemble(base, graph, output)
        assert result["rows"] == 2
        assert result["base_non_graph_sha256"] == result["assembled_non_graph_sha256"]
        assert result["csv_stats"]["compressed"]["bytes"] == output.stat().st_size
        assert result["csv_stats"]["uncompressed"]["bytes"] > 0
        assert result["base_csv_stats"]["uncompressed"]["bytes"] > 0
        with gzip.open(output, "rt", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["graph_cluster"] for row in rows] == ["7", "7"]
        assert [row["venue"] for row in rows] == ["V", "V"]
        duplicate = root / "duplicate.csv.gz"
        write_table(duplicate, [row, row], {"seed": 1})
        try:
            load_graph_table(duplicate)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("duplicate graph keys were accepted")
        bad_meta = root / "bad.json"
        bad_meta.write_text(json.dumps({"seed": 2}), encoding="utf-8")
        try:
            verify_graph_metadata(bad_meta, expected_seed=1)
        except ValueError as error:
            assert "seed" in str(error)
        else:
            raise AssertionError("mismatched graph metadata seed was accepted")
    print("PASS: compact graph table hashes, keyed assembly, non-graph preservation, and duplicate rejection")


if __name__ == "__main__":
    check()
