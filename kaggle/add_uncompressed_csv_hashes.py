"""Add transport-alias hashes to frozen gzip test-input sidecars."""

import gzip
import hashlib
import json
from pathlib import Path
import sys


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def update(folder):
    folder = Path(folder)
    for seed in (0, 1, 2):
        data = folder / f"test_seed{seed}.csv.gz"
        sidecar = folder / f"test_seed{seed}.csv.json"
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        with gzip.open(data, "rb") as stream:
            metadata["uncompressed_csv_sha256"] = sha256_stream(stream)
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: add_uncompressed_csv_hashes.py DATASET_FOLDER")
    update(sys.argv[1])
    print("sidecars_updated")
