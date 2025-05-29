import json
import gzip
from pathlib import Path


def read_json_gz(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)
