#!/usr/bin/env python3
from pathlib import Path
import json
import tempfile

from cache_utils import atomic_json, canonical_json, content_key, complete_manifest, validate_complete


def main():
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert content_key({"a": 1, "b": 2}) == content_key({"b": 2, "a": 1})
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw); artifact=root/"x"; artifact.write_text("stable\n")
        identity={"upstream":"abc","config":{"k":20}}
        first=complete_manifest(root/"manifest.json",identity,[artifact],{"created":"incidental-one"})
        second=complete_manifest(root/"manifest.json",identity,[artifact],{"created":"incidental-two"})
        assert first["content_key"] == second["content_key"]
        assert validate_complete(root/"manifest.json") is not None
        artifact.write_text("changed\n")
        assert validate_complete(root/"manifest.json") is None
    print("shared cache tests: OK")


if __name__ == "__main__": main()
