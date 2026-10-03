#!/usr/bin/env python3
"""Run the frozen blocker unchanged while enabling encoder progress bars."""

import importlib.util
import pathlib
import sys

from sentence_transformers import SentenceTransformer


_encode = SentenceTransformer.encode


def _encode_with_progress(self, *args, **kwargs):
    kwargs["show_progress_bar"] = True
    return _encode(self, *args, **kwargs)


SentenceTransformer.encode = _encode_with_progress

path = pathlib.Path(__file__).with_name("blocker_v1.py")
spec = importlib.util.spec_from_file_location("blocker_v1_frozen", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
sys.exit(module.main())
