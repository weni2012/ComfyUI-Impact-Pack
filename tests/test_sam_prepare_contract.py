"""Validate SAM path/deduplication contracts without importing or loading models."""
import ast
import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


class SamPrepareContract(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "modules/impact/impact_server.py"
        node = next(item for item in ast.parse(path.read_text()).body
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == "sam_prepare")
        node.decorator_list = []
        self.calls = []
        owner = self

        class Lock:
            def __enter__(self): return self
            def __exit__(self, *args): return False

        class Thread:
            def __init__(self, target, args): self.target, self.args = target, args
            def start(self): owner.calls.append(self.args)  # Never execute model code.

        def annotated(filename):
            for kind in ("input", "output", "temp"):
                suffix = " [" + kind + "]"
                if filename.endswith(suffix): return filename[:-len(suffix)], "/virtual/" + kind
            return filename, None

        def checked(filename):
            name, root = annotated(filename)
            if root is None or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid media path")
            return str(Path(root) / name)

        self.folders = SimpleNamespace(
            get_full_path=lambda category, name: "/virtual/sams/" + name,
            annotated_filepath=annotated,
            get_directory_by_type=lambda kind: "/virtual/" + kind if kind in ("input", "output", "temp") else None,
            get_annotated_filepath=checked,
        )
        def forbidden_model(*args): raise AssertionError("A contract test must never load SAM")
        self.namespace = {
            "os": os, "logging": logging, "sam_predictor": None, "last_prepare_data": None,
            "sam_lock": Lock(), "folder_paths": self.folders,
            "threading": SimpleNamespace(Thread=Thread), "async_prepare_sam": forbidden_model,
            "impact": SimpleNamespace(config=SimpleNamespace(get_config=lambda: {"sam_editor_model": "fixture.pth"})),
            "web": SimpleNamespace(Response=lambda status: SimpleNamespace(status=status)),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), self.namespace)
        self.valid = {"sam_model_name": "auto", "filename": "frame.png", "type": "output", "subfolder": "nested"}

    def request(self, payload):
        async def json_data(): return dict(payload)
        return asyncio.run(self.namespace["sam_prepare"](SimpleNamespace(json=json_data))).status

    def test_valid_media_uses_annotation_and_repeated_request_is_deduplicated(self):
        self.assertEqual(self.request(self.valid), 200)
        self.assertEqual(self.calls, [("/virtual/sams/fixture.pth", "nested/frame.png [output]")])
        self.assertEqual(self.request(self.valid), 200)
        self.assertEqual(len(self.calls), 1)

    def test_rejected_paths_do_not_become_successful_cached_requests(self):
        self.assertEqual(self.request(self.valid), 200)
        for change in ({"type": "unknown"}, {"filename": "../outside.png"}, {"subfolder": "../../outside"}):
            with self.subTest(change=change):
                invalid = {**self.valid, **change}
                self.assertEqual(self.request(invalid), 400)
                self.assertEqual(self.request(invalid), 400)
        self.assertEqual(self.request(self.valid), 200)
        self.assertEqual(len(self.calls), 1)

    def test_existing_annotation_is_preserved(self):
        self.assertEqual(self.request({**self.valid, "filename": "other.png [input]"}), 200)
        self.assertEqual(self.calls[0][1], "other.png [input]")

    def test_missing_model_is_rejected_on_every_retry(self):
        self.folders.get_full_path = lambda category, name: None
        self.assertEqual(self.request(self.valid), 400)
        self.assertEqual(self.request(self.valid), 400)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
