# SPDX-License-Identifier: Apache-2.0
"""Metadata-only tests of real Elephant source under raising dependency fakes.

mlx-lm@0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d ignores layer_types;
qwen3_5.TextModelArgs defaults interval to 4 and DecoderLayer uses (i+1)%I.
No installed dependencies, model, arithmetic fakes, assets or downloads needed.
Run standalone with Python 3.11–3.13 -I -S -B (never root pytest/conftest).
"""

import builtins
import copy
import importlib
import importlib.abc
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import types
import unittest
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/elephant_contract.json").read_text())
LINEAR, FULL = "linear_attention", "full_attention"
DEPENDENCIES = (
    "mlx", "mlx_audio", "mlx_lm", "mlx_vlm", "torch", "transformers",
    "numpy", "huggingface_hub", "safetensors",
)
RUNTIME_IMPORTS = (
    "omlx.models.decision", "omlx.utils.model_loading", "omlx.utils.image",
    "omlx.patches", "omlx.model_discovery", "omlx.decision_discovery",
    "omlx.server", "omlx.engine.decision",
)


class SafetyViolation(AssertionError):
    pass


class ConstructorReached(RuntimeError):
    """A valid contract may reach this sentinel, never a real constructor."""


class RequestError(ValueError):
    pass


class ImportTrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in DEPENDENCIES or (
            fullname.startswith("omlx.") and fullname != "omlx.models.elephant"
        ):
            raise SafetyViolation(f"Real dependency import forbidden: {fullname}")
        return None


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.path = Path(self.stack.enter_context(
            tempfile.TemporaryDirectory(prefix="elephant-schedule-")
        ))
        self.events = []
        modules = {}

        def module(name, **attrs):
            obj = types.ModuleType(name)
            obj.__dict__.update(attrs)
            modules[name] = obj
            return obj

        for name, subdir in (("omlx", "omlx"), ("omlx.models", "omlx/models")):
            module(name, __path__=[str(ROOT / subdir)])
        module("mlx", __path__=[])
        module("mlx.core", array=self.forbidden, eval=self.forbidden,
               synchronize=self.forbidden, float32="forbidden-dtype")
        events, forbidden = self.events, self.forbidden

        class RaisingBackbone:
            def __init__(self, *args, **kwargs):
                events.append("constructor")
                raise ConstructorReached("Stopped before any backbone construction")

            load = forbidden
            patch = forbidden
            output_rows = forbidden

        module("omlx.models.decision", DecisionBackbone=RaisingBackbone,
               DecisionRequestError=RequestError,
               DecisionContextLengthError=type("ContextError", (ValueError,), {}))
        # Restore even newly imported modules, including the real source under test.
        self.stack.enter_context(patch.dict(sys.modules, modules))
        sys.modules.pop("omlx.models.elephant", None)
        self.trap = ImportTrap()
        self.stack.enter_context(patch.object(sys, "meta_path", [self.trap, *sys.meta_path]))
        for obj, attr in (
            (socket.socket, "connect"), (socket, "create_connection"),
            (urllib.request, "urlopen"), (subprocess, "Popen"),
            (subprocess, "run"), (os, "system"),
        ):
            self.stack.enter_context(patch.object(obj, attr, self.forbidden))
        # Traps are active BEFORE importing the sole runtime source under test.
        self.runtime = importlib.import_module("omlx.models.elephant")
        self.assertIs(self.runtime.mx, modules["mlx.core"])
        self.assertIs(self.runtime.DecisionRequestError, RequestError)
        self.model = self.runtime.ElephantModel(str(self.path))
        self.config = copy.deepcopy(FIXTURE["config"])
        self.write_contract()

    def forbidden(self, *args, **kwargs):
        self.events.append("forbidden-operation")
        raise SafetyViolation("Numerical, loader, patch, network or subprocess forbidden")

    def write_contract(self):
        (self.path / "config.json").write_text(json.dumps(self.config))
        (self.path / "readout.json").write_text(json.dumps(FIXTURE["readout"]))

    @contextmanager
    def metadata_reads_only(self):
        allowed = {self.path / "config.json", self.path / "readout.json"}

        def guard(original):
            def checked(file, mode="r", *args, **kwargs):
                if not isinstance(file, (str, bytes, os.PathLike)) or (
                    Path(os.fsdecode(file)).absolute() not in allowed
                    or mode not in ("r", "rt")
                ):
                    self.events.append("asset-open")
                    raise SafetyViolation(f"Asset or non-metadata open forbidden: {file}")
                return original(file, mode, *args, **kwargs)
            return checked

        with patch.object(builtins, "open", guard(builtins.open)), \
                patch.object(io, "open", guard(io.open)):
            yield

    def assert_valid(self):
        self.write_contract()
        original = copy.deepcopy(self.config["text_config"])
        config_bytes = (self.path / "config.json").read_bytes()
        with self.metadata_reads_only():
            returned = self.model._contract()
        self.assertEqual(returned, original)
        self.assertEqual(self.config["text_config"], original)
        self.assertEqual((self.path / "config.json").read_bytes(), config_bytes)
        self.assertEqual(self.events, [])
        return returned

    def assert_invalid(self):
        self.write_contract()
        for method in ("_contract", "load"):
            self.events.clear()
            with self.subTest(entry=method), self.metadata_reads_only():
                with self.assertRaises(RequestError):
                    getattr(self.model, method)()
                self.assertEqual(self.events, [], "Rejected before constructor/load/patch/assets")
                self.assertIsNone(self.model.backbone)
                self.assertIsNone(self.model.slot_rows)
                self.assertEqual(self.model.slot_ids, ())

    def schedule(self, depth, interval):
        self.config["text_config"].update(
            num_hidden_layers=depth, full_attention_interval=interval,
            layer_types=[FULL if (i + 1) % interval == 0 else LINEAR for i in range(depth)],
        )

    def test_candidate_explicit_and_default_interval(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                self.config = copy.deepcopy(FIXTURE["config"])
                if explicit:
                    self.config["text_config"]["full_attention_interval"] = 4
                text = self.assert_valid()
                self.assertEqual(text["num_hidden_layers"], 32)
                self.assertEqual(text["layer_types"].count(LINEAR), 24)
                self.assertEqual(text["layer_types"].count(FULL), 8)
                self.assertEqual([i for i, k in enumerate(text["layer_types"]) if k == FULL],
                                 [3, 7, 11, 15, 19, 23, 27, 31])
                self.assertEqual("full_attention_interval" in text, explicit)

    def test_default_four_layer_contract(self):
        self.schedule(4, 4)
        del self.config["text_config"]["full_attention_interval"]
        self.assertNotIn("full_attention_interval", self.assert_valid())

    def test_minimum_and_interval_equal_to_depth(self):
        for depth in (2, 3, 9, 33):
            with self.subTest(depth=depth):
                self.schedule(depth, depth)
                text = self.assert_valid()
                self.assertEqual(text["layer_types"], [LINEAR] * (depth - 1) + [FULL])

    def test_partial_period_without_terminal_full(self):
        self.schedule(5, 3)
        self.assertEqual(self.assert_valid()["layer_types"],
                         [LINEAR, LINEAR, FULL, LINEAR, LINEAR])

    def test_valid_load_reaches_only_constructor_sentinel(self):
        with self.metadata_reads_only(), self.assertRaises(ConstructorReached):
            self.model.load()
        self.assertEqual(self.events, ["constructor"])
        self.assertIsNone(self.model.backbone)

    def test_missing_required_metadata(self):
        for key in ("num_hidden_layers", "layer_types"):
            with self.subTest(missing=key):
                self.config = copy.deepcopy(FIXTURE["config"])
                del self.config["text_config"][key]
                self.assert_invalid()

    def test_invalid_depth_types_and_ranges(self):
        for depth in (None, True, False, 32.0, 4.5, "32", [], {}, 0, -1, 1):
            with self.subTest(depth=depth):
                self.config["text_config"]["num_hidden_layers"] = depth
                self.assert_invalid()

    def test_invalid_interval_types_and_ranges(self):
        for interval in (None, True, False, 4.0, 2.5, "4", [], {}, 0, -2, 1, 33):
            with self.subTest(interval=interval):
                self.config["text_config"]["full_attention_interval"] = interval
                self.assert_invalid()

    def test_malformed_layer_lists(self):
        candidate = copy.deepcopy(FIXTURE["config"]["text_config"]["layer_types"])
        for kinds in (None, True, 32, "linear_attention", {}, [], candidate[:-1],
                      candidate + [LINEAR], [FULL] * 32, [LINEAR] * 32,
                      candidate[:-1] + ["unknown"], candidate[:-1] + [[]],
                      candidate[:-1] + [{}], candidate[:-1] + [None]):
            with self.subTest(kinds=kinds):
                self.config["text_config"]["layer_types"] = kinds
                self.assert_invalid()

    def test_absent_interval_uses_default_not_schedule(self):
        self.schedule(32, 2)
        del self.config["text_config"]["full_attention_interval"]
        self.assert_invalid()
        self.schedule(2, 2)
        del self.config["text_config"]["full_attention_interval"]
        self.assert_invalid()

    def test_explicit_interval_must_match_schedule(self):
        self.config["text_config"]["full_attention_interval"] = 2
        self.assert_invalid()

    def test_same_count_reordered_schedule(self):
        kinds = self.config["text_config"]["layer_types"]
        kinds[2], kinds[3] = kinds[3], kinds[2]
        self.assertEqual(kinds.count(FULL), 8)
        self.assertEqual(kinds.count(LINEAR), 24)
        self.assert_invalid()

    def test_nonperiodic_31_linear_one_full(self):
        self.config["text_config"]["layer_types"] = [LINEAR] * 31 + [FULL]
        self.assert_invalid()

    def test_close_precedes_invalid_contract(self):
        class Previous:
            def close(inner):
                self.events.append("close-previous")
        self.model.backbone = Previous()
        self.model.slot_rows = object()
        self.model.slot_ids = (1,)
        self.config["text_config"]["full_attention_interval"] = 0
        self.write_contract()
        with self.metadata_reads_only(), self.assertRaises(RequestError):
            self.model.load()
        self.assertEqual(self.events, ["close-previous"])
        self.assertIsNone(self.model.backbone)
        self.assertIsNone(self.model.slot_rows)
        self.assertEqual(self.model.slot_ids, ())

    def test_import_traps_independently_block_real_dependencies(self):
        for name in (*DEPENDENCIES, "mlx.core", *RUNTIME_IMPORTS):
            with self.subTest(name=name), patch.dict(sys.modules):
                for cached in list(sys.modules):
                    if cached == name or cached.startswith(name + "."):
                        sys.modules.pop(cached)
                with self.assertRaises(SafetyViolation):
                    importlib.import_module(name)
        self.assertEqual(self.events, [])

    def test_loader_numerical_network_and_subprocess_traps(self):
        backbone = self.runtime.DecisionBackbone
        controls = (
            lambda: backbone("synthetic-path"),
            lambda: backbone.load(), lambda: backbone.patch(),
            lambda: self.runtime.mx.array([1]), lambda: self.runtime.mx.eval(None),
            lambda: self.runtime.mx.synchronize(),
            lambda: urllib.request.urlopen("https://invalid.example"),
            lambda: socket.create_connection(("invalid.example", object())),
            lambda: subprocess.run(["forbidden"]),
            lambda: subprocess.Popen(["forbidden"]), lambda: os.system("forbidden"),
        )
        for index, control in enumerate(controls):
            self.events.clear()
            expected = ConstructorReached if index == 0 else SafetyViolation
            with self.subTest(control=index), self.assertRaises(expected):
                control()
            self.assertEqual(self.events, ["constructor" if index == 0 else "forbidden-operation"])
        self.events.clear()
        with socket.socket() as sock, self.assertRaises(SafetyViolation):
            sock.connect(object())
        self.assertEqual(self.events, ["forbidden-operation"])

    def test_asset_open_traps(self):
        asset = self.path / "synthetic.safetensors"  # Never created or read.
        for opener in (lambda p: builtins.open(p), lambda p: io.open(p),
                       lambda p: p.open(), lambda p: p.read_bytes()):
            self.events.clear()
            with self.subTest(opener=opener), self.metadata_reads_only():
                with self.assertRaises(SafetyViolation):
                    opener(asset)
            self.assertEqual(self.events, ["asset-open"])


if __name__ == "__main__":
    unittest.main()
