# SPDX-License-Identifier: Apache-2.0
"""Offline stdlib tests; never import the dependency-heavy discovery parent."""

import ast
import importlib
import importlib.abc
import json
from contextlib import ExitStack
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MOVED_NAMES = {
    "DecisionKind", "decision_kind", "CLEF_HEAD_WEIGHTS", "CLEF_HEAD_CONFIG",
    "D3_READOUT_WEIGHTS", "D3_DECISION_CONFIG", "D3_FORMAT_ID", "_OPENJEV_HELPER",
    "_is_d3_decision_config", "_model_name_hint", "_decode_hf_cache_model_id",
}


class DependencyTrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            "mlx", "mlx_audio", "mlx_lm", "mlx_vlm", "numpy", "torch",
            "transformers", "gguf", "huggingface_hub", "requests",
        }
        if fullname.split(".", 1)[0] in blocked or (
            fullname.startswith("omlx.") and fullname != "omlx.decision_discovery"
        ):
            raise AssertionError(f"Real dependency import forbidden: {fullname}")
        return None


def forbidden_operation(*args, **kwargs):
    raise AssertionError("Network, subprocess and model execution are forbidden")


class DecisionDiscoveryTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.path = Path(stack.enter_context(TemporaryDirectory()))
        package = ModuleType("omlx")
        package.__path__ = [str(ROOT / "omlx")]
        stack.enter_context(patch.dict(sys.modules, {"omlx": package}))
        sys.modules.pop("omlx.decision_discovery", None)
        stack.enter_context(patch.object(sys, "meta_path", [DependencyTrap(), *sys.meta_path]))
        stack.enter_context(patch.object(socket, "socket", forbidden_operation))
        stack.enter_context(patch.object(socket, "create_connection", forbidden_operation))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden_operation))
        self.discovery = importlib.import_module("omlx.decision_discovery")

    def write_json(self, name, value):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))

    def touch(self, name):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    def test_dependency_and_execution_traps_are_active(self):
        for name in (
            "mlx", "mlx_audio", "mlx_lm", "mlx_vlm", "torch", "numpy",
            "transformers", "gguf", "omlx.gguf", "omlx.models.clef",
            "omlx.utils.model_loading", "omlx.model_discovery",
        ):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                importlib.import_module(name)
        with self.assertRaises(AssertionError):
            socket.create_connection(("invalid.example", 443))
        with self.assertRaises(AssertionError):
            subprocess.Popen(["forbidden"])

    def test_clef_requires_both_files_and_precedes_other_families(self):
        self.touch("joint_head.safetensors")
        self.assertIsNone(self.discovery.decision_kind(self.path, {}))
        self.touch("joint_head_config.json")
        self.touch("readout.safetensors")
        self.write_json("decision_config.json", {"format_id": "d3-code-readout-v1"})
        self.assertEqual(self.discovery.decision_kind(self.path, None), "clef")
        self.assertEqual(self.discovery.decision_kind(self.path, []), "clef")
        (self.path / "joint_head.safetensors").unlink()
        self.assertEqual(self.discovery.decision_kind(self.path), "d3")
        (self.path / "readout.safetensors").unlink()
        self.assertIsNone(self.discovery.decision_kind(self.path, {}))

    def test_d3_requires_weights_and_valid_format_and_precedes_d1(self):
        config = {"model_type": "lfm3_vl", "architectures": ["D1Model"]}
        self.write_json("decision_config.json", {"format_id": "d3-code-readout-v1"})
        self.assertEqual(self.discovery.decision_kind(self.path, config), "d1")
        self.touch("readout.safetensors")
        self.assertEqual(self.discovery.decision_kind(self.path, config), "d3")
        for value in ({}, {"format_id": "wrong"}, [], None, "text", 3):
            with self.subTest(value=value):
                self.write_json("decision_config.json", value)
                self.assertFalse(self.discovery._is_d3_decision_config(
                    self.path / "decision_config.json"
                ))
                self.assertEqual(self.discovery.decision_kind(self.path, config), "d1")
        (self.path / "decision_config.json").write_text("{broken")
        self.assertEqual(self.discovery.decision_kind(self.path, config), "d1")
        (self.path / "decision_config.json").unlink()
        self.assertEqual(self.discovery.decision_kind(self.path, config), "d1")
        (self.path / "decision_config.json").mkdir()
        self.assertFalse(self.discovery._is_d3_decision_config(
            self.path / "decision_config.json"
        ))

    def test_d1_both_model_types_architectures_and_auto_map(self):
        for model_type in ("lfm3_vl", "lfm3_omni"):
            for declaration in (
                {"architectures": ["D1Model"]},
                {"auto_map": {"AutoModel": "remote.code.D1Model"}},
                {"architectures": None, "auto_map": {"AutoModel": "D1Model"}},
            ):
                with self.subTest(model_type=model_type, declaration=declaration):
                    config = {"model_type": model_type, **declaration}
                    self.assertEqual(self.discovery.decision_kind(self.path, config), "d1")
            for declaration in ({}, {"auto_map": None}, {"architectures": ["Other"]}):
                self.assertIsNone(self.discovery.decision_kind(
                    self.path, {"model_type": model_type, **declaration}
                ))
        self.assertIsNone(self.discovery.decision_kind(
            self.path, {"model_type": "qwen3_5", "architectures": ["D1Model"]}
        ))

    def test_openjev_helper_requires_qwen_type(self):
        self.touch("helper/shim.py")
        self.assertEqual(self.discovery.decision_kind(
            self.path, {"model_type": "qwen3_5"}
        ), "openjev")
        for model_type in ("qwen3", "llama", None):
            self.assertIsNone(self.discovery.decision_kind(
                self.path, {"model_type": model_type}
            ))
        (self.path / "helper/shim.py").unlink()
        (self.path / "helper/shim.py").mkdir()
        self.assertIsNone(self.discovery.decision_kind(
            self.path, {"model_type": "qwen3_5"}
        ))

    def test_openjev_name_and_hf_snapshot_hints(self):
        config = {"model_type": "qwen3_5"}
        for path in (
            self.path / "OpenJev-MLX",
            self.path / "models--Org--OpenJev-MLX" / "snapshots" / "abc123",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.discovery._model_name_hint(path), "openjev-mlx")
                self.assertEqual(self.discovery.decision_kind(path, config), "openjev")
                self.assertIsNone(self.discovery.decision_kind(path, {"model_type": "llama"}))
        fallback = self.path / "not-a-cache" / "snapshots" / "OpenJev-Fallback"
        self.assertEqual(self.discovery._model_name_hint(fallback), "openjev-fallback")
        self.assertEqual(self.discovery.decision_kind(fallback, config), "openjev")
        plain = self.path / "models--Org--Qwen3.5" / "snapshots" / "openjev-hash"
        self.assertEqual(self.discovery._model_name_hint(plain), "qwen3.5")
        self.assertIsNone(self.discovery.decision_kind(plain, config))

    def test_hf_decoder_preserves_original_edge_semantics(self):
        cases = {
            "regular": None,
            "models--": None,
            "models--Standalone": ("Standalone", "Standalone"),
            "models--Org--Name": ("Org--Name", "Org/Name"),
            "models--Org--Name--More": ("Org--Name--More", "Org/Name--More"),
            "models----Name": ("--Name", "/Name"),
            "models--Org--": ("Org--", "Org/"),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.discovery._decode_hf_cache_model_id(Path(name)), expected)

    def test_plain_qwen_and_partial_elephant_markers(self):
        for config, expected in (
            ({"model_type": "qwen3_5"}, None),
            ({"model_type": "qwen3_5", "modeling_overrides": {"readout": "slot_lm"}}, "elephant"),
            ({"model_type": "qwen3_5", "modeling_overrides": {"attention": {
                "full_attention": "block_bidirectional"
            }}}, "elephant"),
            ({"model_type": "llama"}, None),
            ({}, None),
        ):
            with self.subTest(config=config):
                self.assertEqual(self.discovery.decision_kind(self.path, config), expected)

    def test_passed_config_overrides_disk_without_reading_it(self):
        self.touch("helper/shim.py")
        self.write_json("config.json", {"model_type": "qwen3_5"})
        self.assertEqual(self.discovery.decision_kind(self.path), "openjev")
        self.assertIsNone(self.discovery.decision_kind(self.path, {}))
        self.assertIsNone(self.discovery.decision_kind(self.path, {"model_type": "llama"}))
        with patch("builtins.open", side_effect=AssertionError("No disk config read")):
            self.assertEqual(self.discovery.decision_kind(
                self.path, {"model_type": "qwen3_5"}
            ), "openjev")

    def test_missing_malformed_and_unreadable_config_return_none_before_hint(self):
        path = self.path / "OpenJev-MLX"
        path.mkdir()
        self.assertIsNone(self.discovery.decision_kind(path))
        config_path = path / "config.json"
        config_path.write_text("{broken")
        self.assertIsNone(self.discovery.decision_kind(path))
        config_path.unlink()
        config_path.mkdir()
        self.assertIsNone(self.discovery.decision_kind(path))
        self.assertEqual(self.discovery.decision_kind(
            path, {"model_type": "qwen3_5"}
        ), "openjev")

    def test_nondict_configs_without_elephant_readout_return_none(self):
        for config in ([], "text", 1, False):
            with self.subTest(config=config):
                self.assertIsNone(self.discovery.decision_kind(self.path, config))
        for config in ([], "text", 1, False, None):
            with self.subTest(disk_config=config):
                self.write_json("config.json", config)
                self.assertIsNone(self.discovery.decision_kind(self.path))

    def test_malformed_d1_fields_preserve_head_errors_and_short_circuit(self):
        for auto_map in (["D1Model"], "D1Model", 1):
            with self.subTest(auto_map=auto_map):
                config = {"model_type": "lfm3_vl", "auto_map": auto_map}
                with self.assertRaises(AttributeError):
                    self.discovery.decision_kind(self.path, config)
                config["architectures"] = ["D1Model"]
                self.assertEqual(self.discovery.decision_kind(self.path, config), "d1")
        with self.assertRaises(TypeError):
            self.discovery.decision_kind(
                self.path, {"model_type": "lfm3_omni", "architectures": 1}
            )

    def test_compatibility_reexports_and_shared_references_without_parent_import(self):
        tree = ast.parse((ROOT / "omlx/model_discovery.py").read_text())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == "decision_discovery"]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].level, 1)
        self.assertEqual({alias.name for alias in imports[0].names}, MOVED_NAMES)
        for alias in imports[0].names:
            self.assertEqual(alias.asname, alias.name)
            self.assertTrue(hasattr(self.discovery, alias.name))
        local_definitions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        } | {
            target.id for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        self.assertFalse(local_definitions & MOVED_NAMES)
        calls = {node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue({"decision_kind", "_model_name_hint", "_decode_hf_cache_model_id"} <= calls)
        self.assertNotIn("omlx.model_discovery", sys.modules)

    def test_lightweight_dependency_surface_and_original_constants(self):
        tree = ast.parse((ROOT / "omlx/decision_discovery.py").read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        self.assertEqual(imports, {"json", "pathlib", "typing"})
        self.assertEqual(self.discovery.DecisionKind.__args__, ("clef", "openjev", "d3", "d1", "elephant"))
        constants = {
            "CLEF_HEAD_WEIGHTS": "joint_head.safetensors",
            "CLEF_HEAD_CONFIG": "joint_head_config.json",
            "D3_READOUT_WEIGHTS": "readout.safetensors",
            "D3_DECISION_CONFIG": "decision_config.json",
            "D3_FORMAT_ID": "d3-code-readout-v1",
            "_OPENJEV_HELPER": Path("helper/shim.py"),
        }
        for name, expected in constants.items():
            self.assertEqual(getattr(self.discovery, name), expected)


if __name__ == "__main__":
    unittest.main()
