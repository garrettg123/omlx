# SPDX-License-Identifier: Apache-2.0
"""No-model tests: stdlib unittest, also collectable by isolated pytest.

Never import the converter, real backbone, MLX, torch or transformers. Fakes
verify production control flow, not attention/FP32 numerical compatibility.
"""

import asyncio
import copy
import importlib
import importlib.abc
import json
import math
import socket
import subprocess
import sys
import tempfile
import types
import unittest
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/elephant_contract.json").read_text())


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected loader, download, subprocess or GPU operation")


class ImportTrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"omlx.utils.model_loading", "omlx.utils.image"} or fullname.split(".")[0] in {
            "mlx", "mlx_audio", "mlx_lm", "mlx_vlm", "torch", "transformers", "huggingface_hub", "safetensors", "numpy"
        }:
            raise AssertionError(f"Real dependency import forbidden: {fullname}")
        return None


class Array:
    """Only list arithmetic and dtype spies; not an MLX approximation."""
    events = []
    def __init__(self, data, dtype="fake"):
        self.data = copy.deepcopy(data)
        self.dtype = dtype

    @property
    def shape(self):
        def shape(data):
            return (len(data),) + shape(data[0]) if isinstance(data, (tuple, list)) else ()
        return shape(self.data)

    def astype(self, dtype):
        return Array(self.data, dtype)

    def tolist(self):
        return copy.deepcopy(self.data)

    def __getitem__(self, key):
        data = self.data
        for part in key if isinstance(key, tuple) else (key,):
            data = data[part]
        return Array(data, self.dtype)

    def __matmul__(self, vector):
        assert self.dtype == vector.dtype == "float32"
        self.events.append(("projection", vector.tolist()))
        return Array([sum(a * b for a, b in zip(row, vector.data)) for row in self.data])


class Cache:
    def __init__(self):
        self.state = []


class Tokenizer:
    def __init__(self, labels):
        self.labels = labels
        self.mode = "normal"
        self.calls = []

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        if text in self.labels:
            self.last_label = text
            index = self.labels.index(text)
            return [0 if self.mode == "collision" else index] if self.mode != "split" else [index, index]
        ids = [ord(c) for c in text]
        # Simulate a tokenizer merge spanning the end of the STATE block.
        boundary = text.find("\n\nQuestion:")
        if self.mode == "merge" and boundary >= 0:
            ids[boundary] = 99999
        return ids if self.mode != "overflow" else ids * 3

    def decode(self, ids):
        if self.mode == "collision":
            return self.last_label  # Adversarial stateful tokenizer: hit ID guard too.
        return "wrong" if self.mode == "decode" else self.labels[ids[0]]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        self.calls.append(copy.deepcopy(messages))
        return messages[0]["content"] + "\nUSER:" + messages[1]["content"] + "\nASSISTANT:"


class FakeBackbone:
    def __init__(self, labels, events):
        self.tokenizer = Tokenizer(labels)
        self.events = events
        self.has_vision = False
        self.closed = False
        self.caches = []
        self.rows = Array([[math.log(2), 0]] + [[0, 0] for _ in range(255)])
        def embed(ids):
            events.append(("embed", tuple(ids.data[0])))
            return Array([[[1.0, float(i)] for i in ids.data[0]]])
        def layer(index):
            def forward(hidden, *, mask, cache):
                assert mask is None
                events.append(("layer", index, tuple(cache.state), hidden.shape[1], id(cache)))
                cache.state.append(hidden.shape[1])
                return hidden
            return forward
        def norm(hidden):
            events.append(("norm", hidden.data[0][-1]))
            return hidden
        self._text_model = types.SimpleNamespace(embed_tokens=embed, layers=[layer(i) for i in range(4)], norm=norm)

    def load(self):
        self.events.append(("fake_load",))

    def close(self):
        self.closed = True

    def output_rows(self, ids):
        assert ids.data == list(range(256))
        self.events.append(("rows",))
        return self.rows

    def make_cache(self):
        cache = [Cache() for _ in range(4)]
        self.caches.append(cache)
        return cache


class ElephantTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="elephant-contract-"))
        self.path = Path(directory)
        self.events = []
        self.stack.enter_context(patch.object(Array, "events", self.events))
        modules = {}
        def module(name, **attrs):
            obj = types.ModuleType(name)
            obj.__dict__.update(attrs)
            modules[name] = obj
            return obj
        for name, subdir in (("omlx", "omlx"), ("omlx.models", "omlx/models"), ("omlx.engine", "omlx/engine")):
            module(name, __path__=[str(ROOT / subdir)])
        module("mlx", __path__=[])
        module("mlx.core", array=Array, float32="float32", eval=lambda *a: self.events.append(("eval",)), synchronize=forbidden)
        class RequestError(ValueError):
            pass
        class ContextError(ValueError):
            pass
        module("omlx.models.decision", DecisionBackbone=forbidden, DecisionRequestError=RequestError, DecisionContextLengthError=ContextError)
        module("omlx.gguf", GgufModel=type("GgufModel", (), {}))
        for name, cls in (("clef", "ClefModel"), ("d1", "D1Model"), ("d3", "D3Model"), ("openjev", "OpenJevModel")):
            module("omlx.models." + name, **{cls: type(cls, (), {})})
        module("omlx.engine_core", get_mlx_executor=lambda: None)
        module("omlx.scheduler", _CONTENDED_CHUNK_FLOOR=1, SchedulerConfig=type("SchedulerConfig", (), {}))
        module("omlx.engine.base", BaseNonStreamingEngine=type("BaseNonStreamingEngine", (), {}))
        module("omlx.engine.forward_fairness", ForwardFairnessGate=lambda *a: None)
        modules["mlx_audio"] = None  # Scoped absence: parent catches ImportError, never real audio.
        self.stack.enter_context(patch.dict(sys.modules, modules))
        trap = ImportTrap()
        sys.meta_path.insert(0, trap)
        self.addCleanup(sys.meta_path.remove, trap)
        for obj, attr in ((socket.socket, "connect"), (urllib.request, "urlopen"), (subprocess, "Popen"), (subprocess, "run")):
            self.stack.enter_context(patch.object(obj, attr, forbidden))
        self.runtime = importlib.import_module("omlx.models.elephant")
        self.discovery = importlib.import_module("omlx.model_discovery")
        self.sibling = importlib.import_module("omlx.decision_discovery")
        self.engine = importlib.import_module("omlx.engine.decision")
        self.backbone = FakeBackbone(FIXTURE["slot_labels"].split(), self.events)
        self.runtime.DecisionBackbone = lambda *a, **k: self.backbone
        self.model = self.runtime.ElephantModel(str(self.path))
        self.config = copy.deepcopy(FIXTURE["config"])
        self.config["text_config"].update(hidden_size=2, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
        self.write_contract()

    def write_contract(self, readout=None):
        (self.path / "config.json").write_text(json.dumps(self.config))
        (self.path / "readout.json").write_text(json.dumps(FIXTURE["readout"] if readout is None else readout))

    def question(self, kind="choice"):
        criteria = {"a": None, "b": None, "c": None} if kind == "choice" else (["low", None, "high"] if kind == "score" else None)
        return self.runtime.render_question("q", {"type": kind, "criteria": criteria})

    def request(self):
        return {"state": "synthetic", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}}}

    def test_safety_traps_are_active(self):
        self.assertIsNone(sys.modules["mlx_audio"])
        with patch.dict(sys.modules):
            sys.modules.pop("mlx_audio")
            with self.assertRaises(AssertionError):
                importlib.import_module("mlx_audio")
        for name in ("torch", "transformers", "mlx_lm", "mlx_vlm", "numpy", "omlx.utils.model_loading"):
            with self.assertRaises(AssertionError):
                importlib.import_module(name)
        with self.assertRaises(AssertionError):
            sys.modules["omlx.models.decision"].DecisionBackbone("forbidden")
        with self.assertRaises(AssertionError):
            urllib.request.urlopen("https://invalid.example")
        with self.assertRaises(AssertionError):
            subprocess.run(["forbidden"])

    def test_golden_rendering_and_complete_label_order(self):
        self.assertEqual(tuple(FIXTURE["slot_labels"].split()), self.runtime.SLOT_LABELS)
        self.assertEqual(len(set(self.runtime.SLOT_LABELS)), 256)
        for fixture in FIXTURE["renderings"]:
            with self.subTest(fixture=fixture["question"]):
                q = self.runtime.render_question("hidden-id", fixture["question"])
                self.assertEqual(self.runtime.messages(fixture["state"], q.suffix), [
                    {"role": "system", "content": FIXTURE["system"]}, {"role": "user", "content": fixture["user"]}])

    def test_algebra_vectors_and_wire_shape(self):
        for kind in ("choice", "score", "noul"):
            fixture = FIXTURE["algebra"][kind]
            answer = self.runtime.assemble(self.question(kind), fixture["probabilities"])
            if kind == "choice":
                self.assertEqual(answer["choice"], fixture["winner"])
                self.assertEqual(answer["confidence"], fixture["confidence"])
                self.assertEqual(answer["probabilities"], {"a": .5, "b": .25, "c": .25})
            elif kind == "score":
                self.assertEqual(answer["score"], fixture["expected_score"])
                self.assertEqual(answer["confidence"], fixture["confidence"])
                self.assertEqual(answer["legend"], {"0": "low", "1": None, "2": "high"})
                self.assertEqual(answer["probabilities"], {"0": .1, "1": .2, "2": .7})
            else:
                self.assertEqual(answer, {"type": "noul", "noul": fixture["p_yes"]})
        self.assertEqual(self.runtime.softmax([math.log(2), 0, 0], 3), [.5, .25, .25])
        self.assertEqual(self.runtime.softmax([10000, 10000], 2), [.5, .5])
        self.assertEqual(self.runtime.assemble(self.question(), [1/3]*3)["choice"], "a")
        self.assertEqual(self.runtime.assemble(self.question("noul"), [.123456789, .876543211])["noul"], .123457)

    def test_invalid_vectors_logits_and_duplicate_labels(self):
        bad = [[], [0, 0, 0], [.5, .5], [1, 0, 0, 0], [float("nan"), 0, 1], [float("inf"), 0, 0], [-.1, .1, 1], [1.1, 0, 0], [.2, .2, .2], [True, 0, 0], {"a": 1, "b": 0}, {"a": 1, "b": 0, "c": 0, "d": 0}]
        for vector in bad:
            with self.subTest(vector=vector), self.assertRaises(self.runtime.DecisionRequestError):
                self.runtime.assemble(self.question(), vector)
        q = self.runtime.Question("q", "choice", ("x", "x"), (None, None), "")
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.runtime.assemble(q, [.5, .5])
        for values in ([float("nan"), 0], [float("inf"), 0], [1], [True, 0], ["x", 0], []):
            with self.assertRaises(self.runtime.DecisionRequestError):
                self.runtime.softmax(values, 2)
        self.assertEqual(self.runtime.assemble(self.question(), {"b": .25, "c": .25, "a": .5})["choice"], "a")
        rounded = self.runtime.assemble(self.question(), [1/3]*3)["probabilities"]
        self.assertLessEqual(abs(sum(rounded.values()) - 1), 1.000001e-6)

    def test_load_slots_fp32_and_cleanup(self):
        self.model.load()
        self.assertEqual(self.model.slot_ids, tuple(range(256)))
        self.assertEqual(self.model.slot_rows.dtype, "float32")
        self.model.close()
        self.assertTrue(self.backbone.closed)
        self.assertIsNone(self.model.backbone)
        for mode in ("split", "collision", "decode"):
            self.backbone.tokenizer.mode = mode
            with self.assertRaises(self.runtime.DecisionRequestError):
                self.model.load()
            self.assertIsNone(self.model.backbone)
            self.assertIsNone(self.model.slot_rows)
        self.backbone.tokenizer.mode = "normal"
        for rows in (Array([[0, 0]]), Array([[float("nan"), 0]] * 256)):
            self.backbone.rows = rows
            with self.assertRaises(self.runtime.DecisionRequestError):
                self.model.load()

    def test_contract_rejects_before_loader(self):
        self.config = copy.deepcopy(FIXTURE["config"])
        self.write_contract()
        self.assertEqual(self.model._contract(), FIXTURE["config"]["text_config"])
        for mutation in (lambda c: c.update(model_type="qwen3"), lambda c: c.update(vision_config={}), lambda c: c["modeling_overrides"].update(readout="head"), lambda c: c["modeling_overrides"]["attention"].update(linear_attention="noncausal"), lambda c: c["modeling_overrides"].update(n_slots=255), lambda c: c["modeling_overrides"].update(unknown="mode")):
            self.config = copy.deepcopy(FIXTURE["config"])
            mutation(self.config)
            self.write_contract()
            with self.assertRaises(self.runtime.DecisionRequestError):
                self.model.load()
        self.assertNotIn(("fake_load",), self.events)
        self.config = copy.deepcopy(FIXTURE["config"])
        self.write_contract({"readout": "head", "n_slots": 256, "head_dim": 256})
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.model.load()
        self.write_contract()
        (self.path / "calibration.json").write_text("{}")
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.model.load()

    def test_partial_load_failures_and_all_contract_files(self):
        self.backbone.has_vision = True
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.model.load()
        self.assertTrue(self.backbone.closed)
        self.assertIsNone(self.model.backbone)
        self.backbone.has_vision = False
        self.backbone.tokenizer.encode = lambda *a, **k: [248320]
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.model.load()
        self.assertIsNone(self.model.slot_rows)
        self.runtime.DecisionBackbone = forbidden
        for content in (None, "[]", "{", '{"readout":"head"}'):
            file = self.path / "readout.json"
            if content is None:
                file.unlink()
            else:
                file.write_text(content)
            self.assertEqual(self.discovery.decision_kind(self.path), "elephant")
            with self.assertRaises(self.runtime.DecisionRequestError):
                self.model.load()
        self.write_contract()
        self.model.trust_remote_code = True
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.model.load()
        self.model.trust_remote_code = False
        for content in ("[]", "{", "null"):
            (self.path / "config.json").write_text(content)
            self.assertEqual(self.discovery.decision_kind(self.path), "elephant")
            with self.assertRaises(self.runtime.DecisionRequestError):
                self.model.load()

    def test_prefix_boundary_merge_retreat_and_limits(self):
        self.model.load()
        plan = self.model.encode(self.request())
        self.assertTrue(plan.prefix and plan.suffix)
        self.assertEqual(list(plan.prefix + plan.suffix), self.backbone.tokenizer.encode(self.backbone.tokenizer.apply_chat_template(self.backbone.tokenizer.calls[0], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
        self.assertLess(len(plan.prefix), len(plan.prefix + plan.suffix))
        self.assertEqual(plan.suffix[0], ord("Q"))
        self.backbone.max_position_embeddings = 1
        with self.assertRaises(self.runtime.DecisionContextLengthError):
            self.model.encode(self.request(), truncate=False)
        self.backbone.max_position_embeddings = 512
        self.assertEqual(self.backbone.tokenizer.calls[0][1]["content"].split("</state>\n\n")[1][:9], "Question:")
        self.backbone.tokenizer.mode = "merge"
        merged = self.model.encode(self.request())
        self.assertEqual(len(merged.prefix), len(plan.prefix) - 2)
        self.backbone.tokenizer.mode = "overflow"
        with self.assertRaises(self.runtime.DecisionContextLengthError):
            self.model.encode(self.request(), truncate=True)
        self.backbone.tokenizer.mode = "normal"
        for state in ("QEV_PREFIX_MARK", "x" * 2049):
            request = self.request()
            request["state"] = state
            with self.assertRaises((self.runtime.DecisionRequestError, self.runtime.DecisionContextLengthError)):
                self.model.encode(request)

    def test_reject_unsupported_requests(self):
        self.model.load()
        cases = [{"images": ["no"]}, {"videos": ["no"]}, {"tools": []}, {"questions": {}}, {"state": [{"type": "image", "image_url": "no"}]}, {"state": "<|vision_start|>"}, {"state": {"<image>": "placeholder"}}, {"questions": {"q": {"type": "choice", "criteria": {"<image>": None, "safe": None}}}}, {"state": [{"type": "audio", "audio_url": "synthetic://no-data"}]}, {"state": [{"type": "input_audio"}]}, {"state": {"audio": "no"}}]
        for change in cases:
            request = self.request() | change
            with self.subTest(change=change), self.assertRaises(self.runtime.DecisionRequestError):
                self.model.encode(request)
        for question in ({"type": "unknown"}, {"type": "choice", "criteria": {1: None, "1": None}}, {"type": "choice", "criteria": {"": None, "a": None}}, {"type": "choice", "criteria": {"a": None}}, {"type": "score", "criteria": [None]*4}, {"type": "noul", "criteria": {"yes": None}}, {"type": "noul", "debias": 2}, {"type": "noul", "instructions": float("nan")}):
            with self.assertRaises(self.runtime.DecisionRequestError):
                self.runtime.render_question("q", question)
        self.model.encode(self.request() | {"images": [], "videos": None})
        request = self.request()
        request["questions"]["other"] = {"type": "noul"}
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.model.encode(request)
        self.events.clear()
        with patch.object(self.backbone.tokenizer, "encode", return_value=[1]), self.assertRaises(self.runtime.DecisionRequestError):
            self.model.encode(self.request())
        self.assertEqual(self.events, [])

    def test_short_plan_gate_and_minimum_bounded_steps(self):
        self.config["text_config"].update(num_hidden_layers=32, layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 8)
        self.write_contract()
        self.backbone._text_model.layers *= 8
        self.model.load()
        for prefix, suffix, chunk in ((1, 1, 1), (7, 8, 1), (8, 7, 1), (8, 8, 1), (9, 11, 1), (9, 11, 100)):
            def encode(text, **kwargs):
                return list(range(prefix if text.endswith("</state>\n\n") else prefix + suffix))
            self.events.clear()
            with patch.object(self.backbone.tokenizer, "encode", encode):
                if min(prefix, suffix) < 8:
                    with patch.object(self.backbone, "make_cache", forbidden), patch.object(self.backbone._text_model, "embed_tokens", forbidden), self.assertRaisesRegex(self.runtime.DecisionRequestError, "at least 8 tokens"):
                        self.model.encode(self.request())
                    with patch.object(self.backbone, "make_cache", forbidden), self.assertRaises(self.runtime.DecisionRequestError):
                        next(self.model.run(self.runtime.Plan((self.question(),), tuple(range(prefix)), tuple(range(suffix))), lambda: chunk))
                    self.assertEqual(self.events, [])
                    continue
                plan = self.model.encode(self.request())
            with patch.object(self.backbone, "make_cache", return_value=[Cache() for _ in range(32)]):
                progress = []
                for amount in self.model.run(plan, lambda: chunk):
                    progress.append(amount)
                    self.assertIn(sum(e[0] == "layer" for e in self.events), range(1, 5))
                    self.events.clear()
                self.assertTrue(all(p > 0 for p in progress))
                self.assertEqual(sum(progress), prefix + suffix)

    def test_two_whole_blocks_cache_order_progress_and_interruption(self):
        self.model.load()
        plan = self.model.encode(self.request())
        steps = self.model.run(plan, lambda: 1)
        progress = []
        while True:
            try:
                progress.append(next(steps))
            except StopIteration as stop:
                result = stop.value
                break
        self.assertTrue(all(p > 0 for p in progress))
        self.assertEqual(sum(progress), len(plan.prefix + plan.suffix))
        self.assertEqual(result["input_tokens"], sum(progress))
        self.assertEqual(result["answers"]["q"]["probabilities"], {"a": .5, "b": .25, "c": .25})
        layer_events = [e for e in self.events if e[0] == "layer"]
        self.assertEqual([e[1] for e in layer_events], list(range(4))*2)
        for prefix, suffix in zip(layer_events[:4], layer_events[4:]):
            self.assertEqual(prefix[2:4], ((), len(plan.prefix)))
            self.assertEqual(suffix[2:4], ((len(plan.prefix),), len(plan.suffix)))
            self.assertNotEqual(prefix[4], suffix[4])
        self.assertEqual([e[1] for e in self.events if e[0] == "embed"], [plan.prefix, plan.suffix])
        self.assertEqual([e[1] for e in self.events if e[0] == "norm"], [[1.0, float(plan.prefix[-1])], [1.0, float(plan.suffix[-1])]])
        self.assertEqual([e[1] for e in self.events if e[0] == "projection"], [[1.0, float(plan.suffix[-1])]])
        self.assertEqual([e[0] for e in self.events][-3:], ["norm", "projection", "eval"])
        self.assertTrue(all(not cache for cache in self.backbone.caches))
        interrupted = self.model.run(plan, lambda: 1)
        next(interrupted)
        interrupted.close()
        self.assertIsNone(interrupted.gi_frame)
        self.assertTrue(all(not cache for cache in self.backbone.caches))
        interrupted_suffix = self.model.run(plan, lambda: 1)
        for _ in range(5):
            next(interrupted_suffix)
        interrupted_suffix.close()
        self.assertIsNone(interrupted_suffix.gi_frame)
        self.assertTrue(all(not cache for cache in self.backbone.caches))
        names = [e[0] for e in self.events]
        self.assertLess(names.index("norm"), names.index("embed", names.index("norm")))

    def test_discovery_other_families_and_conflicting_markers(self):
        for family, files in (
            ("clef", {self.discovery.CLEF_HEAD_WEIGHTS: "", self.discovery.CLEF_HEAD_CONFIG: "{}"}),
            ("d3", {self.discovery.D3_READOUT_WEIGHTS: "", self.discovery.D3_DECISION_CONFIG: json.dumps({"format_id": self.discovery.D3_FORMAT_ID})}),
            ("d1", {"config.json": json.dumps({"model_type": "lfm3_vl", "architectures": ["D1Model"]})}),
        ):
            directory = self.path / family
            directory.mkdir()
            for name, content in files.items():
                (directory / name).write_text(content)
            self.assertEqual(self.discovery.decision_kind(directory), family)
        self.config["modeling_overrides"]["readout"] = "head"
        self.write_contract()
        self.assertEqual(self.discovery.decision_kind(self.path), "elephant")
        self.runtime.DecisionBackbone = forbidden
        with self.assertRaises(self.runtime.DecisionRequestError):
            self.model.load()
        self.config.pop("modeling_overrides")
        self.write_contract()
        self.assertEqual(self.discovery.decision_kind(self.path), "elephant")

    def test_discovery_binding_and_failed_engine_start(self):
        self.assertIs(self.discovery.decision_kind, self.sibling.decision_kind)
        self.assertIs(self.discovery.DecisionKind, self.sibling.DecisionKind)
        self.assertIs(self.engine.decision_kind, self.sibling.decision_kind)
        for name in ("neutral", "misleading-openjev"):
            directory = self.path / name
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(FIXTURE["config"]))
            self.assertEqual(self.discovery.decision_kind(directory), "elephant")
        self.assertIs(self.engine._MODEL_CLASSES["elephant"], self.runtime.ElephantModel)
        for model_type, name, expected in (("qwen3_5", "elephant", None), ("qwen3_5", "openjev", "openjev"), ("qwen3", "openjev", None)):
            directory = self.path / name
            directory.mkdir(exist_ok=True)
            (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
            self.assertEqual(self.discovery.decision_kind(directory), expected)
        engine = self.engine.DecisionEngine(str(self.path))
        self.backbone.load = forbidden
        with self.assertRaises(AssertionError):
            asyncio.run(engine.start())
        self.assertIsNone(engine._model)
        self.assertIsNone(engine._kind)
        self.assertIsNone(self.model.backbone)
        self.assertTrue(self.backbone.closed)


if __name__ == "__main__":
    unittest.main()
