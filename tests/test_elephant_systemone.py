# SPDX-License-Identifier: Apache-2.0
"""Elephant over POST /v1/systemone: strict body parsing and wire contract.

mock contract only - not live inference. No real weights or MLX are loaded.

Stdlib unittest only. The strict parser is loaded by file path because the
``omlx.api`` package imports pydantic. Elephant runs on the fake backbone from
tests/test_elephant.py. server.py needs fastapi and mlx, so its wiring is
covered by a static AST check only.
"""

import ast
import importlib.util
import json
import math
import unittest
from pathlib import Path

import tests.test_elephant as harness

ROOT = Path(__file__).resolve().parents[1]


def _load_strict():
    path = ROOT / "omlx" / "api" / "systemone_strict.py"
    spec = importlib.util.spec_from_file_location("systemone_strict_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strict = _load_strict()


def _body(**overrides):
    body = {"model": "elephant", "state": "synthetic",
            "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}
    body.update(overrides)
    return body


class StrictParserTests(unittest.TestCase):
    def reject(self, raw, fragment):
        with self.assertRaises(strict.SystemOneStrictError) as caught:
            strict.parse_strict_systemone_body(raw)
        self.assertIn(fragment, str(caught.exception))
        return str(caught.exception)

    def test_clean_body_round_trips_in_order(self):
        raw = json.dumps(_body(images=[], videos=None, truncate=False))
        for value in (raw, raw.encode()):
            self.assertEqual(strict.parse_strict_systemone_body(value), json.loads(raw))
        parsed = strict.parse_strict_systemone_body(
            '{"model":"m","state":{"z":1,"a":2},"questions":{"q":{"type":"choice","criteria":{"z":null,"a":null,"m":null}}}}')
        self.assertEqual(list(parsed["questions"]["q"]["criteria"]), ["z", "a", "m"])
        self.assertEqual(list(parsed["state"]), ["z", "a"])

    def test_duplicate_keys_at_any_depth(self):
        for raw in (
            '{"model":"m","model":"n","state":"s","questions":{"q":{"type":"noul"}}}',
            '{"model":"m","state":"s","questions":{"q":{"type":"noul"},"q":{"type":"noul"}}}',
            '{"model":"m","state":"s","questions":{"q":{"type":"noul","type":"choice"}}}',
            '{"model":"m","state":{"a":{"secret":1,"secret":2}},"questions":{"q":{"type":"noul"}}}',
            '{"model":"m","state":[{"k":1,"k":1}],"questions":{"q":{"type":"noul"}}}',
        ):
            with self.subTest(raw=raw):
                message = self.reject(raw, "Duplicate JSON key")
                self.assertNotIn("secret", message)

    def test_non_finite_tokens(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                self.reject('{"model":"m","state":%s,"questions":{"q":{"type":"noul"}}}' % token, "Non-finite")
        self.reject('{"model":"m","state":"s","questions":{"q":{"type":"score","criteria":[1, NaN]}}}', "Non-finite")

    def test_top_level_must_be_object_and_valid_json(self):
        for raw in ("[]", '"text"', "1", "null", "true"):
            with self.subTest(raw=raw):
                self.reject(raw, "must be a JSON object")
        self.reject("{", "not valid JSON")
        self.reject("", "not valid JSON")
        self.reject(b"\xff\xfe", "not valid UTF-8")
        self.reject("[" * 100000 + "]" * 100000, "nested too deeply")
        with self.assertRaises(strict.SystemOneStrictError):
            strict.parse_strict_systemone_body(None)

    def test_unknown_and_unsupported_fields(self):
        for key in ("extra", "stream", "temperature", "Model"):
            with self.subTest(key=key):
                message = self.reject(json.dumps(_body(**{key: 1})), "Unknown field in the request")
                self.assertNotIn(key, message)
        for key in ("audio", "tools", "tool_choice", "input_audio", "image_url", "functions"):
            with self.subTest(key=key):
                self.reject(json.dumps(_body(**{key: []})), f"does not support '{key}' in the request")
        question = {"type": "choice", "criteria": {"a": None, "b": None}}
        for key, fragment in (("extra", "Unknown field in a question"), ("name", "Unknown field in a question"),
                              ("audio", "does not support 'audio' in a question"),
                              ("tools", "does not support 'tools' in a question")):
            with self.subTest(question_key=key):
                self.reject(json.dumps(_body(questions={"q": dict(question, **{key: "x"})})), fragment)

    def test_media_fields_must_be_empty(self):
        for key in ("images", "videos"):
            with self.subTest(key=key):
                self.reject(json.dumps(_body(**{key: ["data:image/png;base64,AAAA"]})), f"'{key}' must be empty")
        for value in (None, []):
            self.assertEqual(strict.parse_strict_systemone_body(json.dumps(_body(images=value, videos=value)))["images"], value)

    def test_error_is_a_value_error_and_never_echoes_values(self):
        self.assertTrue(issubclass(strict.SystemOneStrictError, ValueError))
        message = self.reject(json.dumps(_body(leaked="TOP-SECRET-VALUE")), "Unknown field")
        self.assertNotIn("TOP-SECRET", message)

    def test_allowlists_match_pydantic_models(self):
        """Static: field sets equal the annotated fields in systemone_models.py."""
        tree = ast.parse((ROOT / "omlx" / "api" / "systemone_models.py").read_text())
        fields = {
            node.name: {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
            for node in tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertEqual(fields["SystemOneRequest"], set(strict.REQUEST_FIELDS))
        self.assertEqual(fields["SystemOneQuestion"], set(strict.QUESTION_FIELDS))
        self.assertFalse(strict._UNSUPPORTED_KEYS & (strict.REQUEST_FIELDS | strict.QUESTION_FIELDS))

    def test_module_is_stdlib_only(self):
        tree = ast.parse((ROOT / "omlx" / "api" / "systemone_strict.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or ".").split(".")[0] if node.level == 0 else ".")
        self.assertEqual(imported, {"json", "typing"})


class ElephantSystemOneContractTests(unittest.TestCase):
    """Elephant's /v1/systemone answers on the fake backbone (no MLX)."""

    write_contract = harness.ElephantTests.write_contract

    def setUp(self):
        harness.ElephantTests.setUp(self)
        # Distinct slot logits: hidden[0, -1] is [1.0, id], so logit i == rows[i][0].
        self.logits = [1.0, 0.25, -0.5]
        self.backbone.rows = harness.Array([[x, 0.0] for x in self.logits] + [[0.0, 0.0]] * 253)
        self.model.load()

    def answer(self, question, state="synthetic"):
        body = strict.parse_strict_systemone_body(json.dumps(
            {"model": "elephant", "state": state, "questions": {"q": question}}))
        plan = self.model.encode(body)
        steps = self.model.run(plan, lambda: 1)
        while True:
            try:
                next(steps)
            except StopIteration as stop:
                return plan, stop.value

    def expected(self, n):
        exps = [math.exp(x) for x in self.logits[:n]]
        return [e / sum(exps) for e in exps]

    def assertVector(self, got, n):
        for value, want in zip(got, self.expected(n)):
            self.assertAlmostEqual(value, want, places=6)
        self.assertLessEqual(abs(sum(got) - 1), 1e-5)
        self.assertTrue(all(0 < v < 1 for v in got), "one-hot vector")
        self.assertGreater(max(got) - min(got), 0.1, "uniform vector")

    def test_choice_two_and_three_options_keep_order_and_full_vector(self):
        for criteria in ({"z": None, "a": None}, {"z": None, "a": "first letter", "m": None}):
            with self.subTest(n=len(criteria)):
                _, result = self.answer({"type": "choice", "criteria": criteria})
                answer = result["answers"]["q"]
                self.assertEqual(list(answer["probabilities"]), list(criteria))
                self.assertEqual(set(answer), {"type", "probabilities", "choice", "confidence"})
                self.assertVector(list(answer["probabilities"].values()), len(criteria))
                self.assertEqual(answer["choice"], "z")

    def test_score_vector_and_legend(self):
        _, result = self.answer({"type": "score", "criteria": ["low", None, "high"]})
        answer = result["answers"]["q"]
        self.assertEqual(list(answer["probabilities"]), ["0", "1", "2"])
        self.assertVector(list(answer["probabilities"].values()), 3)
        p = self.expected(3)
        self.assertAlmostEqual(answer["score"], sum(i * v for i, v in enumerate(p)), places=6)
        self.assertEqual(answer["legend"], {"0": "low", "1": None, "2": "high"})

    def test_noul_is_scalar_softmax_yes(self):
        _, result = self.answer({"type": "noul", "instructions": "Is it urgent?"})
        answer = result["answers"]["q"]
        self.assertEqual(set(answer), {"type", "noul"})
        self.assertAlmostEqual(answer["noul"], self.expected(2)[0], places=6)
        self.assertNotIn(answer["noul"], (0, 0.5, 1))

    def test_one_and_four_options_are_rejected_without_padding(self):
        for kind, criteria in (("choice", {"a": None}), ("choice", dict.fromkeys("abcd")),
                               ("score", ["only"]), ("score", [1, 2, 3, 4])):
            with self.subTest(kind=kind, n=len(criteria)), self.assertRaises(self.runtime.DecisionRequestError):
                self.answer({"type": kind, "criteria": criteria})

    def test_usage_is_measured_not_invented(self):
        plan, result = self.answer({"type": "choice", "criteria": {"a": None, "b": None}})
        self.assertEqual(set(result), {"answers", "input_tokens"})
        self.assertEqual(result["input_tokens"], len(plan.prefix) + len(plan.suffix))
        self.assertGreater(result["input_tokens"], 0)

    def test_strict_parser_catches_what_pydantic_would_drop(self):
        # Elephant's own allowlist accepts the keys pydantic leaves behind, so
        # these must be rejected on the raw body before encode().
        for raw in ('{"model":"m","state":"s","questions":{"q":{"type":"noul"}},"audio":"x"}',
                    '{"model":"m","state":"s","questions":{"q":{"type":"noul","tools":[]}}}',
                    '{"model":"m","state":"a","state":"b","questions":{"q":{"type":"noul"}}}'):
            with self.subTest(raw=raw), self.assertRaises(strict.SystemOneStrictError):
                strict.parse_strict_systemone_body(raw)


class ServerWiringStaticTests(unittest.TestCase):
    """Static AST check of omlx/server.py; the handler is not executed here."""

    def test_systemone_handler_runs_strict_parser_for_elephant_before_encode(self):
        tree = ast.parse((ROOT / "omlx" / "server.py").read_text())
        imports = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == "api.systemone_strict" for alias in node.names}
        self.assertEqual(imports, {"SystemOneStrictError", "parse_strict_systemone_body"})
        handlers = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "post"
            and [getattr(a, "value", None) for a in d.args] == ["/v1/systemone"] for d in node.decorator_list)]
        self.assertEqual(len(handlers), 1)
        handler = handlers[0]
        guard = [node for node in ast.walk(handler) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "engine.kind == 'elephant'"]
        self.assertEqual(len(guard), 1)
        guarded = {n.id for n in ast.walk(guard[0]) if isinstance(n, ast.Name)}
        self.assertTrue({"parse_strict_systemone_body", "SystemOneStrictError", "http_request"} <= guarded)
        self.assertIn("parse_strict_systemone_body(await http_request.body())", ast.unparse(guard[0]))
        catches = [h for h in ast.walk(guard[0]) if isinstance(h, ast.ExceptHandler)
                   and ast.unparse(h.type) == "SystemOneStrictError"]
        self.assertEqual(len(catches), 1)
        self.assertIn("status_code=400", ast.unparse(catches[0]))
        encodes = [n.lineno for n in ast.walk(handler) if isinstance(n, ast.Call)
                   and ast.unparse(n.func) == "engine.encode"]
        self.assertEqual(len(encodes), 1)
        self.assertLess(guard[0].lineno, encodes[0])


if __name__ == "__main__":
    unittest.main()
