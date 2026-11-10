# SPDX-License-Identifier: Apache-2.0
"""Opt-in parity of the fork's Elephant runtime against the publisher reference.

SKIPPED unless ELEPHANT_MODEL_PATH names a local elephant-1 MLX checkpoint. Loads the real
model in-process (about 10 GB RSS) and compares the fork's answer probabilities with the
publisher's decision_mlx.py (ELEPHANT_REFERENCE_PATH, default /tmp/el8-decision_mlx.py) for one
8-option choice question and one 5-level score question. The two models are loaded one after
the other, never at the same time.
"""

import gc
import importlib.util
import os
import sys
import unittest
from pathlib import Path

MODEL_PATH = os.environ.get("ELEPHANT_MODEL_PATH")
REFERENCE_PATH = os.environ.get("ELEPHANT_REFERENCE_PATH", "/tmp/el8-decision_mlx.py")
TOLERANCE = 1e-6

STATE = (
    "Ticket #4127: the customer says the invoice PDF they downloaded this morning shows last "
    "month's total, and they need the corrected copy before their accounts team closes the books "
    "on Friday."
)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "invoices, charges and refunds",
            "technical": "bugs, outages and errors",
            "account": "logins, profiles and permissions",
            "sales": "pricing questions and upgrades",
            "shipping": "deliveries and tracking",
            "legal": "contracts and compliance",
            "feedback": "suggestions and praise",
            "spam": None,
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["not urgent", "low", "moderate", "high", "critical"],
    },
}


def _load_reference():
    name = "el8_decision_mlx_reference"
    spec = importlib.util.spec_from_file_location(name, REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and pydantic resolve annotations through sys.modules[__module__].
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _release_mlx():
    gc.collect()
    import mlx.core as mx

    clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if clear is not None:
        clear()


@unittest.skipUnless(MODEL_PATH, "set ELEPHANT_MODEL_PATH to run the real-model parity test")
class ElephantReferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = str(Path(MODEL_PATH).expanduser())
        if not Path(REFERENCE_PATH).is_file():
            raise unittest.SkipTest(f"reference runtime not found at {REFERENCE_PATH}")

        from omlx.models.elephant import ElephantModel

        fork = ElephantModel(model_path)
        fork.load()
        try:
            cls.fork = {}
            for qid, question in QUESTIONS.items():
                plan = fork.encode({"model": "elephant", "state": STATE, "questions": {qid: question}})
                steps = fork.run(plan, lambda: 2048)
                while True:
                    try:
                        next(steps)
                    except StopIteration as stop:
                        cls.fork[qid] = stop.value["answers"][qid]
                        break
        finally:
            fork.close()
            del fork
            _release_mlx()

        reference = _load_reference()
        engine = reference.DecisionEngine(model_path)
        try:
            cls.reference = {}
            for qid, question in QUESTIONS.items():
                request = reference.SystemOneRequest.model_validate(
                    {"model": "elephant", "state": STATE, "questions": {qid: question}})
                response, _ = engine.decide(request, debias=1)
                cls.reference[qid] = response.model_dump()["answers"][qid]
        finally:
            del engine
            _release_mlx()

    def assertParity(self, qid, labels):
        fork, ref = self.fork[qid]["probabilities"], self.reference[qid]["probabilities"]
        self.assertEqual(list(fork), labels)
        self.assertEqual(list(ref), labels)
        diff = max(abs(fork[k] - ref[k]) for k in labels)
        print(f"\n[parity] {qid}: {len(labels)} slots, max abs diff {diff:.3g}", file=sys.stderr)
        self.assertLessEqual(diff, TOLERANCE)
        return fork, ref

    def test_eight_option_choice_matches_reference(self):
        labels = list(QUESTIONS["route"]["criteria"])
        fork, ref = self.assertParity("route", labels)
        self.assertEqual(self.fork["route"]["choice"], self.reference["route"]["choice"])
        self.assertEqual(self.fork["route"]["choice"], max(labels, key=fork.__getitem__))

    def test_five_level_score_matches_reference(self):
        labels = [str(i) for i in range(5)]
        self.assertParity("urgency", labels)
        self.assertLessEqual(abs(self.fork["urgency"]["score"] - self.reference["urgency"]["score"]), TOLERANCE)


if __name__ == "__main__":
    unittest.main()
