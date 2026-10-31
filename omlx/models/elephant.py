# SPDX-License-Identifier: Apache-2.0
"""Bounded text-only qev-labels-v2 Elephant runtime, not a generation path.

Rendering/answers adapted from OmniJev/OneJev@81ce62f (Apache-2.0).
Confidence formulas: MIT License, Copyright (c) 2026 TypeSafe AI,
originally typesafe-ai/system-one-adapter-python; see qev/answers.py.
Two cached blocks follow baselabs/elephant-1's serve.patch. No calibration,
permutation debiasing, vision, alternate templates or truncation is supported.
"""

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .decision import (
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
)

SYSTEM = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EXCLUDED = set(
    "BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ".split()
)
SLOT_LABELS = tuple(LETTERS) + tuple(
    a + b for a in LETTERS for b in LETTERS if a + b not in _EXCLUDED
)[:230]
_BOUNDARY = "QEV_PREFIX_MARK"
_MAX_LAYER_STEP = 4
_MARKERS = (
    "<|vision", "<|image", "<|video", "<image>", "<video>", "<|audio", "<audio>",
    "data:image", "data:video", "data:audio",
)


def _json_value(value):
    """Reject non-JSON, nonfinite and structured media before rendering."""
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, str):
        if any(m in value.lower() for m in _MARKERS):
            raise DecisionRequestError("Elephant supports text only")
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, (dict, list)):
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise DecisionRequestError("JSON object keys must be strings")
            for key in value:
                _json_value(key)  # JSON keys and choice names are model-visible text too.
            if any(k.lower() in {
                "image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"
            } for k in value):
                raise DecisionRequestError("Structured media is unsupported")
            media_type = value.get("type")
            if isinstance(media_type, str) and media_type.lower() in (
                "image", "image_url", "video", "video_url", "audio", "audio_url",
                "input_image", "input_video", "input_audio", "output_audio"
            ):
                raise DecisionRequestError("Structured media is unsupported")
            value = list(value.values())
        for item in value:
            _json_value(item)
        return
    raise DecisionRequestError("Expected finite JSON text data")


def _entry(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class Question:
    qid: str
    kind: str
    labels: tuple[str, ...]
    descriptions: tuple
    suffix: str


@dataclass(frozen=True)
class Plan:
    questions: tuple[Question, ...]
    prefix: tuple[int, ...]
    suffix: tuple[int, ...]


def render_question(qid, question):
    """Validate the restricted question and freeze original insertion order."""
    if not isinstance(qid, str) or not qid.strip() or not isinstance(question, dict):
        raise DecisionRequestError("Question needs a nonempty string ID and object")
    if set(question) - {"type", "instructions", "criteria"}:
        raise DecisionRequestError("Unsupported question fields")
    kind = question.get("type")
    instructions = question.get("instructions")
    criteria = question.get("criteria")
    for entry in (instructions, criteria):
        _json_value(entry)
    if instructions is not None and not isinstance(instructions, (str, dict, list)):
        raise DecisionRequestError("Instructions must be a string, object, array or null")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 3:
            raise DecisionRequestError("Choice requires 2–3 options")
        labels, names, desc = list(criteria), list(criteria), list(criteria.values())
        default = "Which option applies to the state?"
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 3:
            raise DecisionRequestError("Score requires 2–3 levels")
        labels = [str(i) for i in range(len(criteria))]
        names, desc = [f"level {i}" for i in labels], criteria
        default = "Which level describes the state?"
    elif kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
        ):
            raise DecisionRequestError("Noul criteria permit only true/false")
        criteria = criteria or {}
        labels = names = ["yes", "no"]
        desc = [criteria.get("true"), criteria.get("false")]
        defaults = (
            "the statement is true / the answer is yes",
            "the statement is false / the answer is no",
        )
        desc = [d if d is not None else f for d, f in zip(desc, defaults)]
        default = "Is the statement true, or is the answer to the question yes?"
    else:
        raise DecisionRequestError("Unsupported Elephant question kind")
    if (
        any(not isinstance(k, str) or not k.strip() for k in labels)
        or len(set(labels)) != len(labels)
    ):
        raise DecisionRequestError("Option labels must be unique nonempty strings")
    if any(d is not None and not isinstance(d, (str, dict, list)) for d in desc):
        raise DecisionRequestError("Descriptions must be strings, objects, arrays or null")
    options = []
    for slot, name, description in zip(SLOT_LABELS, names, desc):
        text = _entry(description)
        options.append(f"{slot}. {name}" + (f": {text}" if text else ""))
    header = "Rate the state:" if kind == "score" else "Question:"
    suffix = (
        f"{header} {_entry(instructions) or default}\n\nOptions:\n"
        + "\n".join(options)
        + f"\n\nAnswer with one letter: {', '.join(SLOT_LABELS[:len(labels)])}."
    )
    return Question(qid, kind, tuple(labels), tuple(copy.deepcopy(desc)), suffix)


def messages(state, suffix):
    text = state if isinstance(state, str) else (_entry(state) if state is not None else "")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"<state>\n{text}\n</state>\n\n{suffix}"},
    ]


def assemble(question, probabilities):
    """Strict complete vector; original six-place wire rounding only here.

    Unit-mass tolerance is 1e-6. Never repair zero mass or fabricate a vector.
    The public noul wire shape intentionally retains only P(yes).
    """
    labels = question.labels
    if (
        question.kind not in {"choice", "score", "noul"}
        or not 2 <= len(labels) <= 3
        or len(question.descriptions) != len(labels)
        or any(not isinstance(k, str) or not k.strip() for k in labels)
        or len(set(labels)) != len(labels)
        or (question.kind == "noul" and labels != ("yes", "no"))
        or (question.kind == "score" and labels != tuple(map(str, range(len(labels)))))
    ):
        raise DecisionRequestError("Invalid response slot labels")
    if isinstance(probabilities, dict):
        if set(probabilities) != set(labels):
            raise DecisionRequestError("Probability keys do not match slots")
        probabilities = [probabilities[k] for k in labels]
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != len(labels):
        raise DecisionRequestError("Incomplete probability vector")
    if any(
        type(p) not in (int, float) or not 0 <= p <= 1 or not math.isfinite(p)
        for p in probabilities
    ):
        raise DecisionRequestError("Invalid probability")
    total = math.fsum(probabilities)
    if total <= 0 or abs(total - 1) > 1e-6:
        raise DecisionRequestError("Probability mass must be one (tolerance 1e-6)")
    p = [v / total for v in probabilities]
    mode = max(range(len(p)), key=p.__getitem__)
    if question.kind == "noul":
        return {"type": "noul", "noul": round(p[0], 6)}
    result = {
        "type": question.kind,
        "probabilities": dict(zip(labels, [round(x, 6) for x in p])),
    }
    if question.kind == "choice":
        confidence = (max(p) - 1 / len(p)) / (1 - 1 / len(p))
        result.update(choice=labels[mode], confidence=round(confidence, 6))
    else:
        center = (len(p) - 1) / 2
        uniform_mad = sum(abs(i - center) for i in range(len(p))) / len(p)
        distance = sum(pi * abs(i - mode) for i, pi in enumerate(p))
        result.update(
            score=round(sum(i * pi for i, pi in enumerate(p)), 6),
            legend=dict(zip(labels, question.descriptions)),
            confidence=round(max(0, 1 - distance / uniform_mad), 6),
        )
    return result


def softmax(logits, size):
    if (
        not isinstance(logits, (list, tuple)) or size not in (2, 3)
        or len(logits) != size
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)
    ):
        raise DecisionRequestError("Invalid slot logits")
    maximum = max(logits)
    exps = [math.exp(v - maximum) for v in logits]
    total = math.fsum(exps)
    return [v / total for v in exps]


class ElephantModel:
    has_vision = False

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.backbone = None
        self.slot_rows = None
        self.slot_ids = ()

    def _contract(self):
        path = Path(self.model_path)
        try:
            config = json.loads((path / "config.json").read_text())
            readout = json.loads((path / "readout.json").read_text())
            expected = {"readout": "slot_lm", "head_dim": 256, "n_slots": 256}
            overrides = dict(config["modeling_overrides"])
            attention = overrides.pop("attention")
            text = config["text_config"]
            valid = (
                config["model_type"] == "qwen3_5"
                and text["model_type"] == "qwen3_5_text"
                and attention == {
                    "full_attention": "block_bidirectional", "linear_attention": "causal"
                }
                and overrides == expected and readout == expected
                and all(
                    type(meta[k]) is int
                    for meta in (overrides, readout) for k in ("head_dim", "n_slots")
                )
                and config.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
                and config.get("vision_config") is None
                and config.get("audio_config") is None
                and isinstance(text.get("layer_types"), list)
                and set(text["layer_types"]) == {"linear_attention", "full_attention"}
                and len(text["layer_types"]) == text.get("num_hidden_layers")
                and type(text["vocab_size"]) is int and text["vocab_size"] >= 256
                and type(text["hidden_size"]) is int and text["hidden_size"] > 0
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise DecisionRequestError("Invalid Elephant config/readout contract") from exc
        if not valid or self.trust_remote_code or (path / "calibration.json").exists():
            raise DecisionRequestError("Unsupported Elephant config, calibration or remote code")
        return text

    def load(self):
        self.close()
        text = self._contract()  # All file/format guards precede loader/global patches.
        backbone = DecisionBackbone(self.model_path, trust_remote_code=False)
        try:
            backbone.load()
            if backbone.has_vision:
                raise DecisionRequestError("Elephant requires a text-only backbone")
            ids = []
            for label in SLOT_LABELS:
                encoded = backbone.tokenizer.encode(label, add_special_tokens=False)
                if (
                    len(encoded) != 1 or type(encoded[0]) is not int
                    or not 0 <= encoded[0] < text["vocab_size"]
                    or backbone.tokenizer.decode(encoded) != label
                ):
                    raise DecisionRequestError(
                        "All 256 slots must round-trip as single vocabulary tokens"
                    )
                ids.append(encoded[0])
            if len(set(ids)) != 256:
                raise DecisionRequestError("Slot token IDs collide")
            rows = backbone.output_rows(mx.array(ids)).astype(mx.float32)
            mx.eval(rows)
            if rows.shape != (256, text["hidden_size"]) or any(
                not math.isfinite(v) for row in rows.tolist() for v in row
            ):
                raise DecisionRequestError("Invalid FP32 slot readout rows")
        except BaseException:
            backbone.close()
            raise
        self.backbone, self.slot_rows, self.slot_ids = backbone, rows, tuple(ids)

    def close(self):
        if self.backbone is not None:
            self.backbone.close()
        self.backbone = self.slot_rows = None
        self.slot_ids = ()

    def encode(self, request: dict, truncate: bool = True):
        if self.backbone is None:
            raise RuntimeError("Elephant is not loaded")
        allowed = {"model", "state", "questions", "images", "videos", "truncate"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise DecisionRequestError("Unsupported request fields")
        if any(request.get(k) not in (None, []) for k in ("images", "videos")):
            raise DecisionRequestError("Elephant supports text only")
        questions = request.get("questions")
        if not isinstance(questions, dict) or len(questions) != 1:
            raise DecisionRequestError("Elephant requires exactly one question")
        question = render_question(*next(iter(questions.items())))
        state = request.get("state")
        _json_value(state)
        rendered = messages(state, question.suffix)
        if sum(len(m["content"]) for m in rendered) > 2048:
            raise DecisionContextLengthError(
                "Elephant exceeds 2048 rendered text characters; no truncation"
            )
        tokenizer = self.backbone.tokenizer

        def template(msg):
            return tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

        full = template(rendered)
        boundary = template(messages(state, _BOUNDARY))
        if full.count(_BOUNDARY) or boundary.count(_BOUNDARY) != 1:
            raise DecisionRequestError("Ambiguous state boundary sentinel")
        ids = tokenizer.encode(full, add_special_tokens=False)
        limit = min(512, getattr(self.backbone, "max_position_embeddings", None) or 512)
        if not ids or len(ids) > limit:
            raise DecisionContextLengthError(
                f"Elephant exceeds {limit} chat-template tokens; no truncation"
            )
        prefix = tokenizer.encode(
            boundary[:boundary.index(_BOUNDARY)], add_special_tokens=False
        )
        n = min(len(prefix), len(ids))
        while n and prefix[:n] != ids[:n]:
            n -= 1
        if n == 0 or n == len(ids):
            raise DecisionRequestError("Empty prefix/suffix at state boundary")
        self._block_depth(n)
        self._block_depth(len(ids) - n)
        return Plan((question,), tuple(ids[:n]), tuple(ids[n:]))

    def _block_depth(self, length):
        """Positive integer progress needs at least one token per bounded group."""
        depth = len(self.backbone._text_model.layers)
        minimum = (depth + _MAX_LAYER_STEP - 1) // _MAX_LAYER_STEP
        if not depth or length < minimum:
            raise DecisionRequestError(
                f"Each Elephant block needs at least {minimum} tokens for bounded layer steps"
            )
        return depth

    def _block(self, ids, cache, chunk_size):
        text = self.backbone._text_model
        layers, length = text.layers, len(ids)
        depth = self._block_depth(length)
        if len(cache) != depth:
            raise DecisionRequestError("Layer/cache depth mismatch")
        hidden = text.embed_tokens(mx.array([ids]))
        done = credited = 0
        try:
            while done < depth:
                # This minimum guarantees credit on EVERY group, so one next()
                # cannot silently evaluate several groups while waiting for a token.
                group = max(
                    (depth + length - 1) // length,
                    min(_MAX_LAYER_STEP, max(1, chunk_size() * depth // length)),
                )
                end = min(depth, done + group)
                for index in range(done, end):
                    hidden = layers[index](hidden, mask=None, cache=cache[index])
                mx.eval(hidden, [c.state for c in cache])
                progress = length * end // depth
                yield progress - credited
                credited = progress
                done = end
            return text.norm(hidden)
        finally:
            hidden = None

    def run(self, plan, chunk_size):
        if self.backbone is None or self.slot_rows is None:
            raise RuntimeError("Elephant is not loaded")
        self._block_depth(len(plan.prefix))
        self._block_depth(len(plan.suffix))
        base = cache = hidden = logits = None
        try:
            base = self.backbone.make_cache()
            hidden = yield from self._block(plan.prefix, base, chunk_size)
            cache = copy.deepcopy(base)
            hidden = yield from self._block(plan.suffix, cache, chunk_size)
            question = plan.questions[0]
            logits = self.slot_rows[:len(question.labels)] @ hidden[0, -1].astype(mx.float32)
            mx.eval(logits)
            probabilities = softmax(logits.tolist(), len(question.labels))
            return {
                "answers": {question.qid: assemble(question, probabilities)},
                "input_tokens": len(plan.prefix) + len(plan.suffix),
            }
        finally:
            if cache is not None:
                cache.clear()
            if base is not None:
                base.clear()
            base = cache = hidden = logits = None
