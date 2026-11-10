"""Regression: Elephant's token limit is the model's configured context length.

Before the fix ``ElephantModel.encode`` capped prompts at
``min(512, max_position_embeddings)``, so a 262,144-token model rejected any
prompt over 512 chat-template tokens with HTTP 413. Stdlib unittest on the
shared fake backbone (char-per-token tokenizer); no real model or MLX.
"""

import unittest

import tests.test_elephant as harness


class ElephantContextLimitTests(unittest.TestCase):
    setUp = harness.ElephantTests.setUp
    tearDown = getattr(harness.ElephantTests, "tearDown", lambda self: None)
    write_contract = harness.ElephantTests.write_contract
    request = harness.ElephantTests.request

    def long_request(self, chars=1500):
        request = self.request()
        request["state"] = "s" * chars  # under the separate 2,048-char cap
        return request

    def tokens(self, request):
        return len(self.model.encode(request).prefix + self.model.encode(request).suffix)

    def test_prompt_over_512_tokens_is_accepted_at_configured_limit(self):
        self.model.load()
        self.assertEqual(self.backbone.max_position_embeddings, 262144)
        plan = self.model.encode(self.long_request())
        self.assertGreater(len(plan.prefix + plan.suffix), 512)

    def test_prompt_above_configured_limit_still_rejected(self):
        self.model.load()
        needed = self.tokens(self.long_request())
        self.backbone.max_position_embeddings = needed
        self.model.encode(self.long_request())  # exactly at the limit passes
        self.backbone.max_position_embeddings = needed - 1
        with self.assertRaises(self.runtime.DecisionContextLengthError) as caught:
            self.model.encode(self.long_request())
        self.assertIn(f"model limit is {needed - 1}", str(caught.exception))

    def test_unknown_limit_fails_closed(self):
        self.model.load()
        for bad in (None, 0, -1, True, "262144", 2.5):
            self.backbone.max_position_embeddings = bad
            with self.assertRaises(self.runtime.DecisionContextLengthError):
                self.model.encode(self.request())


if __name__ == "__main__":
    unittest.main()
