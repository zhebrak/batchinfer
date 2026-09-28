"""A whitespace tokenizer with the two calls analysis and the engine make, counting them."""


class StubTokenizer:
    name_or_path = "stub"
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0}
        self.calls = 0

    def _ids(self, text):
        return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]

    def __call__(self, texts, add_special_tokens=False):
        self.calls += 1
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [self._ids(t) for t in texts]}

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        return self._ids(text)

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, str(i)) for i in ids if not (skip_special_tokens and i == 0))
