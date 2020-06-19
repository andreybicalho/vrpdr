import numpy as np


class Tokenizer:
    def __init__(self, tokens):
        self.tokens = tokens
        self.SOS_token = 0
        self.EOS_token = 1
        self.UNKNOWN_token = len(tokens) + 2
        self.n_token = len(tokens) + 3

        char_idx = {}

        for i, c in enumerate(tokens):
            char_idx[c] = i + 2

        self.char_idx = char_idx

    def tokenize(self, s):
        label = np.zeros((len(s) + 1,), dtype=np.long)
        label[0] = self.SOS_token

        for i, c in enumerate(s):
            label[i + 1] = self.char_idx.get(c, self.UNKNOWN_token)

        return label

    def translate(self, ts, n=None):
        ret = []

        if n is None:
            n = len(ts)

        for i in range(n):
            t = ts[i]

            if t == self.SOS_token:
                pass
            elif t == self.EOS_token:
                ret.append('-')
            elif t == self.UNKNOWN_token:
                ret.append('?')
            else:
                ret.append(self.tokens[t - 2])

        return ''.join(ret)