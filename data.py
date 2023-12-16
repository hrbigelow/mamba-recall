import torch as t

class InductionData:
    def __init__(self, batch, n_vocab, seq_len, prefix_len):
        """
        Generates synthetic data of the form:
        ... S M .... S M
        where S is an 'induction token' and M is the token to memorize / recall

        n_vocab: token alphabet size
        seq_len: total sequence length to generate
        prefix_len: region where first S should occur
        """
        assert prefix_len < seq_len - 4
        self.B = batch
        self.V = n_vocab
        self.L = seq_len # total sequence
        self.P = prefix_len # region where first induction token occurs
        self.vocab = list(range(self.V))
        self.ind_tok = self.V

    def __iter__(self):
        return self

    def __next__(self):
        mem = t.randint(0, self.V, (self.B,1))
        batch = t.randint(0, self.V, (self.B, self.L))
        # inds = t.randint(0, self.P, (self.B,1))
        inds = t.full((self.B,1), 5)
        inds2 = t.full((self.B,1), self.L-2)
        batch.scatter_(1, inds, self.ind_tok)
        batch.scatter_(1, inds+1, mem)
        batch.scatter_(1, inds2, self.ind_tok)
        batch.scatter_(1, inds2+1, mem)
        return dict(starts=inds, tokens=batch)


