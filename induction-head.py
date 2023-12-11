import torch
from torch.optim import Adam
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from random import choice, choices

"""
Recreate Experiment in Table 2, Section 4.1.2 and Appendix E1 
"""

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

    def gen(self):
        """
        Section E.1
        Training consists of randomly generating data every step, with a batch size of 8.
        """
        # prefix is the region where the special token first occurs
        memory_tok = choice(self.vocab)
        ind_pos = choice(range(self.P))
        # Section 3.1 from https://arxiv.org/pdf/2212.14052.pdf the 'special token'
        cadence = [self.ind_tok, memory_tok]
        pre = choices(self.vocab, k=ind_pos)
        noise = choices(self.vocab, k=self.L-ind_pos-2) 
        seq = pre + cadence + noise + cadence 
        return seq

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for _ in range(self.B):
            batch.append(self.gen())
        return torch.tensor(batch).to('cuda')

def main():
    batch = 8
    n_layer = 2
    # epoch_sz = 256 
    epoch_sz = 8192
    train_len = 256 
    # train_len = 23
    n_vocab = 16
    d_model = 64
    prefix_len = 10
    learn_rate = 1e-3
    n_epoch = 1000
    report_every = 100

    data = InductionData(batch, n_vocab, train_len, prefix_len)
    model = MambaLMHeadModel(d_model, n_layer, n_vocab + 1, device='cuda')
    # print(next(model.parameters()).is_cuda)
    # return

    it = iter(data)
    b = next(it)
    # print(b.shape)
    # return

    xent = torch.nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=learn_rate)

    step = 0
    for epoch in range(n_epoch):
        loss_sum = 0
        for b in range(epoch_sz):
            batch = next(it)
            opt.zero_grad()
            out = model(batch[:,:-1]).logits
            pred = out[:,-1,:]
            targ = batch[:,-1]
            # print(pred.shape, targ.shape)
            # return
            # print(out.shape)
            loss = xent(pred, targ)
            loss.backward()
            loss_sum += loss
            opt.step()
            step += 1
            if step % report_every == 0:
                print(f'{step=}, {loss.item():3.3f}')

        # epoch_loss = (loss_sum / epoch_sz).item()
        # print(f'{epoch=}, {epoch_loss=:3.3f}')

if __name__ == '__main__':
    main()


            


