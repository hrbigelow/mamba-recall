import fire
import torch as t
from torch.optim import Adam
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.selective_scan_interface import MambaInnerFn
from data import InductionData
from streamvis.logger import DataLogger
import hconfig

"""
Recreate Experiment in Table 2, Section 4.1.2 and Appendix E1 
"""

def main(run_name, logger_path, d_model=64, n_epoch=1000, epoch_sz=8192,
         report_every=100, out_file='model.ckpt'):

    hps = hconfig.small
    hps.d_model = d_model
    hps.n_epoch = n_epoch
    hps.epoch_sz = epoch_sz
    hps.report_every = report_every

    data = InductionData(hps.batch, hps.n_vocab, hps.train_len, hps.prefix_len)
    model = MambaLMHeadModel(hps.d_model, hps.n_layer, hps.n_vocab + 1,
                             device='cuda')

    logger = DataLogger(run_name)
    logger.init(path=logger_path)

    it = iter(data)
    xent = t.nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=hps.learn_rate)

    step = 0
    for epoch in range(n_epoch):
        loss_sum = 0
        for b in range(epoch_sz):
            batch = next(it)
            tokens = batch['tokens'].to('cuda')
            opt.zero_grad()
            out = model(tokens[:,:-1]).logits
            pred = out[:,-1,:]
            targ = tokens[:,-1]
            # print(pred.shape, targ.shape)
            # return
            # print(out.shape)
            loss = xent(pred, targ)
            loss.backward()
            loss_sum += loss
            opt.step()
            logger.write('loss', x=step, y=loss.item())
            step += 1
            if step % report_every == 0:
                print(f'{epoch=}, {step=}, {loss.item():3.3f}')

    logger.shutdown()

    state = model.state_dict()
    print(f'Saving to {out_file}')
    t.save(state, out_file)

    # finished training

if __name__ == '__main__':
    fire.Fire(main)

