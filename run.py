import torch as t
import fire
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.selective_scan_interface import MambaInnerFn
from streamvis.logger import DataLogger
from data import InductionData
from funcs import Writer, Dump
import hconfig

def main(run_name, logger_path, ckpt_file='model.ckpt', delta_stub='delta.{}.{}.pt',
         **kwargs):
    hps = hconfig.small
    hps.update(kwargs)

    evals = {}
    eval_batch = 50
    print('Generating eval data...', end='')
    for p in range(6, 9):
        l = 2**p
        eval_data = InductionData(eval_batch, hps.n_vocab, l, hps.prefix_len,
                                  hps.ind_pos)
        evals[p] = next(iter(eval_data))
    print('done.')

    xent = t.nn.CrossEntropyLoss()
    model = MambaLMHeadModel(hps.d_model, hps.n_layer, hps.n_vocab + 1,
                             device='cuda')
    state = t.load(ckpt_file)
    model.load_state_dict(state)
    print(f'Restored model from {ckpt_file}')

    logger = DataLogger(run_name)
    logger.init(path=logger_path)
    # writer = Writer(logger)
    writer = Dump(delta_stub)
    MambaInnerFn.add_hook(writer.write)

    eval_loss = 0
    for p, eval_data in evals.items():
        starts = eval_data['starts']
        tokens = eval_data['tokens'].to('cuda')
        writer.prepare(starts, p)
        out = model(tokens).logits
        pred = out[:,-1,:]
        targ = tokens[:,-1]
        eval_loss += xent(pred, targ)
    eval_loss /= len(evals)
    print(f'Eval: {eval_loss.item():3.3f}')

        # epoch_loss = (loss_sum / epoch_sz).item()
        # print(f'{epoch=}, {epoch_loss=:3.3f}')

if __name__ == '__main__':
    fire.Fire(main)

