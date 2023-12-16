import torch as t
import fire
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.selective_scan_interface import MambaInnerFn
from streamvis.logger import DataLogger
from data import InductionData
from funcs import Writer
import hconfig

def main(run_name, logger_path, ckpt_file='model.ckpt', **kwargs):
    hconf = hconfig.small
    hconf.update(kwargs)

    evals = {}
    eval_batch = 300
    print('Generating eval data...', end='')
    for p in range(6, 9):
        l = 2**p
        eval_data = InductionData(eval_batch, hconf.n_vocab, l, hconf.prefix_len)
        evals[p] = next(iter(eval_data))
    print('done.')

    xent = t.nn.CrossEntropyLoss()
    model = MambaLMHeadModel(hconf.d_model, hconf.n_layer, hconf.n_vocab + 1,
                             device='cuda')
    state = t.load(ckpt_file)
    model.load_state_dict(state)
    print(f'Restored model from {ckpt_file}')

    logger = DataLogger(run_name)
    logger.init(path=logger_path)
    writer = Writer(logger)
    MambaInnerFn.add_hook(writer.write)

    eval_loss = 0
    for p, eval_data in evals.items():
        starts = eval_data['starts']
        tokens = eval_data['tokens'].to('cuda')
        writer.prepare(starts, p)
        out = model(tokens[:,:-1]).logits
        pred = out[:,-1,:]
        targ = tokens[:,-1]
        eval_loss += xent(pred, targ)
    eval_loss /= len(evals)
    print(f'Eval: {eval_loss.item():3.3f}')

        # epoch_loss = (loss_sum / epoch_sz).item()
        # print(f'{epoch=}, {epoch_loss=:3.3f}')

if __name__ == '__main__':
    fire.Fire(main)

