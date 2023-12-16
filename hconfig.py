from hparams import Hyperparams

small = Hyperparams(
        batch=8,
        n_layer=2,
        train_len=256,
        n_vocab=16,
        prefix_len=10,
        learn_rate=1e-3,
        d_model=64,
        n_epoch=1000,
        epoch_sz=8192,
        report_every=100)



