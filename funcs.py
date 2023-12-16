from einops import rearrange
import torch.nn.functional as F

class Writer:
    def __init__(self, logger):
        self.logger = logger

    def prepare(self, induction_pos, length_power):
        """
        Update information for next write
        induction_pos: b (contain index of first induction token in inputs)
        length_power: log_2 length of the synthetic data
        """
        self.induction_pos = induction_pos
        self.length_power = length_power
        self.write_count = 0

    def write(self, delta, **kwargs):
        """
        delta: b d l (before softplus)
        """
        layer = self.write_count
        group_fn = lambda pfx: f'{pfx}-len{self.length_power}-layer{layer}'
        # perform position adjustment
        group_name = group_fn('delta')
        y_offset = 5.0 * layer 
        delta = F.softplus(delta).to('cpu')
        # delta = delta.mean(axis=0)
        length = 2**self.length_power-1
        xs = list(range(length))
        # b=0 single sample per channel
        self.logger.write(group_fn('delta-batch0'), x=xs, y=delta[0,:,:] + y_offset)
        # mean over samples
        self.logger.write(group_fn('delta-mean'), x=xs, y=delta.mean(axis=0) + layer)

        # all samples from a single channel
        self.logger.write(group_fn('delta-channel0'), x=xs, y=delta[:,0,:] + y_offset)

        self.write_count += 1

        # group_name = group_fn('delta.adj')
        # here must truncate
        # self.logger.write(group_name, x=None, y=None)



