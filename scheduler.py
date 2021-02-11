from typing import Tuple
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

class TriangleLR(nn.Module):
    def __init__(self, optimizer, epoch_size:int, knots:Tuple[int, int, int],
                 values:Tuple[float, float, float], *args, **kwargs):
        '''
        Code is based on https://github.com/universome/loss-patterns/blob/master/src/optims/triangle_lr.py
        '''
        self.epoch_size = epoch_size
        self.knots = knots
        self.values = values

        super(TriangleLR, self).__init__()

    def triangle_scheduler(self, it:int, epoch:int):
        step = epoch + (it / self.epoch_size)

        if step > self.knots[-1]:
            return self.values[-1]
        else:
            return np.interp([step], self.knots, self.values)[0]