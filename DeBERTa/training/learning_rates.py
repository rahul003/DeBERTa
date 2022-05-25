# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learning rate decay functions."""

import math
import smdistributed.modelparallel.torch as smp


class LR(object):
    """Anneals the learning rate."""

    def __init__(self, optimizer, start_lr,
                 warmup, total_iters,
                 decay_style, last_iter, min_lr=0.0):

        # Class values.
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.warmup = warmup
        self.num_iters = last_iter
        self.end_iter = total_iters
        assert self.end_iter > 0
        self.decay_style = decay_style
        # Set the learning rate
        self.step(self.num_iters)

        if smp.rank() == 0:
            print('Learning rate decay style: {}'.format(self.decay_style))

    def warmup_linear(self, step, total, warmup, ends=0):
        x = step / total
        x = x - int(x)
        if x < warmup:
            return x / warmup
        return (1 - ends) * (1.0 - x) + ends


    def step(self, step_num=None):
        """Set lr for all parameters groups."""
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.warmup_linear(self.num_iters, self.end_iter, self.warmup)
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr * self.start_lr
