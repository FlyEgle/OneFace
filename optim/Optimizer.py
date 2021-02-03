"""
@author: Mingchao Jiang
@date: 2021-02-02
Optimizer for training
"""
import torch.optim as optim


class BuildOptimzer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.optimizer_type = self.cfg.TRAIN.OPTIMIZER
        self.base_lr = self.cfg.TRAIN.BASE_LR
        self.momentum = self.cfg.TRAIN.MOMENTUM
        self.weight_decay = self.cfg.TRAIN.WEIGHT_DECAY

    def optimizer(self, model):

        if self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.base_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "ADAM":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "ADAMW":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return self.optimizer

