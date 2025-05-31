import torch
import math


class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, opt, T_max, eta_min, warmup_init_lr, warmup_epochs=10):
        lr_max = opt.param_groups[0]["lr"]

        def cosine_annealing_with_warmup(epoch):
            if epoch < warmup_epochs:
                warmup_factor = (
                    warmup_init_lr
                    + (lr_max - warmup_init_lr) / (warmup_epochs - 1) * epoch
                ) / lr_max
                return warmup_factor
            else:
                cosine_annealing_factor = (
                    eta_min
                    + 0.5
                    * (lr_max - eta_min)
                    * (1 + math.cos((epoch - warmup_epochs + 1) / T_max * math.pi))
                ) / lr_max
                return cosine_annealing_factor

        super(CosineAnnealingWithWarmupLR, self).__init__(
            opt, cosine_annealing_with_warmup
        )
