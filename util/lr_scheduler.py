# noinspection PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR


# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            if not self.finished:
                self.finished = True
            return self.after_scheduler.get_last_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            self.after_scheduler.step()
        else:
            super(GradualWarmupScheduler, self).step()

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


def get_scheduler(optimizer, n_iter_per_epoch, args):
    if "cosine" in args.lr_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,
            T_max=(args.epochs - args.warmup_epoch) * n_iter_per_epoch)
    elif "step" in args.lr_scheduler:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=args.lr_decay_rate,
            milestones=[(m - args.warmup_epoch) * n_iter_per_epoch for m in args.lr_decay_epochs])
    else:
        raise NotImplementedError("scheduler {} not supported".format(args.lr_scheduler))

    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=args.warmup_multiplier,
        after_scheduler=scheduler,
        warmup_epoch=args.warmup_epoch * n_iter_per_epoch)
    return scheduler
