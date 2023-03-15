from torch.optim import Optimizer

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.


        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class InverseSquareRootScheduler(_LRScheduler):
    """
    Code adaped from
        https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py

    Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:
        lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
         decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, optimizer, warmup_init_lrs, num_warmup_steps, num_steps, target_lrs=None, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        if target_lrs is None:
            target_lrs = [0 for _ in self.base_lrs]
        assert(len(self.base_lrs) == len(warmup_init_lrs) == len(num_warmup_steps) == len(target_lrs))
        self.num_steps = num_steps
        self.warmup_init_lrs = warmup_init_lrs
        self.num_warmup_steps = num_warmup_steps
        self.target_lrs = target_lrs
        self.lr_linear_steps = [((base_lr - warmup_init_lr) / num_warmup_step)
                                for base_lr, warmup_init_lr, num_warmup_step in
                                    zip(self.base_lrs, self.warmup_init_lrs, self.num_warmup_steps)]
        self.decay_bases = [(base_lr * num_warmup_step ** 0.5)
                              for base_lr, num_warmup_step in
                                zip(self.base_lrs, self.num_warmup_steps)]
        if target_lrs is None:
            self.offset_factors = [0 for _ in self.base_lrs]
        else:
            self.offset_factors = [(decay_base * self.num_steps ** -0.5 - target_lr) / self.num_steps
                                     for decay_base, target_lr in
                                        zip(self.decay_bases, self.target_lrs)]
        self.step(last_epoch + 1)

    def get_lr(self):
        return [self.update_lr(warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor)
                for warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor in
                    zip(self.warmup_init_lrs, self.num_warmup_steps, self.lr_linear_steps, self.decay_bases,
                        self.offset_factors)]

    def update_lr(self, warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor):
        num_steps = (self.last_epoch + 1)
        if self.last_epoch < num_warmup_step:
            lr = warmup_init_lr + num_steps * lr_linear_step
        else:
            lr = decay_base * num_steps ** -0.5 - offset_factor * num_steps
        return lr