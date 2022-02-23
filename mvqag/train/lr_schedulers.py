from torch.optim import Optimizer


__all__ = ['PolyLR']


def lr_poly(base_lr, epoch_num, n_epochs, power=0.9):
    """https://github.com/Rodger-Huang/SYSU-HCP-at-ImageCLEF-VQA-Med-2021/blob/main/utils.py#L296"""
    return base_lr * ((1 - float(epoch_num) / n_epochs) ** power)


class PolyLR(object):
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        n_epochs: int,
        update_interval: int = 10,
        power: float = 0.9,
        verbose: bool = True
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.n_epochs = n_epochs
        self.update_interval = update_interval
        self.power = power
        self.verbose = verbose

        self.lr = base_lr
        self.step(0)

    def get_lr(self):
        return self.lr

    def step(self, epoch_num: int) -> None:
        if epoch_num % self.update_interval == 0:
            self.lr = lr_poly(
                self.base_lr, epoch_num, self.n_epochs, power=self.power
            )
            if self.verbose:
                print(f"Learning rate updated to {self.lr} @ epoch {epoch_num}.")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr