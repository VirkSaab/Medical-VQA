import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from pathlib import Path
from mvqag import CNF_PATH
from mvqag.utils import load_yaml, get_recent_githash, load_qa_file
from mvqag.data import VQADataset, DataLoaders
from mvqag.model import vgg16HGap
from mvqag.train import (
    get_device,
    get_metrics,
    LabelSmoothingCrossEntropyWithSuperLoss,
    SuperLoss,
    VQATrainer,
    TrainingLogger,
    mixup_data,
    mixup_criterion
)


class SYSUDataset(VQADataset):
    def __init__(self, df, img_tfms, n_classes) -> None:
        super().__init__(df, img_tfms, classes=None)
        self.n_classes = n_classes

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img = Image.open(row.PATH)

        if self.img_tfms is not None:
            img = self.img_tfms(img)

        A = torch.tensor(row.A)
        return img, A


class PrepareSYSUData:
    def __init__(self, CNF) -> None:
        self.n_classes = CNF.data.n_classes

        # Load data as dataframes
        self.train_df = pd.read_table(
            CNF.paths.clef_sysu_train_qa,
            sep=' ',
            header=None
        )
        self.train_df.columns = ['ID', 'A']  # Image ID and class
        self.train_df['PATH'] = self.train_df.ID.apply(
            lambda x: f"{CNF.paths.clef_sysu_train_imgs}/{x}.jpg"
        )
        self.val_df = pd.read_table(
            CNF.paths.clef_sysu_val_qa,
            sep=' ',
            header=None
        )
        self.val_df.columns = ['ID', 'A']  # Image ID and class
        self.val_df['PATH'] = self.val_df.ID.apply(
            lambda x: f"{CNF.paths.clef_sysu_val_imgs}/{x.split('/')[-1]}"
        )
        # Augmentation
        self.train_tfms = T.Compose([
            T.Resize(size=(CNF.model.inp_size + 8, CNF.model.inp_size + 8)),
            T.RandomCrop(size=(CNF.model.inp_size, CNF.model.inp_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        self.val_tfms = T.Compose([
            T.Resize(size=(CNF.model.inp_size, CNF.model.inp_size)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        # Make dataset classes
        self.trainset = SYSUDataset(
            self.train_df, self.train_tfms, self.n_classes)
        self.valset = SYSUDataset(self.val_df, self.val_tfms, self.n_classes)
        # Make dataloaders
        self.dls = DataLoaders.from_dataset(
            trainset=self.trainset,
            train_bs=CNF.train.bs,
            valset=self.valset,
            val_bs=1
        )

    def check(self):
        print(f"Data augmentation:\n\t{self.train_tfms}\n\t{self.val_tfms}")
        print(f"# training samples = {self.train_df.shape}")
        print(f"# validation samples = {self.val_df.shape}")
        assert self.train_df.shape == (5683, 3)
        assert self.val_df.shape == (500, 3)
        assert self.train_df.A.nunique() <= self.n_classes
        assert self.val_df.A.nunique() <= self.n_classes
        assert len(self.trainset) == self.train_df.shape[0]
        assert len(self.valset) == self.val_df.shape[0]

        inputs, target = next(iter(self.dls.trainloader))
        print(f"data batch:")
        print(f"\timgs = {inputs.shape}")
        print(f"\tA = {target.shape}")
        print("data check: [green]PASSED[/green]")


class SYSUTrainer(VQATrainer):
    def __init__(self, dls, net, loss_fn, optm_fn, device,
                 metrics=None,
                 checkpointer=None,
                 logger=None,
                 step_lrs=None,
                 epoch_lrs=None,
                 mixup: float = 0.):
        super().__init__(dls, net, loss_fn, optm_fn, device, metrics,
                         checkpointer, logger, step_lrs, epoch_lrs)
        self.mixup = mixup

    def train_one_batch(self, batch):
        inputs, target = batch
        inputs, target = inputs.to(self.device), target.to(self.device)
        if self.mixup > 0.0:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, target, alpha=self.mixup, use_cuda=True
            )
            output = self.net(inputs)
            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam
            )
        else:
            output = self.net(inputs)
            loss = self.loss_fn(output, target)

        # Backpropagation
        self.optm_fn.zero_grad()
        loss.backward()
        self.optm_fn.step()
        return loss, output, target

    def val_one_batch(self, batch):
        inputs, target = batch
        inputs, target = inputs.to(self.device), target.to(self.device)
        output = self.net(inputs)
        # test time augmentation
        assert inputs.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
        inputs_flip = torch.flip(inputs, [3])
        output_flip = self.net(inputs_flip)
        output_avg = (output + output_flip) / 2.0
        loss = self.loss_fn(output_avg, target)
        return loss, output_avg, target


def run(CNF: dict):
    # ---------------------------------------- SETTINGS:
    # Set seed
    torch.manual_seed(CNF.seed)
    torch.cuda.manual_seed(CNF.seed)
    torch.cuda.manual_seed_all(CNF.seed)

    # Set device
    CNF.device, CNF.cuda_ids = get_device()
    # Get the githash of last commit
    CNF.recent_githash = get_recent_githash()

    # ---------------------------------------- DATA:
    data = PrepareSYSUData(CNF)
    data.check()

    # ---------------------------------------- MODEL:
    print(f"Loading {CNF.model.vnet_name} model...", end=' ')
    CNF.wandb_run_name += f"+{CNF.model.vnet_name}"
    if CNF.model.vnet_name == 'vgg16':
        model = eval(f"tv.models.{CNF.model.vnet_name}(pretrained=True)")
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, CNF.data.n_classes)
        )
    elif CNF.model.vnet_name == 'vgg16hgap':
        model = vgg16HGap(pretrained=True,
                          num_classes=CNF.data.n_classes,
                          rank=CNF.device)
    else:
        raise NotImplementedError(
            f'Model {CNF.model.vnet_name} not supported.')
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(CNF.device)
    model = torch.nn.DataParallel(model, device_ids=CNF.cuda_ids)
    print('done.')

    # ---------------------------------------- TRAIN:
    optm_fn = torch.optim.SGD(model.parameters(),
                              lr=CNF.optm.lr,
                              momentum=CNF.optm.mom,
                              weight_decay=CNF.optm.wd,
                              )
    if CNF.loss.fn.lower() == 'lscesl':
        loss_fn = LabelSmoothingCrossEntropyWithSuperLoss(rank=CNF.device)
    elif CNF.loss.fn.lower() == 'superloss':
        loss_fn = SuperLoss(rank=CNF.device)
    elif CNF.loss.fn.lower() == 'crossentropy':
        loss_fn = nn.CrossEntropyLoss(
            reduction='mean',
            label_smoothing=CNF.loss.smoothing
        )
    else:
        raise NotImplementedError(f'Loss_fn {CNF.loss.fn} not supported.')
    print(f"Criterion = {CNF.loss.fn}")
    epoch_lrs = torch.optim.lr_scheduler.StepLR(optm_fn,
                                                step_size=20,
                                                gamma=0.60)
    logger = TrainingLogger(
        logs_dir=CNF.paths.logs_dir, config=CNF, run_name=CNF.wandb_run_name
    )
    trainer = SYSUTrainer(dls=data.dls,
                          net=model,
                          loss_fn=loss_fn,
                          optm_fn=optm_fn,
                          device=CNF.device,
                          metrics=get_metrics(CNF.data.n_classes),
                          logger=logger,
                          checkpointer=None,
                          epoch_lrs=epoch_lrs,
                          mixup=CNF.train.mixup
                          )
    trainer.train(CNF.train.n_epochs)


if __name__ == '__main__':
    # * EXPERIMENT NUMBER
    EXP_NO = 4

    # * Common settings
    CNF = load_yaml(CNF_PATH)
    CNF.seed = 1234
    CNF.data.n_classes = 330
    CNF.model.vnet_name = 'vgg16'  # Without batchnorm
    CNF.loss.smoothing = 0.  # Disable label smoothing
    CNF.optm.lr = 1e-3
    CNF.optm.wd = 5e-4
    CNF.optm.mom = 0.9
    CNF.train.bs = 16
    CNF.train.n_epochs = 60
    CNF.train.mixup = 0.0  # Disable mixup
    CNF.wandb_run_name = '/'.join(__file__.split('.')[0].split('/')[-2:])
    CNF.wandb_run_name = f"{EXP_NO}-{CNF.wandb_run_name}"
    
    if EXP_NO == 1:
        # * EXP 1 - our baseline with SYSU curated dataset
        # Baseline experiment
        pass

    elif EXP_NO == 2:
        # * EXP 2 - add mixup to exp 1
        CNF.train.mixup = 0.2
        CNF.wandb_run_name += '+Mixup'

    elif EXP_NO == 3:
        # * EXP 3 - add label smoothing to exp 2
        CNF.train.mixup = 0.2
        CNF.loss.smoothing = 0.1
        CNF.wandb_run_name += '+Mixup+LabelSmoothing'

    elif EXP_NO == 4:
        # * EXP 4 - add HGap and SuperLoss to exp 3
        CNF.train.mixup = 0.2
        CNF.loss.smoothing = 0.1
        CNF.model.vnet_name = 'vgg16HGap'
        CNF.loss.fn = 'lscesl'
        CNF.wandb_run_name += '+Mixup+LabelSmoothing+vgg16HGap+SuperLoss'

    print(f"[cyan]Running Exp `{CNF.wandb_run_name}`...")
    run(CNF)
    print(f"[green]Done.")
