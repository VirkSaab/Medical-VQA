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


class CLEFDataset(VQADataset):
    def __init__(self, df, img_tfms, classes) -> None:
        super().__init__(df, img_tfms, classes=classes)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img = Image.open(row.PATH)

        if self.img_tfms is not None:
            img = self.img_tfms(img)

        A = torch.tensor(self.ctol(row.A))
        return img, A


class PrepareCLEFData:
    def __init__(self, CNF) -> None:
        """
        This class loads data for ImageCLEF 2021 competition
        results comparision. The training set contains selected
        abnormality-related questions from 2019's train, validation, 
        and test sets, 2020's train, and validation set. The 
        validation set contains 2021's validation data only.
        Note that the Yes/No abnormality questions were also
        dropped to fairly compare with other papers. The main
        dataset is the 2020's training set.
        """
        self.n_classes = CNF.data.n_classes

        # TRAINING DATA
        # This is the main dataset for training
        train20_df = self._make_dataframe(
            columns=CNF.clef_cols.default,
            qa_path=CNF.paths.clef_20_train_qa,
            imgs_path=CNF.paths.clef_20_train_imgs,
            is_main_df=True
        )

        # Get categorical abnormality classes
        self.classes = train20_df.A.unique().tolist()
        if 'no' in self.classes:
            self.classes.remove('no')
        if 'yes' in self.classes:
            self.classes.remove('yes')
        self.classes = sorted(self.classes)
        assert len(self.classes) == self.n_classes
        # Remove yes/no classes from train20_df
        train20_df = pd.DataFrame([
            row for row in train20_df.itertuples() if row.A in self.classes
        ]).drop("Index", axis=1)

        # Filter abnormality data from other ImageCLEF datasets
        train19_df = self._make_dataframe(
            columns=CNF.clef_cols.default,
            qa_path=CNF.paths.clef_19_train_qa,
            imgs_path=CNF.paths.clef_19_train_imgs
        )
        val19_df = self._make_dataframe(
            columns=CNF.clef_cols.default,
            qa_path=CNF.paths.clef_19_val_qa,
            imgs_path=CNF.paths.clef_19_val_imgs
        )
        test19_df = self._make_dataframe(
            columns=CNF.clef_cols.test19,
            qa_path=CNF.paths.clef_19_test_qa,
            imgs_path=CNF.paths.clef_19_test_imgs
        )
        test19_df = test19_df.drop('Task', axis=1)
        val20_df = self._make_dataframe(
            columns=CNF.clef_cols.default,
            qa_path=CNF.paths.clef_20_val_qa,
            imgs_path=CNF.paths.clef_20_val_imgs
        )
        # test20_df = self._make_dataframe(
        #     columns=CNF.clef_cols.test20A,
        #     qa_path=CNF.paths.clef_20_test_qa_sysu,
        #     imgs_path=CNF.paths.clef_20_test_imgs
        # )
        training_dfs = [  # Filter abnormality data
            _df[
                _df.Q.str.contains('normal') |
                _df.Q.str.contains('alarming') |
                _df.Q.str.contains('wrong')
            ]
            for _df in [
                train19_df, val19_df, test19_df,
                train20_df, val20_df,  # test20_df
            ]
        ]
        self.train_df = pd.concat(
            training_dfs, ignore_index=True
        ).reset_index(drop=True)

        # VALIDATION DATA
        self.val_df = self._make_dataframe(
            columns=CNF.clef_cols.default,
            qa_path=CNF.paths.clef_21_val_qa,
            imgs_path=CNF.paths.clef_21_val_imgs
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
        self.trainset = CLEFDataset(
            self.train_df, self.train_tfms, self.classes)
        self.valset = CLEFDataset(self.val_df, self.val_tfms, self.classes)
        # Make dataloaders
        self.dls = DataLoaders.from_dataset(
            trainset=self.trainset,
            train_bs=CNF.train.bs,
            valset=self.valset,
            val_bs=1
        )
    
    def _make_dataframe(self, columns, qa_path, imgs_path, is_main_df=False):
        df = load_qa_file(qa_filepath=qa_path, columns=columns)
        df['PATH'] = df.ID.apply(lambda x: f"{imgs_path}/{x}.jpg")
        if not is_main_df:
            df = pd.DataFrame([
                row for row in df.itertuples() if row.A in self.classes
            ]).drop("Index", axis=1)
        return df

    def check(self):
        print(f"Data augmentation:\n\t{self.train_tfms}\n\t{self.val_tfms}")
        print(f"# training samples = {self.train_df.shape}")
        print(f"# validation samples = {self.val_df.shape}")
        assert self.train_df.shape == (5435, 4)
        assert self.val_df.shape == (500, 4)
        assert self.train_df.A.nunique() <= self.n_classes
        assert self.val_df.A.nunique() <= self.n_classes
        assert len(self.trainset) == self.train_df.shape[0]
        assert len(self.valset) == self.val_df.shape[0]

        inputs, target = next(iter(self.dls.trainloader))
        print(f"data batch:")
        print(f"\timgs = {inputs.shape}")
        print(f"\tA = {target.shape}")
        print("data check: [green]PASSED[/green]")


class CLEFTrainer(VQATrainer):
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
                inputs, target, self.mixup
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
        assert inputs.dim(
        ) == 4, 'You need to provide a [B,C,H,W] image to flip'
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
    data = PrepareCLEFData(CNF)
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
    elif CNF.model.vnet_name == 'vgg16HGap':
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
    trainer = CLEFTrainer(dls=data.dls,
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
    EXP_NO = 1

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
    
    elif EXP_NO == 5:
        CNF.loss.smoothing = 0.1
        CNF.loss.fn = 'superloss'
        CNF.wandb_run_name += '+LabelSmoothing+SuperLoss'

    print(f"[cyan]Running Exp `{CNF.wandb_run_name}`...")
    run(CNF)
    print(f"[green]Done.")
