import torch
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
from typing import Dict, List
from torchcontrib.optim import SWA
from pathlib import Path
from mvqag import CNF_PATH
from mvqag.utils import load_yaml, load_json, get_recent_githash, load_qa_file
from mvqag.data import (
    VQADataset,
    DataLoaders,
    Tokenizer,
    vqa_collate_fn,
    generate_new_questions_dataframe
)
from mvqag.model import VQANet, SANNet
from mvqag.train import (
    get_device,
    get_metrics,
    LabelSmoothingCrossEntropyWithSuperLoss,
    LabelSmoothingCrossEntropy,
    SuperLoss,
    VQATrainer,
    TrainingLogger,
    Checkpointer,
    mixup_data_vqa,
    mixup_criterion_vqa,
)


class PrepareCLEF2020DataWithQ:
    def __init__(self, CNF) -> None:
        self.n_classes = CNF.data.n_classes
        self.QG = CNF.data.QG

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
        if self.n_classes == 330:
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
        training_dfs = [train19_df, val19_df, test19_df, train20_df]
        self.train_df = pd.concat(
            training_dfs, ignore_index=True
        ).reset_index(drop=True)

        self.val_df = self._make_dataframe(
            columns=CNF.clef_cols.default,
            qa_path=CNF.paths.clef_20_val_qa,
            imgs_path=CNF.paths.clef_20_val_imgs
        )

        # Augmentation
        self.train_tfms = T.Compose([
            T.Resize(size=(CNF.model.inp_size + 8, CNF.model.inp_size + 8)),
            # T.AutoAugment(),
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
        # Generate new questions for training dataset
        if CNF.data.QG:
            print(f"Before QG: # training samples = {self.train_df.shape[0]}")
            self.train_df = self._generate_questions(self.train_df,
                                                     CNF.task_keywords)
            print(f"After QG: # training samples = {self.train_df.shape[0]}")
        self.tokenizer = Tokenizer.from_list(
            self.train_df.Q.unique().tolist(),
            max_len=CNF.model.max_len
        )
        # Make dataset classes
        self.trainset = VQADataset(
            self.train_df, self.train_tfms, self.classes, self.tokenizer
        )
        self.valset = VQADataset(
            self.val_df, self.val_tfms, self.classes, self.tokenizer
        )
        # Make dataloaders
        self.dls = DataLoaders.from_dataset(
            trainset=self.trainset,
            train_bs=CNF.train.bs,
            valset=self.valset,
            val_bs=1,
            collate_fn=vqa_collate_fn
        )

    def _make_dataframe(self,
                        columns,
                        qa_path,
                        imgs_path,
                        is_main_df=False):
        df = load_qa_file(qa_filepath=qa_path, columns=columns)
        df['PATH'] = df.ID.apply(lambda x: f"{imgs_path}/{x}.jpg")
        if not is_main_df:
            df = pd.DataFrame([
                row for row in df.itertuples() if row.A in self.classes
            ]).drop("Index", axis=1)
        return df

    def _generate_questions(
        self,
        train_df: pd.DataFrame,
        task_keywords: Dict[str, List[str]]
    ) -> pd.DataFrame:
        new_train_df = generate_new_questions_dataframe(
            train_df, task_keywords
        )
        new_train_df = new_train_df[new_train_df.Task == 'abnormality']
        new_train_df = new_train_df[new_train_df.SubTask == 'categorical']
        return new_train_df

    def check(self):
        print(f"Data augmentation:\n\t{self.train_tfms}", end='')
        print(f"\n\t{self.val_tfms}")
        print(f"# training samples = {self.train_df.shape}")
        print(f"# validation samples = {self.val_df.shape}")
        if self.n_classes == 330:
            assert self.train_df.shape == (4963, 4)
            assert self.val_df.shape == (472, 4)
        elif self.n_classes == 332:
            if self.QG:
                assert self.train_df.shape == (39704, 6)
            else:
                assert self.train_df.shape == (6583, 4)
            assert self.val_df.shape == (500, 4)
        else:
            _errmsg = f"check not added for classes = {self.n_classes}"
            raise NotImplementedError(_errmsg)
        assert self.train_df.A.nunique() <= self.n_classes
        assert self.val_df.A.nunique() <= self.n_classes
        assert len(self.trainset) == self.train_df.shape[0]
        assert len(self.valset) == self.val_df.shape[0]

        batch = next(iter(self.dls.trainloader))
        print(f"data batch:")
        print(f"\tV = {batch['inputs']['V'].shape}")
        print(f"\tQ = {batch['inputs']['Q']['input_ids'].shape}")
        print(f"\tA = {batch['target'].shape}")
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
        target = batch['target'].to(self.device)
        V = batch['inputs']['V'].to(self.device)
        Q = {
            k: v.to(self.device) for k, v in batch['inputs']['Q'].items()
        }
        if self.mixup > 0.0:
            Q = Q['input_ids']
            mixed_v, a_a, a_b, q_a, q_b, lam = mixup_data_vqa(
                V, Q, target, alpha=self.mixup, use_cuda=True
            )
            output_a = self.net(mixed_v, {'input_ids': q_a})
            output_b = self.net(mixed_v, {'input_ids': q_b})
            loss = mixup_criterion_vqa(
                self.loss_fn, output_a, output_b, a_a, a_b, lam)
            output = (lam * output_a) + ((1 - lam) * output_b)
        else:
            output = self.net(V, Q)
            loss = self.loss_fn(output, target)
        # Backpropagation
        self.optm_fn.zero_grad()
        loss.backward()
        self.optm_fn.step()
        return loss, output, target

    def val_one_batch(self, batch):
        target = batch['target'].to(self.device)
        V = batch['inputs']['V'].to(self.device)
        Q = {
            k: v.to(self.device) for k, v in batch['inputs']['Q'].items()
        }
        output = self.net(V, Q)

        # test time augmentation
        assert V.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
        Vs_flip = torch.flip(V, [3])
        output_flip = self.net(Vs_flip, Q)
        output = (output + output_flip) / 2.0

        loss = self.loss_fn(output, target)
        return loss, output, target


def run(CNF: dict, dm=None, ret_model_and_dm: bool = False):
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
    if dm is None:
        dm = PrepareCLEF2020DataWithQ(CNF)
    dm.check()

    # ---------------------------------------- MODEL:
    model_name = f"{CNF.model.vnet_name}_{CNF.model.qnet_name}"
    print(f"Loading {model_name} model...", end=' ')
    if CNF.model.use_SAN:
        CNF.wandb_run_name += f"+{model_name}+SAN"
        model = SANNet(
            n_classes=CNF.data.n_classes,
            vnet_name=CNF.model.vnet_name,
            qnet_name=CNF.model.qnet_name,
            vocab_dim=dm.tokenizer.vocab_dim,
            emb_dim=CNF.model.emb_dim,
            vdp=CNF.model.vdp,
            qdp=CNF.model.qdp
        )
    else:
        CNF.wandb_run_name += f"+{model_name}"
        model = VQANet(
            n_classes=CNF.data.n_classes,
            vnet_name=CNF.model.vnet_name,
            qnet_name=CNF.model.qnet_name,
            vocab_dim=dm.tokenizer.vocab_dim,
            emb_dim=CNF.model.emb_dim,
            hid_dim=1024,
            bidirect=True,
            vdp=CNF.model.vdp,
            qdp=CNF.model.qdp
        )
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
    elif CNF.loss.fn.lower() == 'celsv2':
        loss_fn = LabelSmoothingCrossEntropy(classes=CNF.data.n_classes)
    else:
        raise NotImplementedError(f'Loss_fn {CNF.loss.fn} not supported.')
    print(f"Criterion = {CNF.loss.fn}")
    epoch_lrs = torch.optim.lr_scheduler.StepLR(optm_fn,
                                                step_size=20,
                                                gamma=0.60)
    logger = TrainingLogger(
        logs_dir=CNF.paths.logs_dir, config=CNF, run_name=CNF.wandb_run_name
    )
    checkpointer = Checkpointer(chkpts_dir=CNF.paths.chkpts_dir,
                                chkpt_of=[{'ValAccuracyMicro': 'max'}])
    if CNF.train.use_swa:  # Stochastic Weight Averaging
        optm_fn = SWA(optm_fn, swa_start=20, swa_freq=5, swa_lr=0.005)
    trainer = CLEFTrainer(dls=dm.dls,
                          net=model,
                          loss_fn=loss_fn,
                          optm_fn=optm_fn,
                          device=CNF.device,
                          metrics=get_metrics(CNF.data.n_classes),
                          logger=logger,
                          checkpointer=checkpointer,
                          epoch_lrs=epoch_lrs,
                          mixup=CNF.train.vqa_mixup
                          )
    if CNF.is_test_run:
        trainer.train(2, max_train_iters=5, max_val_iters=5)
    else:
        trainer.train(CNF.train.n_epochs)
    if CNF.train.use_swa:
        optm_fn.swap_swa_sgd()

    if ret_model_and_dm:
        return trainer.net, dm
        

if __name__ == '__main__':
    # * EXPERIMENT NUMBER
    EXP_NO = 2

    # * Baseline settings
    CNF = load_yaml(CNF_PATH)
    CNF.data.n_classes = 332
    CNF.data.QG = False  # Questions generation
    CNF.model.use_SAN = False  # If False, use multiplication fusion
    CNF.train.use_swa = False  # Stochastic Weight Averaging
    CNF.wandb_run_name = '/'.join(__file__.split('.')[0].split('/')[-2:])
    CNF.wandb_run_name = f"{EXP_NO}-{CNF.wandb_run_name}+332"

    CNF.is_test_run = False

    if EXP_NO == 1:
        # * EXP 1 - our baseline with SYSU curated dataset
        # Baseline experiment
        pass

    elif EXP_NO == 2:
        CNF.loss.fn = 'celsv2'
        CNF.train.vqa_mixup = 0.1
        CNF.wandb_run_name += '+VQAMixUp'

    elif EXP_NO == 3:
        CNF.loss.smoothing = 0.1
        CNF.train.vqa_mixup = 0.1
        CNF.wandb_run_name += '+VQAMixUp+LabelSmoothing'

    elif EXP_NO == 4:
        CNF.model.vnet_name = 'vgg16mixpool'
        CNF.loss.smoothing = 0.1
        CNF.train.vqa_mixup = 0.1
        CNF.wandb_run_name += '+VQAMixUp+LabelSmoothing'

    elif EXP_NO == 5:
        CNF.model.vnet_name = 'vgg16mixpool'
        CNF.loss.fn = 'lscesl'
        CNF.train.vqa_mixup = 0.1
        CNF.wandb_run_name += '+VQAMixup+LSCESL'

    elif EXP_NO == 6:
        CNF.train.use_swa = True
        CNF.wandb_run_name += '+SWA'

    elif EXP_NO == 7:
        CNF.train.use_swa = True
        CNF.train.vqa_mixup = 0.1
        CNF.wandb_run_name += '+SWA+VQAMixUp'

    elif EXP_NO == 8:
        CNF.data.QG = True
        CNF.train.vqa_mixup = 0.1
        CNF.loss.smoothing = 0.1
        CNF.wandb_run_name += '+QG+VQAMixUp+LabelSmoothing'

    elif EXP_NO == 9:
        CNF.data.QG = True
        CNF.model.use_SAN = True
        CNF.train.vqa_mixup = 0.1
        CNF.loss.smoothing = 0.1
        CNF.wandb_run_name += '+QG+VQAMixUp+LabelSmoothing'

    elif EXP_NO == 10:
        CNF.data.QG = True
        CNF.model.use_SAN = True
        CNF.train.vqa_mixup = 0.1
        CNF.loss.smoothing = 0.1
        CNF.loss.fn = 'lscesl'
        CNF.wandb_run_name += '+QG+VQAMixUp+LSCESL'

    elif EXP_NO == 11:
        CNF.data.QG = True
        CNF.model.vnet_name = 'vgg16mixpool'
        CNF.model.use_SAN = True
        CNF.train.vqa_mixup = 0.1
        CNF.loss.smoothing = 0.1
        CNF.loss.fn = 'lscesl'
        CNF.wandb_run_name += '+QG+VQAMixUp+LSCESL'

    elif EXP_NO == 12:
        CNF.data.QG = True
        CNF.model.vnet_name = 'vgg16mixpool'
        CNF.loss.fn = 'lscesl'
        CNF.train.vqa_mixup = 0.1
        CNF.wandb_run_name += '+QG+VQAMixup+LSCESL'

    elif EXP_NO == 13:
        CNF.model.vnet_name = 'vgg16mixpool'
        CNF.model.use_SAN = True
        CNF.train.vqa_mixup = 0.1
        CNF.loss.smoothing = 0.1
        CNF.wandb_run_name += '+VQAMixUp+LabelSmoothing'

    elif EXP_NO == 14:
        CNF.data.QG = True
        CNF.wandb_run_name += '+QG'

    elif EXP_NO == 15:
        CNF.model.use_SAN = True

    elif EXP_NO == 16:
        CNF.data.QG = True
        CNF.model.use_SAN = True
        CNF.loss.fn = 'celsv2'  # to run on older pytorch versions
        CNF.loss.smoothing = 0.1
        CNF.train.vqa_mixup = 0.1
        CNF.wandb_run_name += '+VQAMixUp+QG+LabelSmoothing'

        
    # Sanity check
    assert CNF.train.bs == 32
    run(CNF)
