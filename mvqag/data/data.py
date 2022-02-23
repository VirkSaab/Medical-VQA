import math
import torch
import torchvision.transforms as T
from typing import Union, Dict, List, Any, Optional, Callable
from pandas import DataFrame
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

__all__ = [
    # Functions
    'vqa_collate_fn', 'get_data_transforms',

    # Classes
    'VQADataset', 'DataLoaders',
]

# ============================= FUNCTIONS ==============================


def vqa_collate_fn(batch):
    collated = {
        'inputs': {
            'V': [],  # Image
            'Q': {'input_ids': []},  # Question
        },
        'target': []  # Answers
    }
    # Images
    for sample in batch:
        collated['inputs']['V'].append(sample['inputs']['V'].unsqueeze(0))
    collated['inputs']['V'] = torch.cat(collated['inputs']['V'])
    # Questions
    if batch[0]['inputs']['Q'] is None:
        collated['inputs']['Q'] = None
    else:
        for sample in batch:
            for k, v in sample['inputs']['Q'].items():
                collated['inputs']['Q'][k].append(v)
        collated['inputs']['Q'] = {
            k: torch.cat(v) for k, v in collated['inputs']['Q'].items()
        }

    # Answers
    for sample in batch:
        collated['target'].append(sample['target'].unsqueeze(0))
    collated['target'] = torch.cat(collated['target'])
    # Info
    collated['info'] = [sample['info'] for sample in batch]
    return collated


def get_data_transforms(type: Union[None, str] = 'auto',
                        normalize: bool = True,
                        normalize_mean: list = [0.485, 0.456, 0.406],
                        normalize_std: list = [0.229, 0.224, 0.225],
                        resize_to: Optional[int] = None,
                        ) -> Dict[str, T.Compose]:
    """Get train and validation transforms
    Args:
        #TODO
    """
    train_tfms, val_tfms = [], []
    if resize_to is not None:
        train_tfms.append(T.Resize((resize_to+9, resize_to+9)))
        val_tfms.append(T.Resize((resize_to, resize_to)))

    if type == 'manual':
        if resize_to is not None:
            train_tfms += [
                T.RandomChoice([
                    T.RandomCrop(resize_to),
                    T.CenterCrop(resize_to),
                ])
            ]
        train_tfms += [
            T.RandomHorizontalFlip(),
            # T.RandomRotation(degrees=(0, 70)),
        ]
    elif type == 'auto':
        if resize_to is not None:
            train_tfms.append(T.RandomCrop(resize_to))
        train_tfms.append(T.AutoAugment())
    train_tfms.append(T.ToTensor())
    val_tfms.append(T.ToTensor())

    if normalize:
        norm = T.Normalize(normalize_mean, normalize_std)
        train_tfms += [norm]
        val_tfms += [norm]

    return T.Compose(train_tfms), T.Compose(val_tfms)


# ============================= CLASSES ==============================
class VQADataset(Dataset):
    def __init__(self,
                 df: DataFrame,
                 img_tfms: T.Compose,
                 classes: Union[None, List[str]],
                 tokenizer: Union[None, Callable] = None) -> None:
        """PyTorch Dataset abstract class wrapper

        Args:
            #TODO
        """
        super().__init__()

        self.df = df
        self.img_tfms = img_tfms
        self.classes = classes
        if self.classes is not None:
            self.n_classes = len(self.classes)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Visual
        img = Image.open(row.PATH).convert('RGB')
        if self.img_tfms is not None:
            img = self.img_tfms(img)

        # Question
        if self.tokenizer is None:
            question = None
        else:
            question = self.tokenizer.encode(row.Q)
            question = {
                k: torch.tensor(v).unsqueeze(0) 
                for k, v in question.items()
            }

        # Answer
        target = torch.tensor(self.ctol(row.A))

        return {
            'inputs': {'V': img, 'Q': question},
            'target': target,
            'info': row
        }

    def ctol(self, c) -> int:
        return self.classes.index(c)

    def ltoc(self, l) -> str:
        return self.classes[l]

    @property
    def cls_wts(self) -> torch.Tensor:
        """Class balancing weights for balanced training"""
        n_samples = [None]*self.n_classes
        for cls, count in self.df.A.value_counts().to_dict().items():
            if self.classes is not None:
                label = self.ctol[cls]
            else: 
                label = cls
            n_samples[label] = count
        return sum(n_samples) / torch.tensor(n_samples)  # Class weights


class DataLoaders:
    def __init__(self,
                 trainloader: Union[DataLoader, None] = None,
                 valloader: Union[DataLoader, None] = None,
                 testloader: Union[DataLoader, None] = None,
                 n_workers: int = 4) -> None:
        if trainloader:
            self.train_bs = trainloader.batch_size
            self.trainset = trainloader.dataset
            self.trainloader = trainloader
        if valloader:
            self.val_bs = valloader.batch_size
            self.valset = valloader.dataset
            self.valloader = valloader
        if testloader:
            self.test_bs = testloader.batch_size
            self.testset = testloader.dataset
            self.testloader = testloader
        self.n_workers = n_workers

    @classmethod
    def from_dataset(cls,
                     trainset: Optional[Dataset] = None,
                     train_bs: Optional[int] = None,
                     valset: Optional[Dataset] = None,
                     val_bs: Optional[int] = None,
                     testset: Optional[Dataset] = None,
                     test_bs: Optional[int] = None,
                     n_workers: Optional[int] = 4,
                     collate_fn: Optional[Callable] = None
                     ) -> Dict[str, DataLoader]:
        """Create dataloaders from dataset classes

        Args:
            #TODO
        """
        if trainset == valset == testset == None:
            raise ValueError("At least one set is required")
        trainloader, valloader, testloader = None, None, None
        trainset, train_bs = trainset, train_bs
        valset, val_bs = valset, val_bs
        testset, test_bs = testset, test_bs
        if trainset is not None:
            trainloader = DataLoader(
                dataset=trainset,
                batch_size=train_bs,
                shuffle=True,
                num_workers=n_workers,
                collate_fn=collate_fn
            )
        if valset is not None:
            valloader = DataLoader(
                dataset=valset,
                batch_size=val_bs,
                shuffle=False,
                num_workers=n_workers,
                collate_fn=collate_fn
            )
        if testset is not None:
            testloader = DataLoader(
                dataset=testset,
                batch_size=test_bs,
                shuffle=False,
                num_workers=n_workers,
                collate_fn=collate_fn
            )
        return cls(
            trainloader=trainloader,
            valloader=valloader,
            testloader=testloader,
            n_workers=n_workers
        )

    def collate_fn(self):
        return None

    @property
    def n_train_batches(self) -> int:
        if self.trainset is not None:
            return math.ceil(len(self.trainset) / self.train_bs)

    @property
    def n_val_batches(self) -> Union[None, int]:
        if self.valset is not None:
            return math.ceil(len(self.valset) / self.val_bs)

    @property
    def n_test_batches(self) -> Union[None, int]:
        if self.testset is not None:
            return math.ceil(len(self.testset) / self.test_bs)

    @classmethod
    def from_dicts(cls,
                   train_dict: Dict[str, Any],
                   val_dict: Optional[Dict[str, Any]] = None,
                   test_dict: Optional[Dict[str, Any]] = None):
        valset, val_bs = val_dict['ds'], val_dict['bs']
        testset, test_bs = test_dict['ds'], test_dict['bs']
        return cls(trainset=train_dict['ds'],  # DataSet
                   train_bs=train_dict['bs'],  # BatchSize
                   valset=valset,
                   val_bs=val_bs,
                   testset=testset,
                   test_bs=test_bs
                   )
