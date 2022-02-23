import torchmetrics
from torchmetrics.detection.map import MeanAveragePrecision


__all__ = ['get_metrics']


def get_metrics(n_classes: int) -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection({
        'AccuracyMicro': torchmetrics.Accuracy(num_classes=n_classes, average='micro'),
        # 'Accuracy': torchmetrics.Accuracy(num_classes=n_classes, average='macro'),
        # 'Precision': torchmetrics.Precision(num_classes=n_classes, average='macro'),
        # 'Recall': torchmetrics.Recall(num_classes=n_classes, average='macro'),
        # 'F1': torchmetrics.F1(num_classes=n_classes, average='macro'),
    })
